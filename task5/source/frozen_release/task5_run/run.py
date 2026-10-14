"""Runnable evaluation entry points. Model execution is approval gated."""
import argparse
import gc
from pathlib import Path
import tempfile
from . import core


def verify_source(release):
    manifest = core.read_json(release / 'MANIFEST.json')
    for relative, expected in manifest.items():
        p = release / relative
        core.require(p.is_file() and not p.is_symlink() and core.file_sha(p) == expected,
                     'Evaluation release file differs: ' + relative)
    return core.file_sha(release / 'MANIFEST.json')


def inputs(project, release):
    b = core.read_json(release / 'bindings.json')
    for relative, expected in b['asset_sha256'].items():
        p = project / relative
        core.require(p.is_file() and not p.is_symlink() and core.file_sha(p) == expected,
                     'Fixed course input differs: ' + relative)
    rows = {d: core.read_rows(project / b['datasets'][d]) for d in core.DOMAINS}
    core.generation_schedule(rows)
    return b, rows


def load_responses(root, policy, domain, rows, expected_release_sha=None):
    folder = root / 'generations' / policy / domain
    core.require((folder / 'BINDING.json').is_file(), 'Missing generated responses.')
    journal = core.Journal(folder, core.read_json(folder / 'BINDING.json'))
    binding = core.read_json(folder / 'BINDING.json')
    core.require(binding.get('phase') == 'generation' and binding.get('policy') == policy
                 and binding.get('domain') == domain, 'Saved policy/domain binding differs.')
    if expected_release_sha is not None:
        core.require(binding['release_sha256'] == expected_release_sha,
                     'Generated responses come from a different evaluation release.')
    records = journal.records()
    core.require(len(records) == len(rows), 'Incomplete generated policy/domain responses.')
    for i, record in enumerate(records):
        p = record['payload']
        from task5_feedback.rlvr import exact_reward, extract_designated_final
        core.require(p['row_index'] == i and p['row_sha256'] == core.checksum(rows[i])
                     and p['policy'] == policy and p['domain'] == domain
                     and record['rng_after'] is not None, 'Saved generation row binding differs.')
        core.require(p['exact_reward'] == exact_reward(p['response'], rows[i]['gold_final'])
                     and p['format_compliant'] == (extract_designated_final(p['response']) is not None),
                     'Saved generated-response verifier metrics differ.')
    return [r['payload'] for r in records], journal


def validate_policy(model, torch):
    core.require(not model.training and not getattr(model, 'is_quantized', False),
                 'Policy mode/precision differs from released nonquantized evaluation.')
    core.require(all(p.device.type == 'cuda' for p in model.parameters()), 'Policy offload differs.')
    core.require(all(p.dtype == torch.float16 for name, p in model.named_parameters()
                     if 'lora_' not in name), 'Policy base precision differs from course loader.')


def verify_adapter_loaded(model, adapter):
    import torch
    from safetensors.torch import load_file
    saved = load_file(str(adapter / 'adapter_model.safetensors'), device='cpu')
    actual = model.state_dict()
    for key, value in saved.items():
        parts = key.split('.')
        require_index = next((i for i, p in enumerate(parts) if p in ('lora_A', 'lora_B')), None)
        core.require(require_index is not None, 'Unexpected supplied adapter tensor key.')
        parts.insert(require_index + 1, 'default')
        target = '.'.join(parts)
        core.require(target in actual, 'Supplied adapter tensor missing from loaded policy.')
        loaded = actual[target].detach().cpu()
        core.require(torch.equal(loaded, value.to(dtype=loaded.dtype)), 'Loaded adapter tensor differs.')


def generate(project, release):
    from .runtime import execution_gate
    hardware = execution_gate(project, release, 'generation')
    release_sha = verify_source(release)
    b, rows = inputs(project, release)
    import torch
    from common.models import load_policy, load_tokenizer
    from common.logging_utils import set_seed
    from task5_feedback.rlvr import exact_reward, extract_designated_final
    from .inference import TimedModel, capture_rng, restore_rng, generate_one, effective_generation, prompt_preview
    root = project / 'task5/experiment_v1/evaluation_v1'
    with core.process_lock(root):
        for policy in core.POLICIES:
            snapshot = project / b['snapshots']['policy']['path']
            adapter = None if policy == 'sft' else project / b['adapters'][policy]
            model = load_policy({'base_model': str(snapshot), 'dtype': 'float16'},
                                adapter_path=str(adapter) if adapter else None, trainable=False)
            validate_policy(model, torch)
            for p in model.parameters():
                p.requires_grad_(False)
            if adapter:
                verify_adapter_loaded(model, adapter)
            tokenizer = load_tokenizer(str(snapshot))
            observed = TimedModel(model, int(model.config.max_position_embeddings))
            settings = effective_generation(model, tokenizer)
            for domain in core.DOMAINS:
                # Check every full prompt before the first generation in this run.
                for row in rows[domain]:
                    from common.data import prompt_messages
                    prompt_preview(tokenizer, prompt_messages(row), observed.context_limit)
                binding = {'phase': 'generation', 'policy': policy, 'domain': domain,
                           'release_sha256': release_sha, 'hardware': hardware, 'settings': settings,
                           'dataset_sha256': b['asset_sha256'][b['datasets'][domain]],
                           'row_sha256_in_order': [core.checksum(r) for r in rows[domain]],
                           'seed': 6304}
                journal = core.Journal(root / 'generations' / policy / domain, binding)
                saved = journal.records()
                core.require(len(saved) <= len(rows[domain]), 'Too many completed generation units.')
                for i, record in enumerate(saved):
                    core.require(record['payload']['row_sha256'] == core.checksum(rows[domain][i])
                                 and record['payload']['row_index'] == i and record['rng_after'] is not None,
                                 'Saved generation order or RNG checkpoint differs.')
                set_seed(6304)
                if saved:
                    restore_rng(saved[-1]['rng_after'])
                for i in range(len(saved), len(rows[domain])):
                    row = rows[domain][i]
                    response = generate_one(observed, tokenizer, row)
                    response.update({'policy': policy, 'domain': domain, 'row_index': i,
                        'row_sha256': core.checksum(row), 'source_index': row['source_index'],
                        'extracted_final': extract_designated_final(response['response']),
                        'exact_reward': exact_reward(response['response'], row['gold_final']),
                        'format_compliant': extract_designated_final(response['response']) is not None})
                    journal.append(response, capture_rng())
                    print(f'{policy}/{domain}: {i+1}/{len(rows[domain])} answers saved', flush=True)
            del observed, model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()


def comparison_specs(root, rows, diagnostic_rows, variants, release_sha=None):
    responses = {(p, d): load_responses(root, p, d, rows[d], release_sha)[0]
                 for p in core.POLICIES for d in core.DOMAINS}
    specs = []
    for policy in ('rlvr', 'rlaif'):
        for domain in core.DOMAINS:
            for i, row in enumerate(rows[domain]):
                a, b = responses[policy, domain][i], responses['sft', domain][i]
                specs.append({'unit_id': f'{policy}/{domain}/{i}', 'kind': 'policy',
                              'policy': policy, 'domain': domain, 'row_index': i,
                              'problem': row['question'], 'a': a['response'], 'b': b['response'],
                              'verifier_preference': core.verifier_preference(a['exact_reward'], b['exact_reward'])})
    groups = {}
    for row in diagnostic_rows:
        groups.setdefault(str(row['problem_id']), {})[row['variant_type']] = row
    from task5_feedback.rlvr import exact_reward
    for pid, group in groups.items():
        clean = group['clean_correct']
        for variant in variants[1:]:
            modified = group[variant]
            specs.append({'unit_id': f'diagnostic/{pid}/{variant}', 'kind': 'diagnostic',
                'problem_id': pid, 'variant': variant, 'problem': clean['question'],
                'a': clean['response'], 'b': modified['response'],
                'verifier_preference': core.verifier_preference(
                    exact_reward(clean['response'], clean['gold_final']),
                    exact_reward(modified['response'], modified['gold_final']))})
    core.require(len(specs) == 880 and len({s['unit_id'] for s in specs}) == 880,
                 'Expected 800 policy plus 80 diagnostic comparison events.')
    return specs, responses


def judge(project, release):
    from .runtime import execution_gate
    hardware = execution_gate(project, release, 'ai_judging')
    release_sha = verify_source(release)
    b, rows = inputs(project, release)
    from .diagnostics import VARIANTS, group_rows
    diagnostic_rows = core.read_rows(project / b['diagnostics'])
    group_rows(diagnostic_rows)
    from .inference import load_judge, judge_one, capture_rng, restore_rng
    root = project / 'task5/experiment_v1/evaluation_v1'
    with core.process_lock(root):
        specs, responses = comparison_specs(root, rows, diagnostic_rows, VARIANTS, release_sha)
        binding = {'phase': 'ai_judging', 'release_sha256': release_sha, 'hardware': hardware,
                   'ordered_spec_sha256': [core.checksum(s) for s in specs],
                   'generated_response_sha256': core.checksum({p + '/' + d: r for (p, d), r in responses.items()}),
                   'canonical_judge': 'Qwen/Qwen2.5-3B-Instruct', 'max_new_tokens': 4}
        journal = core.Journal(root / 'judgments', binding)
        saved = journal.records()
        core.require(len(saved) <= len(specs), 'Too many saved judge units.')
        for i, record in enumerate(saved):
            core.require(record['payload']['spec_sha256'] == core.checksum(specs[i]),
                         'Saved judge unit differs from current comparison inputs.')
        with tempfile.TemporaryDirectory(prefix='task5_fresh_judge_cache_') as temporary:
            fixed, captured = load_judge(project / b['snapshots']['judge']['path'],
                                        Path(temporary) / 'cache.json', saved)
            if saved:
                restore_rng(saved[-1]['rng_after'])
            for i in range(len(saved), len(specs)):
                spec = specs[i]
                result = judge_one(fixed, captured, spec['problem'], spec['a'], spec['b'])
                result.update({k: v for k, v in spec.items() if k not in ('problem', 'a', 'b')})
                result['spec_sha256'] = core.checksum(spec)
                journal.append(result, capture_rng())
                print(f'AI comparisons: {i+1}/880 saved', flush=True)


def aggregate(project, release):
    release_sha = verify_source(release)
    b, rows = inputs(project, release)
    root = project / 'task5/experiment_v1/evaluation_v1'
    from .diagnostics import VARIANTS, group_rows
    diagnostics = core.read_rows(project / b['diagnostics'])
    group_rows(diagnostics)
    specs, responses = comparison_specs(root, rows, diagnostics, VARIANTS, release_sha)
    binding = core.read_json(root / 'judgments/BINDING.json')
    records = core.Journal(root / 'judgments', binding).records()
    core.require(len(records) == 880 and binding['ordered_spec_sha256'] == [core.checksum(s) for s in specs],
                 'Judgment set is incomplete or comparison inputs changed.')
    judgments = [r['payload'] for r in records]
    for i, record in enumerate(judgments):
        core.require(record['spec_sha256'] == core.checksum(specs[i]), 'Judgment row inputs differ.')
        key = core.judge_key(specs[i]['problem'], specs[i]['a'], specs[i]['b'])
        core.require(record['judge_key'] == key and record['swapped'] == (int(key[:8], 16) % 2 == 1)
                     and record['verifier_preference'] == specs[i]['verifier_preference'],
                     'Saved judge orientation or exact preference differs.')
        expected, ambiguous = core.decode_preference(record['raw_judge_output'], record['swapped'])
        core.require(record['ai_preference'] == expected and record['unparseable'] == ambiguous,
                     'Saved raw judge output and preference disagree.')
    metrics = {}
    for policy in core.POLICIES:
        metrics[policy] = {}
        for domain in core.DOMAINS:
            pairs = [p for p in judgments if p['kind'] == 'policy' and p['policy'] == policy and p['domain'] == domain]
            metrics[policy][domain] = core.policy_summary(responses[policy, domain], pairs)
        metrics[policy]['transfer_drop'] = core.transfer_drop(metrics[policy]['gsm'], metrics[policy]['transfer'])
    result = {'policies': metrics, 'diagnostics': core.diagnostic_summary(
        [p for p in judgments if p['kind'] == 'diagnostic'], VARIANTS),
        'ai_comparison_events': 880, 'ai_model_generate_calls': sum(not p['cache_hit'] for p in judgments),
        'ai_inference_seconds': sum(p['judge_seconds'] for p in judgments),
        'ai_inference_output_tokens': sum(p['judge_output_tokens'] for p in judgments if not p['cache_hit']),
        'release_sha256': core.file_sha(release / 'MANIFEST.json'),
        'qualitative_evidence_pending': True}
    destination = root / 'aggregate' / (core.checksum(result) + '.json')
    if destination.exists():
        core.require(core.read_json(destination) == result, 'Existing aggregate differs.')
    else:
        core.write_once(destination, result)
    print('Saved aggregate:', destination)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['generate', 'judge', 'aggregate'])
    ap.add_argument('--project', type=Path, default=Path('/content/drive/MyDrive/ATML-Assignment-2'))
    args = ap.parse_args()
    release = Path(__file__).resolve().parents[1]
    {'generate': generate, 'judge': judge, 'aggregate': aggregate}[args.phase](args.project, release)


if __name__ == '__main__':
    main()
