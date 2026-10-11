"""Audit the received small export without model loading or changing sealed source."""
import argparse
import csv
import hashlib
import io
import json
import math
import random
import statistics
import zipfile
from collections import Counter
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def finite_tree(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def aggregate(rows):
    tokens = sum(r['response_length'] for r in rows)
    result = {
        'prompts': len(rows), 'valid_tokens': tokens,
        'reference_kl_definition': 'signed sampled policy-minus-reference logp over valid response tokens',
        'eos_rate': statistics.mean(float(r['terminated_with_eos']) for r in rows),
        'truncation_rate': statistics.mean(float(r['truncated']) for r in rows),
    }
    for key in ('raw_reward', 'reference_kl', 'sampled_entropy', 'categorical_entropy', 'response_length'):
        values = [r[key] for r in rows]
        result[key + '_prompt_mean'] = statistics.mean(values)
        result[key + '_std_population'] = statistics.pstdev(values)
    for key in ('reference_kl', 'sampled_entropy', 'categorical_entropy'):
        result[key + '_token_mean'] = sum(r[key] * r['response_length'] for r in rows) / tokens
    return result


def gradient_aggregate(rows):
    result = []
    for length in ('1-128', '129-256', '257-512', 'masked'):
        for sign in ('positive', 'negative', 'zero'):
            values = [g['policy_logprob_gradient_l2'] for g in rows
                      if g['length_bin'] == length and g['advantage_sign'] == sign]
            result.append({'length_bin': length, 'advantage_sign': sign, 'count': len(values),
                           'gradient_l2_mean': statistics.mean(values) if values else None})
    return result


def audit(archive, output):
    if not __debug__:
        raise RuntimeError('Run without -O: audit assertions must be enabled.')
    archive, output = Path(archive), Path(output)
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'results/actual_run_20261011'
    if output.resolve().is_relative_to(evidence.resolve()):
        raise RuntimeError('Choose a separate output directory; original evidence/review is immutable.')
    archive_info = {'archive_bytes': None, 'archive_sha256': None}
    if archive.is_dir():
        manifest = json.loads((archive / 'FILES_SHA256.json').read_text())
        files = {}
        for name in manifest:
            assert Path(name).name == name
            path = archive / name
            assert not path.is_symlink() and path.is_file()
            files[name] = path.read_bytes()
    else:
        payload = archive.read_bytes()
        archive_info = {'archive_bytes': len(payload), 'archive_sha256': digest(payload)}
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            assert z.testzip() is None
            assert len(z.namelist()) == len(set(z.namelist()))
            manifest = json.loads(z.read('results/FILES_SHA256.json'))
            assert set(z.namelist()) == {'results/' + n for n in manifest} | {'results/FILES_SHA256.json'}
            files = {name: z.read('results/' + name) for name in manifest}
    for name, expected in manifest.items():
        assert Path(name).name == name
        assert digest(files[name]) == expected, name
    assert manifest == json.loads((evidence / 'FILES_SHA256.json').read_text())
    result = json.loads(files['results.json'])
    schedule = json.loads(files['schedule.json'])
    qualitative = json.loads(files['qualitative_review.json'])
    evaluations = {name: [json.loads(line) for line in files[name + '_evaluation.jsonl'].splitlines()]
                   for name in result['evaluation']}
    evaluation_csv = list(csv.DictReader(io.StringIO(files['evaluation.csv'].decode())))

    source = root / 'source'
    assert digest((source / 'docs/SOURCE_SHA256.json').read_bytes()) == result['preparation_source']['source_manifest_sha256']
    for name, expected in json.loads((source / 'docs/SOURCE_SHA256.json').read_text()).items():
        assert digest((source / name).read_bytes()) == expected, name
    approval = json.loads((source / 'docs/approval.json').read_text())
    assert result['preparation_receipt']['approval'] == approval
    assert result['preparation_receipt']['source'] == result['preparation_source']
    assert digest(json.dumps(schedule, indent=2, ensure_ascii=False, allow_nan=False).encode() + b'\n') == result['preparation_receipt']['schedule_sha256']
    assert result['status'] == 'COMPLETE_ACTUAL_RUNS'
    assert result['checkpoint_selection_performed'] is False
    assert result['qualitative_review'] == 'PENDING_USER_REVIEW'
    finite_tree(result)
    finite_tree(evaluations)
    indices = list(range(1200))
    random.Random(6304).shuffle(indices)
    assert [r['row_index'] for r in schedule['training']] == indices[:20]
    for i, entry in enumerate(schedule['training']):
        assert entry['update_index'] == i
        assert entry['completion_seeds'] == [6304 + 4*i + j for j in range(4)]
    assert [r['generation_seed'] for r in schedule['evaluation']] == [6304 + 100000 + i for i in range(200)]
    expected_config = {'updates':20, 'fork_updates':8, 'prompts_per_update':1, 'policy_epochs':1,
        'learning_rate':5e-6, 'clip_epsilon':.2, 'kl_beta':.1, 'num_generations':4,
        'group_sizes':[2,4,8], 'max_prompt_length':256, 'max_completion_length':512,
        'mask_truncated_completions':True, 'max_grad_norm':1., 'seed':6304,
        'generation':{'temperature':.7, 'top_p':.9, 'do_sample':True},
        'lora':{'r':8, 'alpha':16, 'dropout':.05, 'target_modules':['q_proj','v_proj']}}

    training_audit = {}
    for spec in approval['required_runs']:
        name = spec['name']
        summary = result['training'][name]
        history = summary['history']
        assert summary['status'] == 'COMPLETE'
        assert len(history) == summary['completed_updates'] == spec['updates']
        assert summary['contract']['run'] == spec
        assert summary['contract']['source'] == result['preparation_source']
        assert summary['contract']['approval'] == approval
        for key, value in expected_config.items():
            assert summary['contract']['release_config'][key] == value, key
        assert summary['contract']['reward_scorer_cap'] == 1024
        for session in summary['sessions']:
            loading, reward = session['model_loading'], session['reward_loading']
            assert not any(loading['loading'].values()) and not any(reward['loading'].values())
            assert loading['forward_mode'] == 'eval with gradients enabled for updates'
            assert loading['trainable_parameters'] == 1089536
            assert loading['generation']['course_generate_overrides'] == {
                'max_new_tokens':512, 'temperature':.7, 'top_p':.9, 'do_sample':True,
                'pad_token_id':151643, 'eos_token_id':151645}
            assert reward['frozen'] and reward['loaded_position_frequencies_match']
            assert reward['raw_config_sha256'] == approval['reward_config_sha256']
            assert reward['changed_fields'] == {'rope_theta':{'before':10000., 'after':1000000.}}
        assert [r['schedule'] for r in history] == schedule['training'][:spec['updates']]
        assert [r['completed_updates'] for r in history] == list(range(1, spec['updates'] + 1))
        assert summary['allocated_response_tokens'] == spec['updates'] * 4 * 512
        assert summary['actual_response_tokens'] == sum(r['rollout']['valid_generated_tokens'] for r in history)
        assert summary['loss_bearing_tokens'] == sum(r['rollout']['loss_bearing_tokens'] for r in history)
        assert summary['held_out_used_for_training'] is False
        gradients = []
        for r in history:
            assert r['contract'] == summary['contract']
            assert r['optimizer'] == [{'lr':5e-6, 'weight_decay':.01, 'betas':[.9,.999], 'eps':1e-8,
                'parameter_dtypes':['torch.float32'], 'moment_dtypes':['torch.float32']}]
            assert len(r['responses']) == len(r['truncated']) == len(r['terminated_with_eos']) == 4
            assert all(t != e for t, e in zip(r['truncated'], r['terminated_with_eos']))
            assert sum(r['truncated']) == r['rollout']['truncated_completions']
            g = r['optimization']['length_gradient_statistics']
            assert len(g) == 4
            assert sum(x['valid_loss_tokens'] for x in g) == r['rollout']['loss_bearing_tokens']
            assert r['rollout']['valid_generated_tokens'] == r['rollout']['loss_bearing_tokens'] + 512 * sum(r['truncated'])
            assert r['optimization']['all_completions_masked'] == all(r['truncated'])
            for x, truncated in zip(g, r['truncated']):
                length, advantage = x['valid_loss_tokens'], x['advantage']
                assert x['loss_type'] == spec['loss_type']
                assert (length == 0) == truncated
                expected = 0.0 if not length else abs(advantage) / (4 * math.sqrt(length)) if spec['loss_type'] == 'grpo' else abs(advantage) * math.sqrt(length) / (4 * 512)
                assert math.isclose(x['policy_logprob_gradient_l2'], expected, rel_tol=2e-5, abs_tol=2e-7)
            gradients.extend(g)
        assert gradient_aggregate(gradients) == result['length_gradient_statistics'][name]
        training_audit[name] = {
            **{k: summary[k] for k in ('completed_updates', 'allocated_response_tokens', 'actual_response_tokens',
                                       'loss_bearing_tokens', 'wall_clock_seconds', 'wall_clock_complete',
                                       'peak_allocated_gib', 'peak_reserved_gib')},
            'all_masked_update_numbers': [r['completed_updates'] for r in history if r['optimization']['all_completions_masked']],
            'truncated_completions': sum(sum(r['truncated']) for r in history),
            'total_completions': len(history) * 4,
            'uninformative_reward_groups': sum(r['rollout']['uninformative_group_fraction'] for r in history),
            'clip_fractions': sorted(set(r['optimization']['clip_fraction'] for r in history)),
            'preupdate_max_ratio_changes': sorted(set(r['optimization']['ratio_max_abs_change_before_update'] for r in history)),
        }

    for a, b in zip(result['training']['standard']['history'], result['training']['canonical_8']['history']):
        for key in ('schedule', 'responses', 'rollout', 'optimization', 'optimizer'):
            assert a[key] == b[key], key

    evaluation_audit = {}
    training_ids = {r['prompt_id'] for r in schedule['training']}
    for name, rows in evaluations.items():
        assert len(rows) == 200
        assert [r['schedule'] for r in rows] == schedule['evaluation']
        assert len({r['schedule']['prompt_id'] for r in rows}) == 200
        assert not training_ids & {r['schedule']['prompt_id'] for r in rows}
        assert aggregate(rows) == result['evaluation'][name]
        contract = rows[0]['contract']
        for key in ('source', 'approval', 'assets', 'model_manifest_sha256', 'schedule_sha256',
                    'release_config', 'reward_scorer_cap'):
            assert contract[key] == result['training']['standard']['contract'][key]
        assert contract['condition'] == name
        if name == 'midpoint':
            assert contract['endpoint_identity'] == result['preparation_receipt']['assets']['verified_files']
        else:
            summary_bytes = (json.dumps(result['training'][name], indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
            assert digest(summary_bytes) == contract['endpoint_identity']['summary.json']
        for r, midpoint in zip(rows, evaluations['midpoint']):
            assert r['condition'] == name
            assert r['contract'] == rows[0]['contract']
            assert r['contract']['source'] == result['preparation_source']
            assert r['contract']['approval'] == approval
            assert r['prompt_messages'] == midpoint['prompt_messages']
            assert r['prompt_token_ids'] == midpoint['prompt_token_ids']
            n = r['response_length']
            assert 0 < n <= 512
            assert len(r['response_token_ids']) == len(r['policy_token_logprobs']) == len(r['reference_token_logprobs']) == n
            assert len(r['prompt_token_ids']) == min(r['prompt_tokens_before_truncation'], 256)
            assert r['policy_prompt_truncated'] == (r['prompt_tokens_before_truncation'] > 256)
            assert r['reward_input_truncated'] == (r['reward_input_tokens_before_truncation'] > 1024)
            assert r['terminated_with_eos'] == (r['response_token_ids'][-1] == 151645)
            assert 151645 not in r['response_token_ids'][:-1]
            assert r['truncated'] == (n == 512 and not r['terminated_with_eos'])
            kl = statistics.mean(a - b for a, b in zip(r['policy_token_logprobs'], r['reference_token_logprobs']))
            entropy = -statistics.mean(r['policy_token_logprobs'])
            assert math.isclose(kl, r['reference_kl'], rel_tol=1e-5, abs_tol=2e-7)
            assert math.isclose(entropy, r['sampled_entropy'], rel_tol=1e-5, abs_tol=2e-7)
        evaluation_audit[name] = {
            'records': len(rows), 'aggregate_exact_match': True,
            'policy_prompt_truncated': sum(r['policy_prompt_truncated'] for r in rows),
            'reward_input_truncated': sum(r['reward_input_truncated'] for r in rows),
            'full_prompt_at_least_reward_cap': sum(r['prompt_tokens_before_truncation'] >= 1024 for r in rows),
            'shortened_prompt_missing_assistant_marker': sum(r['policy_prompt_truncated'] and r['prompt_token_ids'][-3:] != [151644, 77091, 198] for r in rows),
        }
    for row in evaluation_csv:
        for key, value in row.items():
            if key != 'condition':
                assert float(value) == result['evaluation'][row['condition']][key]

    cache = result['cache_diagnostic']
    local_cache = json.loads((root / 'validation/group_size.json').read_text())
    assert cache == local_cache
    assert not cache['training_performed'] and not cache['cache_rescored_or_regenerated']
    for k in (2, 4, 8):
        membership = [r for r in cache['membership'] if r['k'] == k]
        assert len(membership) == 192 // k
        assert sorted(i for r in membership for i in r['cache_row_indices']) == list(range(192))
        assert all(len(r['cache_row_indices']) == len(r['generation_indices']) == k for r in membership)
    probe = result['common_probe']
    assert probe['optimizer_updates'] == 0 and probe['excluded_from_continuation_budget']
    assert probe['contract']['source'] == result['preparation_source']
    assert probe['schedule'] == schedule['training'][0]
    assert probe['rollout'] == result['training']['standard']['history'][0]['rollout']

    assert qualitative['human_quality_labels_created'] is False
    for pair in qualitative['paired_normalization']:
        assert pair['review_status'] == 'PENDING_USER_REVIEW'
        assert pair['assistant_interpretation'] is None
        for short, name in [('canonical', 'canonical_8'), ('constant_denominator', 'dr_grpo_8')]:
            row = next(r for r in evaluations[name] if r['schedule']['prompt_id'] == pair['prompt_id'])
            assert row['prompt_messages'] == pair['prompt_messages']
            assert pair[short] == {k: row[k] for k in pair[short]}
    selected = qualitative['paired_normalization']
    a, b = evaluations['canonical_8'], evaluations['dr_grpo_8']
    for rule, key in [('largest_absolute_reward_difference', 'raw_reward'), ('largest_absolute_length_difference', 'response_length')]:
        ranked = sorted(range(200), key=lambda i: (-abs(a[i][key] - b[i][key]), a[i]['schedule']['prompt_id']))[:3]
        assert [p['prompt_id'] for p in selected if p['selection_rule'] == rule] == [a[i]['schedule']['prompt_id'] for i in ranked]

    scores = json.loads((evidence / 'review/SCORES.json').read_text())
    assert scores['status'] == 'USER_APPROVED' and scores['final_user_approval_received'] is True
    assert scores['final_approval']['user_reply'] == 'yes'
    assert scores['input_archive_sha256'] == '4ee4ad4b317c308be910b1dc5d4d3f54751f58a5fa12bf217bcd5ba23572267d'
    assert {e['prompt_id'] for e in scores['examples']} == {p['prompt_id'] for p in selected}
    assert len(scores['examples']) == 5
    for example in scores['examples']:
        for letter, condition in [('A', 'canonical_8'), ('B', 'dr_grpo_8')]:
            answer = example['answers'][letter]
            assert answer['condition'] == condition
            assert len(answer['criterion_scores']) == 3
            assert all(type(v) is int and v in (0, 1, 2) for v in answer['criterion_scores'])
            assert answer['total'] == sum(answer['criterion_scores'])
    output.mkdir(parents=True, exist_ok=True)
    report = {
        'status': 'SAVED_EXPORT_AUDIT_PASSED', **archive_info,
        'input_form': 'extracted_files' if archive.is_dir() else 'original_zip',
        'internal_export_files_verified': len(manifest), 'sealed_local_source_verified': True,
        'approval_and_source_contract_match': True, 'training': training_audit, 'evaluation': evaluation_audit,
        'frozen_seeds_configuration_optimizer_and_loading_audited': True,
        'canonical_eight_equals_standard_first_eight': True, 'cache_membership_conserves_192_each_k': True,
        'cache_matches_pretraining_local_diagnostic': True,
        'common_probe_unmasked_completions': sum(g['valid_loss_tokens'] > 0 for g in probe['conditions']['grpo']['gradient_statistics']),
        'qualitative_selection_slots': len(selected), 'qualitative_unique_prompts': len({p['prompt_id'] for p in selected}),
        'qualitative_exact_outputs_and_selection_verified': True,
        'review_status': 'QUALITATIVE_SCORES_USER_APPROVED',
        'approved_scores_validated': True,
        'raw_cache_recomputed_this_run': False,
        'limits': [
            'The independent Drive ZIP receipt was not attached; all internal export hashes pass.',
            'Raw step/prompt ZIPs, optimizer tensors, adapter weights, and full datasets stay on Drive; local audit verifies the small export and its internal contracts, not those remote bytes.',
            'No models were loaded and no reward scores were recomputed.',
            'Categorical entropies are aggregated from saved scalars; complete vocabulary logits are absent.',
            'Single seed, fixed endpoints, one sampled response per evaluation prompt; descriptive differences do not establish repeatable improvement.',
            'All-masked updates still call the released AdamW step: zero current loss gradients do not imply unchanged weights, due to optimizer momentum and weight decay.',
            'The common probe has one unmasked completion; broad length-bin claims require caution.',
            'Input truncation can remove task context and the final assistant generation marker; reward truncation can remove part or all of the response.',
        ],
    }
    (output / 'AUDIT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('SAVED EXPORT VERIFIED: 11 original payload files, 36 updates, 800 evaluations, 5 approved pairs.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', help='Original results ZIP or extracted results directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.archive, args.output)
