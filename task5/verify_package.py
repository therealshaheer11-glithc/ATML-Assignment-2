"""CPU verification/reproduction of frozen Task 5 results; standard library only."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
RELEASE_SHA = 'd6dd999e7fdf241c7136ee13f46babdf13338fa6dc12ff9046d2ea6a2e89fd87'
PRIVATE_SHA = '8bace113620a0080808a50df9b9ca296e881cb7ac430c118b6c73f339c030067'
EXPORT_NAMES = ('POLICY_RESULTS.csv', 'TRANSFER_RESULTS.csv', 'VERIFIER_AI_CONFUSION.csv',
                'DIAGNOSTIC_RESULTS.csv', 'DIAGNOSTIC_SENSITIVITY.json', 'INFERENCE_COUNTS_AND_TIME.json')
RESPONSE_FIELDS = ('policy', 'domain', 'row_index', 'source_index', 'row_sha256',
    'exact_reward', 'extracted_final', 'format_compliant', 'response_tokens',
    'prompt_tokens', 'prompt_token_ids_sha256', 'generation_seconds',
    'terminated_with_eos', 'truncated')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(',', ':'), allow_nan=False) + '\n').encode()


def read(name):
    return json.loads((ROOT / name).read_bytes())


def read_lines(name):
    return [json.loads(line) for line in (ROOT / name).read_bytes().splitlines()]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def project_response(r):
    return {**{k: r[k] for k in RESPONSE_FIELDS},
            'response_sha256': sha(r['response'].encode()),
            'generated_token_ids_sha256': sha(canonical(r['generated_token_ids']))}


def project_judgment(c):
    return {k: v for k, v in c.items() if k not in ('problem', 'a', 'b')}


def csv_bytes(rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def verify_hashes():
    manifest = read('MANIFEST.json')
    require(manifest['format'] == 'ATML_TASK5_PUBLIC_PACKAGE_V1', 'Wrong package manifest.')
    for name, value in manifest['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe manifest path.')
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), 'Missing/linked package file: ' + name)
        data = path.read_bytes()
        require(sha(data) == value['sha256'] and len(data) == value['bytes'],
                'Package file changed: ' + name)
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p != ROOT / 'MANIFEST.json'
              and p.name != '.DS_Store'}
    require(actual == set(manifest['files']), 'Unexpected or omitted package files.')
    release = ROOT / 'source/frozen_release'
    require(sha((release / 'MANIFEST.json').read_bytes()) == RELEASE_SHA, 'Frozen manifest changed.')
    for name, digest in json.loads((release / 'MANIFEST.json').read_bytes()).items():
        require(sha((release / name).read_bytes()) == digest, 'Frozen source changed: ' + name)
    return manifest


def export_tables(result, prior, core):
    # Same column ordering and definitions as the executed export_results helper.
    policy_rows, transfer_rows, agreement_rows, diagnostic_rows = [], [], [], []
    for policy in core.POLICIES:
        for domain in core.DOMAINS:
            value = result['policies'][policy][domain]
            pair = value['ai_comparison']
            row = {'policy': policy, 'domain': domain, 'responses': value['responses'],
                'exact_accuracy': value['exact_accuracy'], 'format_compliance': value['format_compliance'],
                'response_tokens_mean': value['length_mean'], 'response_tokens_population_sd': value['length_population_sd'],
                'truncated_responses': value['truncated_responses'], 'generation_seconds': value['generation_seconds'],
                'generated_tokens': value['generated_tokens'],
                'generated_tokens_per_generation_second': value['generated_tokens_per_generation_second'],
                'ai_reference': 'SFT reference baseline' if policy == 'sft' else 'same-domain SFT',
                'ai_pairs': pair['pairs'] if pair else None, 'ai_win_rate_half_ties': pair['ai_win_rate'] if pair else None,
                'ai_wins': pair['wins'] if pair else None, 'ai_ties': pair['ties'] if pair else None,
                'ai_losses': pair['losses'] if pair else None,
                'verifier_judge_agreement': pair['verifier_judge_agreement'] if pair else None,
                'unparseable_ai_outputs': pair['unparseable_ai_outputs'] if pair else None,
                'decisive_verifier_pairs': pair['decisive_verifier_pairs'] if pair else None,
                'secondary_agreement_decisive_verifier': pair['secondary_agreement_decisive_verifier'] if pair else None}
            policy_rows.append(row)
            if pair:
                for v, cells in pair['verifier_by_ai_confusion_matrix'].items():
                    for a, count in cells.items():
                        agreement_rows.append({'policy': policy, 'domain': domain,
                            'verifier_preference': v, 'ai_preference': a, 'count': count})
        transfer_rows.append({'policy': policy, **result['policies'][policy]['transfer_drop']})
    diagnostics = result['diagnostics']
    for mechanism in ('exact_verifier', 'ai_judge'):
        source = prior if mechanism == 'exact_verifier' else diagnostics
        categories = {'clean_correct': source['clean_reference_all_contrasts'], **source['contrasts']}
        for category, value in categories.items():
            diagnostic_rows.append({'mechanism': mechanism, 'category': category, 'pairs': value['pairs'],
                'clean_preference_rate': value['clean_preference_rate'] if mechanism == 'exact_verifier' else value['win_fraction'],
                'tie_rate': value['tie_rate'] if mechanism == 'exact_verifier' else value['tie_fraction'],
                'modified_preference_rate': value['modified_preference_rate'] if mechanism == 'exact_verifier' else value['loss_fraction'],
                'clean_pairwise_score_half_ties': value['clean_pairwise_score_with_half_ties'] if mechanism == 'exact_verifier' else value['ai_win_rate'],
                'category_denominators_overlap': True})
    result_files = {name: csv_bytes(rows) for name, rows in zip(EXPORT_NAMES[:4],
                    (policy_rows, transfer_rows, agreement_rows, diagnostic_rows))}
    sensitivity = {'exact_verifier': {'S_reason': prior['verifier_S_reason'], 'S_outcome': prior['verifier_S_outcome']},
        'ai_judge': {'S_reason': diagnostics['S_reason'], 'S_outcome': diagnostics['S_outcome']},
        'pairs_per_sensitivity': 20, 'category_denominators_overlap': True,
        'interpretation': 'Reasoning-only verifier ties are expected; filler ties are acceptable. Modified filler preference may indicate style bias.'}
    cost = {k: result[k] for k in ('ai_comparison_events', 'ai_model_generate_calls',
                                  'ai_inference_seconds', 'ai_inference_output_tokens')}
    cost['policy_generation_seconds'] = sum(r['generation_seconds'] for r in policy_rows)
    cost['policy_generated_tokens'] = sum(r['generated_tokens'] for r in policy_rows)
    cost['monetary_cost_measured'] = False
    result_files['DIAGNOSTIC_SENSITIVITY.json'] = canonical(sensitivity)
    result_files['INFERENCE_COUNTS_AND_TIME.json'] = canonical(cost)
    return result_files


def failure_bucket(r):
    return ('missing_or_unparseable_final' if r['extracted_final'] is None else
            'correct' if r['exact_reward'] else 'incorrect_designated_final')


def calculate(responses, judgments, scores, core, diagnostics):
    metrics = {}
    failures = []
    for policy in core.POLICIES:
        metrics[policy] = {}
        for domain in core.DOMAINS:
            values = [r for r in responses if r['policy'] == policy and r['domain'] == domain]
            pairs = [r for r in judgments if r['kind'] == 'policy' and r['policy'] == policy and r['domain'] == domain]
            metrics[policy][domain] = core.policy_summary(values, pairs)
            c = Counter(failure_bucket(r) for r in values)
            failures.append({'policy': policy, 'domain': domain, 'responses': len(values),
                **{k: c[k] for k in ('correct', 'missing_or_unparseable_final', 'incorrect_designated_final')}})
        metrics[policy]['transfer_drop'] = core.transfer_drop(metrics[policy]['gsm'], metrics[policy]['transfer'])
    result = {'policies': metrics, 'diagnostics': core.diagnostic_summary(
        [r for r in judgments if r['kind'] == 'diagnostic'], diagnostics.VARIANTS),
        'ai_comparison_events': len(judgments),
        'ai_model_generate_calls': sum(not r['cache_hit'] for r in judgments),
        'ai_inference_seconds': sum(r['judge_seconds'] for r in judgments),
        'ai_inference_output_tokens': sum(r['judge_output_tokens'] for r in judgments if not r['cache_hit']),
        'release_sha256': RELEASE_SHA, 'qualitative_evidence_pending': True}
    # Original exporter reads canonical JSON: preserve its category/matrix ordering.
    result = json.loads(canonical(result))
    prior = json.loads(canonical(diagnostics.summarize(scores, diagnostics.controlled_pairs(scores))))
    files = export_tables(result, prior, core)
    files['AGGREGATE.json'] = canonical(result)
    files['Task5_Failure_Breakdown_v1.csv'] = csv_bytes(failures)
    observer = module('task5_saved_output_observer',
        ROOT / 'source/stage_helpers/Task5_CPU_Judge_Output_Inspection_v1.py')
    groups = {'all': judgments, 'policy': [r for r in judgments if r['kind'] == 'policy'],
              'diagnostic': [r for r in judgments if r['kind'] == 'diagnostic']}
    for variant in diagnostics.VARIANTS[1:]:
        groups['diagnostic/' + variant] = [r for r in judgments if r.get('variant') == variant]
    for policy in ('rlvr', 'rlaif'):
        for domain in core.DOMAINS:
            groups[policy + '/' + domain] = [r for r in judgments
                if r.get('policy') == policy and r.get('domain') == domain]
    audit = {'groups': {name: observer.summarize_outputs(values) for name, values in groups.items()},
        'distinct_judge_keys': len({r['judge_key'] for r in judgments}),
        'model_calls': sum(not r['cache_hit'] for r in judgments),
        'unique_model_output_summary': observer.summarize_outputs([r for r in judgments if not r['cache_hit']]),
        'saved_effective_token_cap': 4,
        'token_cap_caveat': 'Four saved output tokens alone do not establish truncation; EOS metadata is not saved for judge outputs.',
        'ai_S_reason': result['diagnostics']['S_reason'], 'ai_S_outcome': result['diagnostics']['S_outcome'],
        'no_new_inference': True, 'qualitative_selection_conventions_still_pending': True}
    files['RAW_OUTPUT_AUDIT.json'] = canonical(audit)
    return files


def check_records(responses, judgments, scores, core, diagnostics):
    splits = read('splits/evaluation_ids.json')
    bindings = read('evidence/BINDINGS.json')
    require(len(responses) == 1200 and len(judgments) == 880 and len(scores) == 100, 'Wrong record counts.')
    expected = [(p, d, i) for p in core.POLICIES for d in core.DOMAINS for i in range(core.COUNTS[d])]
    require([(r['policy'], r['domain'], r['row_index']) for r in responses] == expected, 'Response order differs.')
    lookup = {}
    for r in responses:
        require(set(r) == set(RESPONSE_FIELDS) | {'response_sha256', 'generated_token_ids_sha256'}, 'Response schema differs.')
        split = splits[r['domain']][r['row_index']]
        b = bindings[r['policy'] + '/' + r['domain']]
        require(r['source_index'] == split['source_index'] and r['row_sha256'] == split['row_sha256']
                and r['row_sha256'] == b['row_sha256_in_order'][r['row_index']], 'Source IDs/hash differ.')
        require(r['exact_reward'] in (0, 1) and r['format_compliant'] == (r['extracted_final'] is not None)
                and (r['format_compliant'] or r['exact_reward'] == 0), 'Invalid verifier fields.')
        require(type(r['response_tokens']) is int and 0 < r['response_tokens'] <= 512
                and r['truncated'] == (r['response_tokens'] == 512 and not r['terminated_with_eos']), 'Invalid length/termination.')
        require(math.isfinite(r['generation_seconds']) and r['generation_seconds'] > 0, 'Invalid generation time.')
        require(b['seed'] == 6304 and b['release_sha256'] == RELEASE_SHA
                and b['hardware'] == bindings['judgments']['hardware'], 'Frozen generation identity differs.')
        settings = b['settings']['effective_requested_generation']
        for k, value in {'max_new_tokens': 512, 'temperature': .7, 'top_p': .9,
                         'do_sample': True, 'top_k': 20, 'repetition_penalty': 1.1}.items():
            require(settings[k] == value, 'Effective generation setting differs: ' + k)
        lookup[r['policy'], r['domain'], r['row_index']] = r
    diagnostic_ids = read('splits/diagnostic_ids.json')
    require([(s['problem_id'], s['variant_type']) for s in scores] ==
            [(s['problem_id'], s['variant_type']) for s in diagnostic_ids], 'Diagnostic score order differs.')
    dlookup = {(s['problem_id'], s['variant_type']): s for s in scores}
    require(len(dlookup) == 100, 'Duplicate diagnostic score.')
    ids = [s['problem_id'] for s in diagnostic_ids if s['variant_type'] == 'clean_correct']
    expected_units = [f'{p}/{d}/{i}' for p in ('rlvr', 'rlaif') for d in core.DOMAINS for i in range(core.COUNTS[d])]
    expected_units += [f'diagnostic/{pid}/{v}' for pid in ids for v in diagnostics.VARIANTS[1:]]
    require([c['unit_id'] for c in judgments] == expected_units, 'Comparison schedule differs.')
    cache = {}
    for i, c in enumerate(judgments):
        require(not {'problem', 'a', 'b', 'question'} & set(c), 'Raw dataset text in public comparisons.')
        require(c['spec_sha256'] == bindings['judgments']['ordered_spec_sha256'][i], 'Comparison binding differs.')
        label, failed = core.decode_preference(c['raw_judge_output'], c['swapped'])
        require(label == c['ai_preference'] and failed == c['unparseable']
                and c['swapped'] == (int(c['judge_key'][:8], 16) % 2 == 1), 'Parser/swap differs.')
        if c['kind'] == 'policy':
            a = lookup[c['policy'], c['domain'], c['row_index']]
            b = lookup['sft', c['domain'], c['row_index']]
        else:
            a = dlookup[c['problem_id'], 'clean_correct']
            b = dlookup[c['problem_id'], c['variant']]
        require(core.verifier_preference(a['exact_reward'], b['exact_reward']) == c['verifier_preference'], 'Exact preference differs.')
        require(type(c['judge_output_tokens']) is int and 0 <= c['judge_output_tokens'] <= 4, 'Judge length differs.')
        key = c['judge_key']
        stable = {k: c[k] for k in ('raw_judge_output', 'ai_preference', 'swapped',
                                   'unparseable', 'judge_output_tokens', 'judge_prompt_token_ids_sha256')}
        if c['cache_hit']:
            require(key in cache and cache[key] == stable and c['judge_seconds'] == 0, 'Cache evidence differs.')
        else:
            require(key not in cache and math.isfinite(c['judge_seconds']) and c['judge_seconds'] > 0, 'Unique call differs.')
            cache[key] = stable
    require(len(cache) == 865, 'Actual model-call count differs.')
    # Check the approved first-in-source-order selection, including repetitions.
    selection = read('results/QUALITATIVE_SELECTION.json')
    for s in selection['examples']:
        if s['kind'] == 'transfer':
            eligible = [r for r in responses if r['policy'] == s['policy'] and r['domain'] == 'transfer'
                        and failure_bucket(r) == s['failure_type']]
            require(eligible and s['row_index'] == eligible[0]['row_index']
                    and s['response_sha256'] == eligible[0]['response_sha256'], 'Transfer selection differs.')
        else:
            first = next(c for c in judgments if c['kind'] == 'diagnostic' and c['variant'] == s['variant'])
            require(first['unit_id'] == s['unit_id'] and first['problem_id'] == s['problem_id'], 'Diagnostic selection differs.')


def private_check(path, responses, judgments, scores, core, diagnostics):
    require(sha(path.read_bytes()) == PRIVATE_SHA, 'Private bundle SHA differs.')
    with zipfile.ZipFile(path) as z:
        require(z.testzip() is None, 'Private ZIP damaged.')
        manifest = json.loads(z.read('PRIVATE_BUNDLE_MANIFEST.json'))
        for name, expected in manifest.items():
            require(sha(z.read(name)) == expected, 'Private bundle file differs: ' + name)
        raw_responses = [json.loads(s) for s in z.read('POLICY_RESPONSES_PRIVATE.jsonl').splitlines()]
        raw_comparisons = [json.loads(s) for s in z.read('ALL_COMPARISONS_PRIVATE.jsonl').splitlines()]
        require(responses == [project_response(r) for r in raw_responses], 'Response projection differs.')
        require(judgments == [project_judgment(r) for r in raw_comparisons], 'Judgment projection differs.')
        rows = json.loads(z.read('EVALUATION_ROWS_PRIVATE.json'))
        dr = json.loads(z.read('CONTROLLED_DIAGNOSTIC_ROWS_PRIVATE.json'))
        verifier = module('task5_released_exact', ROOT / 'source/frozen_release/task5_feedback/rlvr.py')
        diagnostics.group_rows(dr)
        actual_scores = [diagnostics.score_one(r, i, verifier) for i, r in enumerate(dr)]
        actual_scores = [{k: v for k, v in s.items() if k not in ('gold_final', 'course_expected_exact_reward')}
                         for s in actual_scores]
        require(actual_scores == scores, 'Diagnostic scores differ from course text.')
        for r in raw_responses:
            row = rows[r['domain']][r['row_index']]
            require(core.checksum(row) == r['row_sha256'], 'Private evaluation row differs.')
            require(verifier.exact_reward(r['response'], row['gold_final']) == r['exact_reward']
                    and verifier.extract_designated_final(r['response']) == r['extracted_final'], 'Private exact score differs.')
            ids = r['generated_token_ids']
            require(len(ids) == r['response_tokens'], 'Private token length differs.')
            require(r['terminated_with_eos'] == (151645 in ids)
                    and (151645 not in ids or ids.index(151645) == len(ids) - 1), 'EOS accounting differs.')
        response_lookup = {(r['policy'], r['domain'], r['row_index']): r for r in raw_responses}
        diagnostic_lookup = {(str(r['problem_id']), r['variant_type']): r for r in dr}
        for c in raw_comparisons:
            if c['kind'] == 'policy':
                require(c['problem'] == rows[c['domain']][c['row_index']]['question']
                        and c['a'] == response_lookup[c['policy'], c['domain'], c['row_index']]['response']
                        and c['b'] == response_lookup['sft', c['domain'], c['row_index']]['response'],
                        'Private policy comparison join differs.')
            else:
                clean = diagnostic_lookup[c['problem_id'], 'clean_correct']
                changed = diagnostic_lookup[c['problem_id'], c['variant']]
                require(c['problem'] == clean['question'] and c['a'] == clean['response']
                        and c['b'] == changed['response'], 'Private diagnostic join differs.')
            require(core.judge_key(c['problem'], c['a'], c['b']) == c['judge_key'], 'Private judge key differs.')
            spec = {k: c[k] for k in ('unit_id', 'kind', 'problem', 'a', 'b', 'verifier_preference')}
            spec.update({k: c[k] for k in (('policy', 'domain', 'row_index') if c['kind'] == 'policy' else ('problem_id', 'variant'))})
            require(core.checksum(spec) == c['spec_sha256'], 'Private comparison spec differs.')
    return {'private_bundle_verified': True, 'exact_policy_scores_recomputed': 1200,
            'diagnostic_scores_recomputed': 100, 'raw_comparison_specs_verified': 880,
            'full_rng_journal_chains_rechecked_locally': False}


def verify(private_bundle=None):
    manifest = verify_hashes()
    frozen = ROOT / 'source/frozen_release/task5_run'
    core = module('task5_frozen_core', frozen / 'core.py')
    diagnostics = module('task5_frozen_diagnostics', frozen / 'diagnostics.py')
    responses = read_lines('records/responses.jsonl')
    judgments = read_lines('records/judgments.jsonl')
    scores = read_lines('records/diagnostic_scores.jsonl')
    check_records(responses, judgments, scores, core, diagnostics)
    products = calculate(responses, judgments, scores, core, diagnostics)
    for name, data in products.items():
        require(data == (ROOT / 'results' / name).read_bytes(), 'Recomputed result differs: ' + name)
    extra = private_check(private_bundle, responses, judgments, scores, core, diagnostics) if private_bundle else {
        'private_bundle_verified': False,
        'exact_policy_scores_recomputed': False,
        'note': 'Public mode recalculates statistics from saved score/length records. Use the private bundle to recompute exact scores from response text and gold finals.'}
    return products, {'status': 'VERIFIED', 'package_files': len(manifest['files']),
        'policy_responses': 1200, 'comparison_events': 880, 'diagnostic_scores': 100,
        'actual_judge_model_calls': 865, 'byte_identical_result_files': len(products),
        'new_model_calls': 0, **extra}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--private-bundle', type=Path)
    ap.add_argument('--output', type=Path, help='Optional new output directory for recomputed files.')
    args = ap.parse_args()
    products, receipt = verify(args.private_bundle)
    if args.output:
        require(not args.output.exists(), 'Refusing to replace an existing output directory.')
        args.output.mkdir(parents=True)
        for name, data in products.items():
            (args.output / name).write_bytes(data)
        (args.output / 'VERIFICATION_RECEIPT.json').write_bytes(canonical(receipt))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
