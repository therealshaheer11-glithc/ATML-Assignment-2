"""Read-only CPU inspection of saved labels; private evidence bundle for local review."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import contextlib
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import types
import uuid
import zipfile
from unittest.mock import patch

PROJECT = Path('/content/drive/MyDrive/ATML-Assignment-2')
RELEASE_SHA = 'd6dd999e7fdf241c7136ee13f46babdf13338fa6dc12ff9046d2ea6a2e89fd87'
FEEDBACK_PATH = 'feedback_attempts/20261014T054901.853724Z_8eb2b1ac'
FEEDBACK_SHA = '1e8a459adbd3a2d3e2aa1b4a01972cf1141c235481f4e7f7b54ca95dbe959baa'
PIPELINE_SHA = 'c57b3094100ea785250de015e15e8250d99fec273268ac6a16b7ef1c0850f166'
AGGREGATE_SHA = 'a571244ca85c7316987efb08b0606fe4904793c23b698d3d13c688a3350920ab'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def summarize_outputs(values):
    n = len(values)
    require(n > 0, 'Empty judge-output audit group.')
    raw = Counter(v['raw_judge_output'].strip().upper() for v in values)
    prefs = Counter(v['ai_preference'] for v in values)
    return {'comparisons': n, 'canonical_A': prefs['A'], 'ties': prefs['TIE'], 'canonical_B': prefs['B'],
            'unparseable_fallback_ties': sum(v['unparseable'] for v in values),
            'explicit_raw_TIE': sum(v['raw_judge_output'].strip().upper() == 'TIE' for v in values),
            'parsed_ties_in_other_text': sum(v['ai_preference'] == 'TIE' and not v['unparseable']
                 and v['raw_judge_output'].strip().upper() != 'TIE' for v in values),
            'exact_label_only': sum(v['raw_judge_output'].strip().upper() in ('A', 'B', 'TIE') for v in values),
            'cache_hits': sum(v['cache_hit'] for v in values),
            'saved_output_tokens_equal_four': sum(v['judge_output_tokens'] == 4 for v in values),
            'output_token_histogram': dict(sorted(Counter(v['judge_output_tokens'] for v in values).items())),
            'top_raw_outputs': [{'raw_normalized': text, 'count': count} for text, count in raw.most_common(12)]}


def read_chain(folder, expected_count, core, label, attempt, report):
    require(folder.is_dir() and not folder.is_symlink() and (folder/'units').is_dir(), 'Missing/linked checkpoint folder: '+label)
    binding = core.read_json(folder/'BINDING.json')
    binding_sha = core.checksum(binding)
    paths = sorted((folder/'units').glob('*.json'))
    require(len(paths) == expected_count and all(p.name == f'{i:06d}.json' for i, p in enumerate(paths)),
            'Checkpoint count/order differs: '+label)
    records, digests, previous = [], [], binding_sha
    for i, path in enumerate(paths):
        require(path.is_file() and not path.is_symlink(), 'Missing/linked checkpoint file.')
        record = core.read_json(path)
        contents = {k: v for k, v in record.items() if k != 'record_sha256'}
        require(record['index'] == i and record['binding_sha256'] == binding_sha
                and record['previous_sha256'] == previous and record['record_sha256'] == core.checksum(contents)
                and record['rng_after'] is not None, 'Checkpoint checksum/chain/RNG differs: '+label)
        previous = core.checksum(record)
        digests.append({'index': i, 'record_sha256': previous})
        records.append(record)
        report['verified_prefix_counts'][label] = i+1
        if (i+1) % 100 == 0 or i+1 == expected_count:
            print(f'  {label}: verified {i+1}/{expected_count}', flush=True)
            core.write_once(attempt/'progress'/f"{label.replace('/', '_')}_{i+1:06d}.json",
                            {'verified_count': i+1, 'last_record_sha256': previous})
    return binding, records, digests


def inspect(project, release, feedback, attempt, core, batch, report):
    from task5_run import run, diagnostics
    from task5_feedback.rlvr import exact_reward, extract_designated_final
    root = project/'task5/experiment_v1'
    saved_report = batch.checked(feedback/'REPORT.json', FEEDBACK_SHA)
    require(saved_report['results_complete'] is True and not saved_report['errors']
            and saved_report['verified_saved_ai_comparisons'] == 880
            and saved_report['release_sha256'] == RELEASE_SHA, 'Completed feedback report differs.')
    index = batch.checked(feedback/'RESULTS_INDEX.json', saved_report['results_index_sha256'])
    aggregate_path = Path(index['aggregate']['path'])
    require(aggregate_path.parent == root/'evaluation_v1/aggregate'
            and index['aggregate']['sha256'] == AGGREGATE_SHA, 'Expected aggregate identity differs.')
    aggregate_value = batch.checked(aggregate_path, AGGREGATE_SHA)
    for value in index['exports'].values():
        path = Path(value['path'])
        require(path.parent == feedback and batch.digest(path) == value['sha256'], 'Export checksum/path differs.')
    print('1/3: verifying all saved answer and judgment chains on CPU.', flush=True)
    b, rows = run.inputs(project, release)
    diagnostic_rows = core.read_rows(project/b['diagnostics'])
    diagnostics.group_rows(diagnostic_rows)
    responses, bindings, checkpoint_digests = {}, {}, {}
    for policy in core.POLICIES:
        for domain in core.DOMAINS:
            label = policy+'/'+domain
            binding, records, hashes = read_chain(root/'evaluation_v1/generations'/policy/domain,
                                                  core.COUNTS[domain], core, label, attempt, report)
            require(binding['phase'] == 'generation' and binding['policy'] == policy and binding['domain'] == domain
                    and binding['release_sha256'] == RELEASE_SHA and binding['seed'] == 6304
                    and binding['dataset_sha256'] == b['asset_sha256'][b['datasets'][domain]]
                    and binding['row_sha256_in_order'] == [core.checksum(row) for row in rows[domain]],
                    'Generation binding differs: '+label)
            for i, record in enumerate(records):
                value, row = record['payload'], rows[domain][i]
                require(value['row_index'] == i and value['policy'] == policy and value['domain'] == domain
                        and value['row_sha256'] == core.checksum(row) and value['source_index'] == row['source_index']
                        and value['exact_reward'] == exact_reward(value['response'], row['gold_final'])
                        and value['extracted_final'] == extract_designated_final(value['response'])
                        and value['format_compliant'] == (value['extracted_final'] is not None), 'Saved response/verifier differs.')
            responses[policy, domain] = [r['payload'] for r in records]
            bindings[label], checkpoint_digests[label] = binding, hashes
    judge_folder = root/'evaluation_v1/judgments'
    judge_binding, judge_records, hashes = read_chain(judge_folder, 880, core, 'judgments', attempt, report)
    bindings['judgments'], checkpoint_digests['judgments'] = judge_binding, hashes
    require(judge_binding['phase'] == 'ai_judging' and judge_binding['release_sha256'] == RELEASE_SHA
            and judge_binding['canonical_judge'] == 'Qwen/Qwen2.5-3B-Instruct' and judge_binding['max_new_tokens'] == 4
            and all(value['hardware'] == judge_binding['hardware'] for key, value in bindings.items() if key != 'judgments'),
            'Saved feedback mechanism/runtime differs.')
    def cached_responses(root_arg, policy, domain, rows_arg, release_arg=None):
        require(rows_arg == rows[domain] and release_arg == RELEASE_SHA, 'Calculator row identity differs.')
        return responses[policy, domain], None
    with patch.object(run, 'load_responses', side_effect=cached_responses):
        specs, _ = run.comparison_specs(root/'evaluation_v1', rows, diagnostic_rows, diagnostics.VARIANTS, RELEASE_SHA)
    require(judge_binding['ordered_spec_sha256'] == [core.checksum(s) for s in specs]
            and judge_binding['generated_response_sha256'] == core.checksum({p+'/'+d: r for (p,d),r in responses.items()}),
            'Judgment input binding differs.')
    values, cache_values = [], {}
    for record, spec in zip(judge_records, specs):
        value = record['payload']
        key = core.judge_key(spec['problem'], spec['a'], spec['b'])
        swapped = int(key[:8], 16) % 2 == 1
        preference, unparseable = core.decode_preference(value['raw_judge_output'], swapped)
        require(value['spec_sha256'] == core.checksum(spec) and value['judge_key'] == key and value['swapped'] == swapped
                and value['ai_preference'] == preference and value['unparseable'] == unparseable
                and value['verifier_preference'] == spec['verifier_preference']
                and type(value['judge_output_tokens']) is int and 0 <= value['judge_output_tokens'] <= 4,
                'Saved judge inputs/raw parser mapping differ.')
        require(all(value.get(k) == v for k, v in spec.items() if k not in ('problem', 'a', 'b')),
                'Saved comparison metadata differs from its bound inputs.')
        stable = {k: value[k] for k in ('ai_preference', 'raw_judge_output', 'unparseable', 'swapped',
                                      'judge_output_tokens', 'judge_prompt_token_ids_sha256')}
        if value['cache_hit']:
            require(key in cache_values and stable == cache_values[key] and value['judge_seconds'] == 0,
                    'Cached raw-label evidence differs.')
        else:
            require(key not in cache_values, 'Unexpected repeated uncached judge key.')
            cache_values[key] = stable
        values.append(value)
    print('2/3: checking saved calculations and separating explicit ties from parser fallback ties.', flush=True)
    class CachedJudgmentJournal:
        def __init__(self, folder_arg, binding_arg):
            require(folder_arg == judge_folder and binding_arg == judge_binding, 'Unexpected calculator journal access.')
        def records(self):
            return list(judge_records)
    capture = io.StringIO()
    with patch.object(run, 'load_responses', side_effect=cached_responses), \
            patch.object(core, 'Journal', CachedJudgmentJournal), \
            patch.object(core, 'write_once', side_effect=RuntimeError('Audit refused to rewrite frozen results.')), \
            contextlib.redirect_stdout(capture):
        run.aggregate(project, release)
    require(capture.getvalue().strip() == 'Saved aggregate: '+str(aggregate_path), 'Recomputed aggregate differs.')
    groups = {'all': values, 'policy': [v for v in values if v['kind'] == 'policy'],
              'diagnostic': [v for v in values if v['kind'] == 'diagnostic']}
    for variant in diagnostics.VARIANTS[1:]:
        groups['diagnostic/'+variant] = [v for v in values if v.get('variant') == variant]
    for policy in ('rlvr', 'rlaif'):
        for domain in core.DOMAINS:
            groups[policy+'/'+domain] = [v for v in values if v.get('policy') == policy and v.get('domain') == domain]
    audit = {'groups': {name: summarize_outputs(group) for name, group in groups.items()},
             'distinct_judge_keys': len(cache_values), 'model_calls': sum(not v['cache_hit'] for v in values),
             'unique_model_output_summary': summarize_outputs([v for v in values if not v['cache_hit']]),
             'saved_effective_token_cap': 4,
             'token_cap_caveat': 'Four saved output tokens alone do not establish truncation; EOS metadata is not saved for judge outputs.',
             'ai_S_reason': aggregate_value['diagnostics']['S_reason'], 'ai_S_outcome': aggregate_value['diagnostics']['S_outcome'],
             'no_new_inference': True, 'qualitative_selection_conventions_still_pending': True}
    core.write_once(attempt/'RAW_OUTPUT_AUDIT.json', audit)
    def jsonl(name, records):
        batch.write_blob(attempt/name, b''.join(core.json_bytes(record) for record in records))
    jsonl('ALL_COMPARISONS_PRIVATE.jsonl', ({**spec, **value} for spec, value in zip(specs, values)))
    jsonl('DIAGNOSTIC_COMPARISONS_PRIVATE.jsonl', ({**spec, **value} for spec, value in zip(specs, values) if spec['kind'] == 'diagnostic'))
    jsonl('POLICY_RESPONSES_PRIVATE.jsonl', (value for records in responses.values() for value in records))
    for name, value in (('EVALUATION_ROWS_PRIVATE.json', rows), ('CONTROLLED_DIAGNOSTIC_ROWS_PRIVATE.json', diagnostic_rows),
                        ('BINDINGS.json', bindings), ('CHECKPOINT_DIGESTS.json', checkpoint_digests)):
        core.write_once(attempt/name, value)
    copies = {'AGGREGATE.json': aggregate_path, 'FEEDBACK_REPORT.json': feedback/'REPORT.json',
              'RESULTS_INDEX.json': feedback/'RESULTS_INDEX.json', 'SOURCE_MANIFEST.json': release/'MANIFEST.json',
              'PROTOCOL.json': release/'protocol.json'}
    copies.update({name: Path(value['path']) for name, value in index['exports'].items()})
    for name, path in copies.items():
        batch.write_blob(attempt/name, path.read_bytes())
    batch.write_blob(attempt/'README_PRIVATE.txt', b'Private local-review evidence. Contains course questions and saved responses. Do not commit this bundle or its raw-data files to the public submission repository. No weights, wheels or credentials are included.\n')
    return audit


def main():
    from google.colab import drive, files
    smi = shutil.which('nvidia-smi')
    gpu = subprocess.run([smi, '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True, timeout=15) if smi else None
    require(not ((gpu and gpu.returncode == 0 and gpu.stdout.strip()) or os.environ.get('COLAB_TPU_ADDR')
                 or os.environ.get('TPU_NAME')), 'Use Hardware accelerator: None (CPU) for this cell.')
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    root = PROJECT/'task5/experiment_v1'
    feedback = root/FEEDBACK_PATH
    source = (feedback/'PIPELINE_SOURCE.py').read_bytes()
    require(hashlib.sha256(source).hexdigest() == PIPELINE_SHA, 'Saved inspection utility source differs.')
    batch = types.ModuleType('verified_task5_saved_batch_utilities')
    exec(compile(source, 'verified_task5_batch.py', 'exec'), batch.__dict__)
    release = root/'evaluation_releases'/RELEASE_SHA
    batch.verify_source(release, RELEASE_SHA)
    sys.path.insert(0, str(release))
    for name, module in list(sys.modules.items()):
        if name.startswith(('task5_run', 'task5_feedback')) and getattr(module, '__file__', None):
            require(Path(module.__file__).resolve().is_relative_to(release.resolve()), 'A different Task 5 module is already imported.')
    core = importlib.import_module('task5_run.core')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_'+uuid.uuid4().hex[:8]
    attempt = root/'judge_output_audit_attempts'/stamp
    attempt.mkdir(parents=True, exist_ok=False)
    own_source = globals().get('SOURCE_BYTES')
    if own_source is not None:
        require(hashlib.sha256(own_source).hexdigest() == globals().get('WRAPPER_SOURCE_SHA256'), 'Inspection source differs.')
        batch.write_blob(attempt/'INSPECTION_SOURCE.py', own_source)
    report = {'status': 'CPU_JUDGE_OUTPUT_AUDIT_RUNNING', 'errors': [], 'audit_complete': False,
              'verified_prefix_counts': {}, 'source_sha256': globals().get('WRAPPER_SOURCE_SHA256'),
              'baseline_feedback_report_sha256': FEEDBACK_SHA, 'baseline_aggregate_sha256': AGGREGATE_SHA,
              'models_loaded': 0, 'answers_generated': 0, 'ai_judgments_generated': 0,
              'installs': 0, 'downloads_from_internet': 0, 'existing_results_changed': False}
    core.write_once(attempt/'START.json', report)
    try:
        audit = inspect(PROJECT, release, feedback, attempt, core, batch, report)
        report.update({'status': 'CPU_JUDGE_OUTPUT_AUDIT_COMPLETE', 'audit_complete': True,
                       'verified_policy_answers': 1200, 'verified_ai_comparison_events': 880,
                       'all_output_summary': audit['groups']['all'], 'diagnostic_summary': audit['groups']['diagnostic']})
    except KeyboardInterrupt:
        report['status'] = 'CPU_JUDGE_OUTPUT_AUDIT_INTERRUPTED_REVIEW_PENDING'
        report['errors'].append('Audit interrupted; verified prefix counts are partial only.')
    except Exception as exc:
        report['status'] = 'CPU_JUDGE_OUTPUT_AUDIT_STOPPED_REVIEW_PENDING'
        report['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        core.write_once(attempt/'REPORT.json', report)
        print('Status:', report['status'])
        print('Errors:', report['errors'])
        print('Saved audit report:', attempt/'REPORT.json')
        print('Audit report SHA-256:', batch.digest(attempt/'REPORT.json'))
    if not report['audit_complete']:
        print('Send the printed output for review; no inference was performed.')
        return
    print(json.dumps(audit, indent=2), flush=True)
    print('3/3: saving and downloading the private evidence bundle.', flush=True)
    entries = sorted(p for p in attempt.rglob('*') if p.is_file())
    manifest = {p.relative_to(attempt).as_posix(): batch.digest(p) for p in entries}
    core.write_once(attempt/'PRIVATE_BUNDLE_MANIFEST.json', manifest)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in entries+[attempt/'PRIVATE_BUNDLE_MANIFEST.json']:
            bundle.writestr(path.relative_to(attempt).as_posix(), path.read_bytes())
    zip_path = attempt/'Task5_CPU_Inspection_Private.zip'
    batch.write_blob(zip_path, buffer.getvalue())
    receipt = {'status': 'CPU_PRIVATE_REVIEW_BUNDLE_READY', 'path': str(zip_path), 'sha256': batch.digest(zip_path),
               'bytes': zip_path.stat().st_size, 'audit_report_sha256': batch.digest(attempt/'REPORT.json')}
    core.write_once(attempt/'DOWNLOAD_RECEIPT.json', receipt)
    print('Private bundle:', zip_path, flush=True)
    print('Bundle SHA-256:', receipt['sha256'], flush=True)
    print('0 installs; 0 models loaded; 0 new answers; 0 new AI judgments.', flush=True)
    print('Send the printed output and attach the downloaded ZIP here.', flush=True)
    files.download(str(zip_path))


if __name__ == '__main__':
    main()
