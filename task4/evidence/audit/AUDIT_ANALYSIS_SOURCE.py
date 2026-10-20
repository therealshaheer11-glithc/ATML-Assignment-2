"""Verify a frozen completed audit and compare with original fixed judge; CPU only."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path('/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1')
PREPARATION = '20261012T070912.954005Z'
PREPARATION_SHA = '3d3e28e563ea28fe1c3ca4a0ddccc44ae401784604bce3ce9d228d9dcac235fc'
SUMMARY = '20261012T065057.284118Z'
PACKET_ID = '66f81874042a5dcb2bbc6a5df519e84a7bad968847ab44aee1994422dcda57a2'
LABEL_FILE = 'Task4_Completed_Audit_Labels.json'
LABEL_SHA = '664a961436f6707faab3e057fc1c7887f95c64ca2904b4469593ed796b010232'
ORIGIN = 'fixed_rubric_automated_audit'
POLICIES = ['sft', 'dpo', 'ppo', 'grpo']
LABELS = ['SAFE_ANSWER', 'JUSTIFIED_REFUSAL', 'UNSAFE_COMPLIANCE', 'OVER_REFUSAL', 'AMBIGUOUS']


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/linked evidence file: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(folder, relative):
    require(isinstance(relative, str), 'Evidence path is not text')
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in relative and relative not in {'', '.'}, 'Unsafe evidence path')
    result = folder / p
    require(result.resolve().is_relative_to(folder.resolve()), 'Evidence escaped folder')
    return result


def save(path, value):
    data = value if isinstance(value, bytes) else encoded(value)
    require(not path.exists() and not path.is_symlink(), 'Refusing output overwrite')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.pending-' + path.name)
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(not path.exists(), 'Concurrent writer')
    os.replace(temp, path)
    require(path.read_bytes() == data, 'Saved output readback failed')


def csv_bytes(fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def validate_labels(backup, packet):
    require(backup['format'] == 'ATML_TASK4_HUMAN_LABELS_V1' and backup['packet_id'] == packet['packet_id'] == PACKET_ID,
            'Audit packet identity differs')
    require(backup['required'] == backup['completed'] == len(backup['labels']) == 240
            and backup['label_origin'] == ORIGIN, 'Audit completion or origin differs')
    ids = {row['id'] for row in packet['records']}
    require(len(ids) == 240 and set(backup['labels']) == ids
            and all(label in LABELS for label in backup['labels'].values()), 'Incomplete/invalid audit labels')


def bind_mapping(packet, mapping, generated, audit_ids):
    require(len(mapping) == len(packet['records']) == 240 and len({r['id'] for r in mapping}) == 240,
            'Mapping coverage/uniqueness differs')
    public = {r['id']: r for r in packet['records']}
    by_policy = {p: {r['xstest_id']: r for r in generated[p]} for p in POLICIES}
    seen = set()
    for row in mapping:
        require(set(row) == {'id', 'policy', 'xstest_id', 'generation_record_sha256'} and row['policy'] in POLICIES,
                'Mapping schema/policy differs')
        key = (row['policy'], row['xstest_id'])
        require(key not in seen and row['xstest_id'] in audit_ids and row['id'] in public, 'Mapping pair differs')
        seen.add(key)
        original = by_policy[row['policy']][row['xstest_id']]
        require(hashlib.sha256(encoded(original)).hexdigest() == row['generation_record_sha256'], 'Mapped generation checksum differs')
        require(public[row['id']] == {'id': row['id'], 'prompt': original['prompt'], 'response': original['response']},
                'Public audit text differs from its original response')
    require(seen == {(p, i) for p in POLICIES for i in audit_ids}, 'Audit mapping omits or adds pairs')


def comparison(core, joined, audit_ids, labels):
    result = core.manual_agreement(joined, audit_ids, labels)
    # Preserve the original mathematical function and all sixty denominators;
    # describe the fixed-rubric automated audit consistently.
    result['confusion_rows'] = 'audit'
    result['confusion_columns'] = 'original_AI_judge'
    result['audit_label_origin'] = ORIGIN
    require(result['n'] == 60 and result['ambiguous_pairs_retained'] is True,
            'Fixed agreement denominator or ambiguity treatment differs')
    return result


def main():
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    attempt = ROOT / 'audit_analysis_attempts' / stamp
    attempt.mkdir(parents=True, exist_ok=False)
    report = {'format': 'ATML_TASK4_AUDIT_ANALYSIS_ATTEMPT_V1', 'status': 'RUNNING',
              'source_sha256': sha(__file__), 'artifact_sha256': {}, 'models_loaded': 0,
              'inference_calls': 0, 'package_changes': 0, 'training_performed': False,
              'audit_labels_imported': 0,
              'audit_label_origin': ORIGIN}
    try:
        save(attempt / 'AUDIT_ANALYSIS_SOURCE.py', Path(__file__).read_bytes())
        print('Step 1/3: verifying saved packet, frozen audit and completed label file.', flush=True)
        require(sha(LABEL_FILE) == LABEL_SHA, 'Completed audit-label checksum differs')
        backup = read(LABEL_FILE)
        folder = ROOT / 'human_audit_preparation_attempts' / PREPARATION
        receipt = read(folder / 'REPORT.json')
        require(receipt['status'] == 'BLIND_AUDIT_FORM_READY_HUMAN_LABELS_PENDING'
                and receipt['source_sha256'] == sha(folder / 'AUDIT_PREPARATION_SOURCE.py') == PREPARATION_SHA
                and receipt['packet_id'] == PACKET_ID and receipt['anonymous_records'] == 240,
                'Successful preparation receipt/source differs')
        for name, expected in receipt['artifact_sha256'].items():
            require(sha(safe(folder, name)) == expected, 'Saved packet artifact differs: ' + name)
        contract = read(folder / 'PACKET_CONTRACT.json')
        private = read(folder / 'PRIVATE_MAPPING_DO_NOT_USE_WHILE_LABELING.json')
        require(private['packet_id'] == contract['packet_id'] == PACKET_ID
                and private['evidence'] == contract['evidence'], 'Private mapping identity differs')
        html = (folder / 'Task4_Blind_Human_Audit.html').read_text()
        match = re.search(r'<script id="packet" type="application/json">(.*?)</script>', html, re.S)
        require(match is not None, 'Packet JSON missing from saved form')
        packet = json.loads(match.group(1))
        require(packet['format'] == 'ATML_TASK4_BLIND_PACKET_V1'
                and all(set(r) == {'id', 'prompt', 'response'} for r in packet['records']), 'Public packet schema differs')
        validate_labels(backup, packet)
        summary_folder = ROOT / 'aggregation_attempts' / SUMMARY
        require(sha(summary_folder / 'REPORT.json') == receipt['summary_report_sha256'], 'Original summary receipt differs')
        summary_receipt = read(summary_folder / 'REPORT.json')
        require(summary_receipt['status'] == 'CPU_SUMMARIES_COMPLETE_HUMAN_AUDIT_PENDING', 'Original summary was incomplete')
        for name, expected in summary_receipt['artifact_sha256'].items():
            require(sha(safe(summary_folder, name)) == expected, 'Original summary artifact differs')
        summary = read(summary_folder / 'SUMMARY.json')
        evidence = summary['evidence']
        require(sha(summary_folder / 'SUMMARY.json') == contract['evidence']['summary_sha256']
                and contract['evidence']['export_sha256'] == evidence['export_sha256'], 'Summary/export identity differs')
        for name, key in [('JUDGING_EXECUTION_CONTRACT.json', 'judging_execution_contract_sha256'),
                          ('JUDGING_COMPLETION_CONTRACT.json', 'judging_completion_contract_sha256'),
                          ('GENERATION_COMPLETION_CONTRACT.json', 'generation_completion_contract_sha256'),
                          ('INPUT_IDENTITY_CONTRACT.json', 'input_identity_contract_sha256')]:
            require(sha(ROOT / name) == evidence[key], 'Frozen contract differs: ' + name)
        execution = read(ROOT / 'JUDGING_EXECUTION_CONTRACT.json')
        for name, expected in execution['historical_metadata_sha256'].items():
            require(sha(safe(ROOT, name)) == expected, 'Historical evidence differs')
        require(sha(ROOT / 'AUDIT_ID_CONTRACT.json') == contract['evidence']['audit_id_contract_sha256'], 'Fixed audit IDs differ')
        audit = read(ROOT / 'AUDIT_ID_CONTRACT.json')
        ids = audit['ordered_audit_prompt_ids']
        require(ids == sorted(set(ids)) and len(ids) == 60 and audit['seed'] == 6304 and audit['per_class'] == 30,
                'Fixed audit subset differs')
        core_folder = ROOT / 'orchestration_releases' / evidence['orchestration_source_manifest_sha256']
        require(sha(core_folder / 'MANIFEST.json') == evidence['orchestration_source_manifest_sha256'], 'Original core manifest differs')
        for name, expected in read(core_folder / 'MANIFEST.json').items():
            require(sha(safe(core_folder, name)) == expected, 'Original core payload differs')
        require(sha(core_folder / 'core.py') == evidence['aggregation_core_sha256'], 'Original agreement core differs')
        spec = importlib.util.spec_from_file_location('audit_comparison_core', core_folder / 'core.py')
        core = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(core)
        finally:
            sys.dont_write_bytecode = previous
        generated, joined = {}, {}
        for policy in POLICIES:
            for kind, target in [('generation', generated), ('judgment', joined)]:
                path = ROOT / 'evaluation_v1/exports' / kind / (policy + '.jsonl')
                require(sha(path) == evidence['export_sha256'][kind + '/' + policy], 'Primary export differs')
                target[policy] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                require(len(target[policy]) == 450, 'Export coverage differs')
            require(Counter(r['benchmark_class'] for r in generated[policy] if r['xstest_id'] in ids)
                    == {'SAFE': 30, 'UNSAFE': 30}, 'Audit class balance differs')
            require(len({r['xstest_id'] for r in generated[policy]}) == 450, 'Duplicate generation IDs')
            for original, judged in zip(generated[policy], joined[policy]):
                require(set(judged) == set(original) | {'judge'} and {k:v for k,v in judged.items() if k != 'judge'} == original,
                        'Joined original response differs')
                core.validate_judge_output(judged['judge'])
        bind_mapping(packet, private['mapping'], generated, ids)
        save(attempt / LABEL_FILE, Path(LABEL_FILE).read_bytes())
        print('Step 2/3: comparing all sixty fixed audit responses per policy, retaining ambiguity.', flush=True)
        results, joined_rows, matrix_rows = {}, [], []
        lookup = {(r['policy'], r['xstest_id']): r['id'] for r in private['mapping']}
        for policy in POLICIES:
            labels = {i: backup['labels'][lookup[(policy, i)]] for i in ids}
            results[policy] = comparison(core, joined[policy], ids, labels)
            save(attempt / 'agreement' / (policy + '.json'), results[policy])
            matrix_rows += [{'policy': policy, 'audit_label': a, 'original_judge_label': j, 'count': n}
                            for a, values in results[policy]['confusion_counts'].items() for j, n in values.items()]
            by_id = {r['xstest_id']: r for r in joined[policy]}
            for i in ids:
                r = by_id[i]
                judge_label = r['judge']['parsed']['label']
                joined_rows.append({'anonymous_id': lookup[(policy, i)], 'policy': policy, 'xstest_id': i,
                    'benchmark_class': r['benchmark_class'], 'type': r['type'], 'prompt': r['prompt'],
                    'response': r['response'], 'audit_label': labels[i], 'original_judge_label': judge_label,
                    'agree': labels[i] == judge_label})
            print(policy + ': agreement ' + str(results[policy]['matching_labels']) + '/60 ('
                  + format(results[policy]['agreement_percentage'], '.2f') + '%)', flush=True)
        print('Step 3/3: saving exact comparison tables and provenance.', flush=True)
        total_matches = sum(r['matching_labels'] for r in results.values())
        save(attempt / 'AUDIT_COMPARISON.json', {'format': 'ATML_TASK4_AUDIT_COMPARISON_V1', 'packet_id': PACKET_ID,
            'audit_label_origin': ORIGIN, 'policy_order': POLICIES,
            'policy_results': results, 'overall': {'n': 240, 'matching_labels': total_matches,
            'agreement_percentage': total_matches / 240 * 100}, 'ambiguous_pairs_retained': True,
            'audit_label_counts': dict(Counter(backup['labels'].values()))})
        save(attempt / 'agreement.csv', csv_bytes(['policy', 'n', 'matching_labels', 'agreement_percentage'],
            [{'policy': p, **{k: results[p][k] for k in ['n', 'matching_labels', 'agreement_percentage']}} for p in POLICIES]))
        save(attempt / 'confusion_counts.csv', csv_bytes(['policy', 'audit_label', 'original_judge_label', 'count'], matrix_rows))
        save(attempt / 'PRIVATE_AUDIT_COMPARISON_ROWS.json', joined_rows)
        provenance = {'packet_id': PACKET_ID, 'labels_sha256': LABEL_SHA, 'audit_label_origin': ORIGIN,
            'preparation_report_sha256': sha(folder / 'REPORT.json'), 'summary_report_sha256': sha(summary_folder / 'REPORT.json'),
            'packet_contract_sha256': sha(folder / 'PACKET_CONTRACT.json'), 'private_mapping_sha256': sha(folder / 'PRIVATE_MAPPING_DO_NOT_USE_WHILE_LABELING.json'),
            'original_agreement_core_sha256': evidence['aggregation_core_sha256'],
            'original_agreement_function_unchanged': True, 'row_metadata_adapted_to_actual_label_origin': True,
            'frozen_export_sha256': evidence['export_sha256'], 'subset_resampled': False,
            'judgments_remapped': False, 'confidence_used_for_filtering_or_weighting': False}
        save(attempt / 'AUDIT_ANALYSIS_PROVENANCE.json', provenance)
        for name, expected in execution['historical_metadata_sha256'].items():
            require(sha(safe(ROOT, name)) == expected, 'Historical contract changed during audit analysis')
        report.update(status='AUDIT_LABELS_VERIFIED_AND_COMPARISON_SAVED', audit_labels_imported=240,
                      policies_compared=4, responses_per_policy=60, matching_labels=total_matches,
                      agreement_percentage=total_matches / 240 * 100, provenance=provenance)
    except BaseException as error:
        import traceback
        report.update(status='AUDIT_ANALYSIS_STOPPED_EVIDENCE_PRESERVED',
                      error={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
    for path in sorted(attempt.rglob('*')):
        if path.is_file() and not path.name.startswith('.pending-'):
            report['artifact_sha256'][path.relative_to(attempt).as_posix()] = sha(path)
    save(attempt / 'REPORT.json', report)
    print('Saved report:', attempt / 'REPORT.json', flush=True)
    print('Status:', report['status'], flush=True)
    if 'error' in report:
        print(json.dumps(report['error'], indent=2), flush=True)
    print('Audit labels imported:', report['audit_labels_imported'], '/240. No model loading, inference, installation or training.', flush=True)
    print('Return the full output before interpretation and report preparation.', flush=True)


if __name__ == '__main__':
    main()
