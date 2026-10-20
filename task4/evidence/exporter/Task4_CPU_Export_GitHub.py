"""Export the frozen Task 4 evidence for a reproducible GitHub package (CPU only).

This reads the previously verified Drive experiment and writes a new export attempt.
It never installs packages, loads models, invokes a judge, or changes prior evidence.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path('/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1')
POLICIES = ['sft', 'dpo', 'ppo', 'grpo']
ORIGIN = 'fixed_rubric_automated_audit'
SUMMARY_ATTEMPT = 'aggregation_attempts/20261012T065057.284118Z'
AUDIT_ATTEMPT = 'audit_analysis_attempts/20261012T074320.223449Z'
PREPARATION_ATTEMPT = 'human_audit_preparation_attempts/20261012T070912.954005Z'
VERIFY_ATTEMPT = 'judgment_verification_attempts/20261012T064103.775752Z'
JUDGE_ATTEMPT = 'judging_attempts/20261012T053854.082678Z'
GENERATION_ATTEMPT = 'generation_attempts/20261011T174101.363739Z'
PREPARATION_REPORT_SHA = '263c464402b980a118a8ddbc5beb3e12e64f612265541199e5185471ded2e2f4'
CORE_MANIFEST_SHA = '35ca690761d2bfda0992ff37855b16fc72c03ffd4b47afaa81edc533ab7a5f65'
CORE_SHA = '6cfa7bc2179771051e093df550d7d495dede2316f48aa461f177bc5176258489'
# These hashes come from the independently verified 24-file review archive.
REVIEW_HASHES = {'audit/AUDIT_ANALYSIS_PROVENANCE.json': 'ec602a183a74c9f972271d59b140e1a5772c8da4318e4d064dbf59d2a56b91f2', 'audit/AUDIT_ANALYSIS_SOURCE.py': 'f911bada1c9562378673b93d28472bf93c4aac3324c6104fa7ab5011d7f8d377', 'audit/AUDIT_COMPARISON.json': '2770a743a4f0a075ae6275c9fff4f557e525f1363d8a4f69a84366e44f45c6c0', 'audit/PRIVATE_AUDIT_COMPARISON_ROWS.json': 'fc8f38e0cb5ecc5b311642c9b1cba065def304b7037a9eb3d5c49b2d14a2475d', 'audit/REPORT.json': '619077010f5a7a52805b38c6c6e82d29f9b5ae9313b853f404902cbedd4a8a40', 'audit/Task4_Completed_Audit_Labels.json': '4d95102c5385b166b291d8579c38986fc4dd9c0112742b8d0874fdc11b1d2d0c', 'audit/agreement.csv': '25691fa1b9238ef379cfbb168cb0b278aaf458551ec0552afe21c8e067c3a907', 'audit/agreement/dpo.json': 'a0789238a832d678041628e531bbe876e7011ac7617a839cd62e75be1ab774c0', 'audit/agreement/grpo.json': 'd0e7594248bef11f7c6c5eeab406c02c6dc065c6bf4ac771ec641a80f08688a8', 'audit/agreement/ppo.json': '5b183882b261821890a94f32a123b2f76cba852e105810f781fdba2955df31c0', 'audit/agreement/sft.json': '5b183882b261821890a94f32a123b2f76cba852e105810f781fdba2955df31c0', 'audit/confusion_counts.csv': 'a640f717a129f1c9d0fd880512a8e8d8f7e7583afb13c9da42d159d7af3faaa7', 'summaries/AGGREGATION_PROVENANCE.json': 'ed1aeb487f4bc5c8872bdc49f954ddaa579fc769f75889797b9b1184d91b4074', 'summaries/AGGREGATION_SOURCE.py': '30e0841b49339db01ca8e2bd45fd37e84a70185c8131f1bb5b973d3b3076941b', 'summaries/REPORT.json': 'e17d37642169ca99473d9976837e5fe966c2c5fccd8db71fb646caa5a4becf7d', 'summaries/SUMMARY.json': '1f1c37628397a4d1a1e40ca0593feb048d850f1f8f5abc029f4cdf8655b44a26', 'summaries/ambiguity.csv': '36b05e41ecc2aaacaf06f6d25c57cde846211a75408a7b3a63a8219834b9edbd', 'summaries/label_distributions.csv': 'f83b0712e7b171ebe5e7d42a305688df0c73cee2a4162254177e1ef9d5b58b33', 'summaries/metrics/dpo.json': 'ecbaaa3dfec15a6c32bc841a0a70e2a03327154c0efbcb9d78b6acccdbd164fc', 'summaries/metrics/grpo.json': '8a73fd3d71457ce192df627eed49dff3411865010b0144935f70116eb1adec0b', 'summaries/metrics/ppo.json': '868c265558f9747e2fa9eddb2df36f07398d753aa7154eeb7a62b898aeef4380', 'summaries/metrics/sft.json': '29025b00b7f4401fa8c0398442f885669f22cecb2c522289287279e8c4859a29', 'summaries/rates.csv': '58ebe4edbbdff7af515ab576d790c61461630e4f08ed507fe1a3560234000bc8', 'summaries/response_lengths.csv': '6962aa5c701e613c534de762d1cff25b7dc2525c9739bd601a52b8a7d8f35d28'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + '\n').encode('utf-8')


def relative_name(name):
    require(isinstance(name, str) and name not in {'', '.'} and '\\' not in name,
            'Invalid relative path')
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and str(p) == name,
            'Unsafe or noncanonical relative path: ' + name)
    return p


class Collector:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.payloads = {}
        self.inputs = {}
        self.total_bytes = 0

    def add(self, source, destination, expected=None):
        relative_name(source)
        relative_name(destination)
        path = self.root / source
        require(path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(self.root),
                'Missing, linked, or escaped evidence: ' + source)
        require(path.suffix.lower() in {'.json', '.jsonl', '.csv', '.py', '.txt', '.log', '.md', '.html'},
                'Unexpected artifact type; stop for review: ' + source)
        require(path.stat().st_size <= 16 * 1024 * 1024, 'Artifact too large: ' + source)
        if source in self.inputs:
            data = self.payloads[self.inputs[source]['destination']]
        else:
            data = path.read_bytes()
            require(len(data) <= 16 * 1024 * 1024, 'Artifact grew during read: ' + source)
        actual = digest(data)
        require(expected is None or actual == expected, 'Evidence checksum differs: ' + source)
        if destination in self.payloads:
            require(self.payloads[destination] == data, 'Export destination collision: ' + destination)
        else:
            self.total_bytes += len(data)
            require(self.total_bytes <= 128 * 1024 * 1024, 'Small-evidence export exceeded size limit')
            self.payloads[destination] = data
        self.inputs[source] = {'destination': destination, 'sha256': actual, 'bytes': len(data)}
        if len(self.inputs) % 10 == 0:
            print('Evidence files read and checked:', len(self.inputs), flush=True)
        return data

    def object(self, source, destination, expected=None):
        return json.loads(self.add(source, destination, expected))

    def report(self, folder, destination, expected=None, status=None):
        report = self.object(folder + '/REPORT.json', destination + '/REPORT.json', expected)
        require(status is None or report['status'] == status, 'Unexpected report status: ' + folder)
        for name, checksum in report.get('artifact_sha256', {}).items():
            self.add(folder + '/' + name, destination + '/' + name, checksum)
        return report


def check_exports(core, contract, generations, judgments, saved_metrics, audit_ids,
                  audit_rows, saved_agreements, labels):
    require(contract['policies'] == POLICIES and contract['settings'] == core['SETTINGS'],
            'Frozen policy order or decoding settings differ')
    require(len(audit_ids) == len(set(audit_ids)) == 60 and audit_ids == sorted(audit_ids),
            'Audit ID coverage differs')
    require(labels['completed'] == labels['required'] == len(labels['labels']) == 240
            and labels['label_origin'] == ORIGIN, 'Audit label completion or origin differs')
    require(len(audit_rows) == 240 and len({r['anonymous_id'] for r in audit_rows}) == 240,
            'Audit row coverage differs')
    require({r['anonymous_id'] for r in audit_rows} == set(labels['labels']), 'Anonymous label IDs differ')
    require({(r['policy'], r['xstest_id']) for r in audit_rows}
            == {(p, i) for p in POLICIES for i in audit_ids}, 'Audit policy/ID mapping differs')
    meta = core['META']
    for policy in POLICIES:
        generated, joined = generations[policy], judgments[policy]
        require(len(generated) == len(joined) == 450, 'Export row count differs: ' + policy)
        rows = [{k: r[k] for k in meta} for r in generated]
        ids = core['validate_rows'](rows)
        require(ids == contract['ordered_prompt_ids'] and len(set(ids)) == 450
                and core['checksum'](rows) == contract['prompt_metadata_sha256'],
                'Prompt IDs, order, or metadata differ: ' + policy)
        require(Counter(r['benchmark_class'] for r in rows) == {'SAFE': 250, 'UNSAFE': 200},
                'Benchmark class counts differ')
        require(Counter(r['benchmark_class'] for r in rows if r['xstest_id'] in audit_ids)
                == {'SAFE': 30, 'UNSAFE': 30}, 'Audit balance differs')
        for source, generated_row, joined_row in zip(rows, generated, joined):
            core['validate_generated']([generated_row], policy, [source], contract['policy_eos_token_id'])
            require(set(joined_row) == set(generated_row) | {'judge'}
                    and {k: v for k, v in joined_row.items() if k != 'judge'} == generated_row,
                    'Joined judgment changed a generated response: ' + policy)
            core['validate_judge_output'](joined_row['judge'])
        require(core['aggregate_policy'](policy, rows, joined) == saved_metrics[policy],
                'Recomputed full-benchmark metrics differ: ' + policy)
        by_id = {r['xstest_id']: r for r in joined}
        policy_labels = {}
        for r in audit_rows:
            if r['policy'] != policy:
                continue
            original = by_id[r['xstest_id']]
            require(all(r[k] == original[k] for k in ['prompt', 'response', 'benchmark_class', 'type'])
                    and r['original_judge_label'] == original['judge']['parsed']['label']
                    and labels['labels'][r['anonymous_id']] == r['audit_label']
                    and r['agree'] == (r['audit_label'] == r['original_judge_label']),
                    'Audit row does not bind to the original response, judgment, and label')
            policy_labels[r['xstest_id']] = r['audit_label']
        agreement = core['manual_agreement'](joined, audit_ids, policy_labels)
        agreement.update(confusion_rows='audit', confusion_columns='original_AI_judge',
                         audit_label_origin=ORIGIN)
        require(agreement == saved_agreements[policy], 'Recomputed audit comparison differs: ' + policy)
        print(policy + ': 450 response/judgment pairs and 60 audit labels verified; saved metrics reproduced.', flush=True)


def build_payload(root):
    c = Collector(root)
    require(len(REVIEW_HASHES) == 24, 'Pinned review manifest is missing')
    print('Step 1/4: verifying the frozen summaries and completed audit.', flush=True)
    for name, checksum in REVIEW_HASHES.items():
        group, relative = name.split('/', 1)
        folder = {'summaries': SUMMARY_ATTEMPT, 'audit': AUDIT_ATTEMPT}[group]
        c.add(folder + '/' + relative, name, checksum)
    summary = json.loads(c.payloads['summaries/SUMMARY.json'])
    audit_proof = json.loads(c.payloads['audit/AUDIT_ANALYSIS_PROVENANCE.json'])
    evidence = summary['evidence']
    require(audit_proof['frozen_export_sha256'] == evidence['export_sha256'], 'Summary/audit exports differ')
    print('Step 2/4: reading the frozen contracts, source, and eight full response/judgment exports.', flush=True)
    contracts = {}
    for name, field in [('INPUT_IDENTITY_CONTRACT.json', 'input_identity_contract_sha256'),
                        ('GENERATION_COMPLETION_CONTRACT.json', 'generation_completion_contract_sha256'),
                        ('JUDGING_COMPLETION_CONTRACT.json', 'judging_completion_contract_sha256'),
                        ('JUDGING_EXECUTION_CONTRACT.json', 'judging_execution_contract_sha256')]:
        contracts[name] = c.object(name, 'experiment/' + name, evidence[field])
    execution = contracts['JUDGING_EXECUTION_CONTRACT.json']
    for name, checksum in execution['historical_metadata_sha256'].items():
        require(name.endswith('.json'), 'Unexpected historical metadata type: ' + name)
        c.add(name, 'experiment/' + name, checksum)
    contract = c.object('EVALUATION_EXECUTION_CONTRACT.json', 'experiment/EVALUATION_EXECUTION_CONTRACT.json',
                        execution['evaluation_contract_sha256'])
    c.add('evaluation_v1/CONTRACT.json', 'experiment/evaluation_v1/CONTRACT.json', execution['evaluation_contract_sha256'])
    preparation = c.report(PREPARATION_ATTEMPT, 'audit_preparation', PREPARATION_REPORT_SHA,
                           'BLIND_AUDIT_FORM_READY_HUMAN_LABELS_PENDING')
    packet = json.loads(c.payloads['audit_preparation/PACKET_CONTRACT.json'])
    require(digest(c.payloads['audit_preparation/PACKET_CONTRACT.json']) == audit_proof['packet_contract_sha256'],
            'Audit packet contract differs')
    audit_contract = c.object('AUDIT_ID_CONTRACT.json', 'experiment/AUDIT_ID_CONTRACT.json',
                              packet['evidence']['audit_id_contract_sha256'])
    require(audit_contract['seed'] == 6304 and audit_contract['per_class'] == 30, 'Audit sampling settings differ')
    source_folder = 'orchestration_releases/' + CORE_MANIFEST_SHA
    manifest = c.object(source_folder + '/MANIFEST.json', 'source/orchestration/MANIFEST.json', CORE_MANIFEST_SHA)
    for name, checksum in manifest.items():
        c.add(source_folder + '/' + name, 'source/orchestration/' + name, checksum)
    core_bytes = c.payloads['source/orchestration/core.py']
    require(digest(core_bytes) == CORE_SHA == evidence['aggregation_core_sha256'], 'Original aggregation source differs')
    core = {'__name__': 'task4_frozen_export_core', '__file__': source_folder + '/core.py'}
    exec(compile(core_bytes, core['__file__'], 'exec'), core)
    generations, judgments = {}, {}
    for policy in POLICIES:
        for kind, output, completion_name, count_name in [
                ('generation', generations, 'GENERATION_COMPLETION_CONTRACT.json', 'responses'),
                ('judgment', judgments, 'JUDGING_COMPLETION_CONTRACT.json', 'judgments')]:
            key = kind + '/' + policy
            relative = 'evaluation_v1/exports/' + key + '.jsonl'
            entry = contracts[completion_name]['exports'][policy]
            require(entry[count_name] == 450 and entry['sha256'] == evidence['export_sha256'][key],
                    'Completion contract export differs: ' + key)
            data = c.add(relative, 'experiment/' + relative, entry['sha256'])
            output[policy] = [json.loads(line) for line in data.splitlines() if line.strip()]
            print(key + ': ' + str(len(output[policy])) + '/450 records read.', flush=True)
    print('Step 3/4: independently recomputing metrics and audit comparisons on CPU.', flush=True)
    check_exports(core, contract, generations, judgments, summary['metrics'],
                  audit_contract['ordered_audit_prompt_ids'],
                  json.loads(c.payloads['audit/PRIVATE_AUDIT_COMPARISON_ROWS.json']),
                  {p: json.loads(c.payloads['audit/agreement/' + p + '.json']) for p in POLICIES},
                  json.loads(c.payloads['audit/Task4_Completed_Audit_Labels.json']))
    verification = c.object(VERIFY_ATTEMPT + '/REPORT.json', 'execution/judgment_verification/REPORT.json',
                            evidence['verification_report_sha256'])
    require(verification['judgments'] == 1800 and verification['remaining_judgments'] == 0,
            'Original judgment verification was incomplete')
    c.report(JUDGE_ATTEMPT, 'execution/judging', verification['judging_report_sha256'],
             'ALL_FOUR_POLICIES_JUDGED_AND_SAVED')
    generation_report = c.report(GENERATION_ATTEMPT, 'execution/generation',
                                  status='ALL_FOUR_POLICIES_GENERATED_AND_SAVED')
    require(generation_report['generation_completion_contract_sha256']
            == evidence['generation_completion_contract_sha256'],
            'Generation report does not bind the frozen completion contract')
    return c


def write_new(path, data):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), 'Refusing to overwrite: ' + str(path))
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == data, 'Written export failed readback: ' + str(path))


def main(root=ROOT, local_directory=Path('/content')):
    root = Path(root)
    require(root.is_dir(), 'Mount Google Drive first; the saved Task 4 experiment was not found')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    attempt = root / 'github_export_attempts' / stamp
    attempt.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    report = {'format': 'ATML_TASK4_GITHUB_EXPORT_REPORT_V1', 'status': 'RUNNING',
              'created_utc': stamp, 'source_sha256': digest(source), 'models_loaded': 0,
              'generation_calls': 0, 'judge_calls': 0, 'training_performed': False,
              'prior_evidence_modified': False, 'raw_dataset_or_weights_included': False}
    write_new(attempt / 'EXPORT_SOURCE.py', source)
    try:
        c = build_payload(root)
        print('Step 4/4: saving a new checksummed evidence ZIP and a Drive backup.', flush=True)
        c.payloads['exporter/Task4_CPU_Export_GitHub.py'] = source
        c.payloads['SOURCE_PATHS.json'] = encoded(c.inputs)
        verification = {**report, 'status': 'FULL_EXPORTS_AND_METRICS_VERIFIED',
                        'responses': 1800, 'judgments': 1800, 'audit_labels': 240,
                        'policy_metrics_recomputed': 4, 'audit_matrices_recomputed': 4,
                        'scope': 'Verified through mounted Drive; model weights and individual unit envelopes are not exported. Complete ordered generation and judgment records are included.'}
        c.payloads['EXPORT_VERIFICATION.json'] = encoded(verification)
        manifest = {name: {'sha256': digest(data), 'bytes': len(data)} for name, data in sorted(c.payloads.items())}
        c.payloads['EXPORT_MANIFEST.json'] = encoded({'format': 'ATML_TASK4_GITHUB_EVIDENCE_V1', 'files': manifest})
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(c.payloads.items()):
                archive.writestr(name, data)
        archive_bytes = buffer.getvalue()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            require(archive.testzip() is None and set(archive.namelist()) == set(c.payloads), 'ZIP validation failed')
            for name, item in manifest.items():
                require(digest(archive.read(name)) == item['sha256'], 'ZIP payload checksum differs')
        saved = attempt / 'Task4_GitHub_Evidence.zip'
        local = Path(local_directory) / ('Task4_GitHub_Evidence_' + stamp + '.zip')
        write_new(saved, archive_bytes)
        write_new(local, archive_bytes)
        report.update(status='GITHUB_EVIDENCE_EXPORTED_AND_VERIFIED', responses=1800, judgments=1800,
                      audit_labels=240, files=len(manifest), zip_sha256=digest(archive_bytes),
                      zip_bytes=len(archive_bytes), drive_zip=str(saved), local_zip=str(local))
    except Exception as exc:
        report.update(status='EXPORT_STOPPED_FOR_REVIEW', error={'type': type(exc).__name__, 'message': str(exc)})
        write_new(attempt / 'REPORT.json', encoded(report))
        print('Saved report:', attempt / 'REPORT.json', flush=True)
        print('Status:', report['status'], flush=True)
        raise
    write_new(attempt / 'REPORT.json', encoded(report))
    print('Saved report:', attempt / 'REPORT.json', flush=True)
    print('Status:', report['status'], flush=True)
    print('Responses: 1800/1800; judgments: 1800/1800; audit labels: 240/240.', flush=True)
    print('ZIP SHA-256:', report['zip_sha256'], flush=True)
    print('Drive backup:', saved, flush=True)
    print('No inference, model loading, installations, or training.', flush=True)
    return str(local)


if __name__ == '__main__':
    EXPORT_ZIP = main()
