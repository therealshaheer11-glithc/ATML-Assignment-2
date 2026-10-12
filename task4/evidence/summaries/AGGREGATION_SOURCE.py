"""CPU-only fixed Task 4 summaries from verified frozen response/judge exports."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path('/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1')
VERIFY_ATTEMPT = '20261012T064103.775752Z'
JUDGE_ATTEMPT = '20261012T053854.082678Z'
JUDGE_SOURCE_SHA = 'b7a51de411113a6a3412176f03b33a58206115b9a60ecaa83e5b21b3d07405b8'
CORE_SOURCE = '35ca690761d2bfda0992ff37855b16fc72c03ffd4b47afaa81edc533ab7a5f65'
POLICIES = ['sft', 'dpo', 'ppo', 'grpo']


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or linked file: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe(folder, relative):
    path = Path(relative)
    require(isinstance(relative, str) and not path.is_absolute() and '..' not in path.parts
            and '\\' not in relative and relative not in {'', '.'}, 'Unsafe evidence path')
    result = folder / path
    require(result.resolve().is_relative_to(folder.resolve()), 'Evidence escaped folder')
    return result


def immutable(path, value):
    path = Path(path)
    payload = value if isinstance(value, bytes) else (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    require(not path.exists() and not path.is_symlink(), 'Refusing output overwrite')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / ('.pending-' + path.name)
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    require(not path.exists(), 'Concurrent writer created evidence')
    os.replace(temporary, path)
    require(path.read_bytes() == payload, 'Output readback failed')


def csv_bytes(fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def check_joined(policy, rows, generated, joined, core, eos):
    require(len(generated) == len(joined) == len(rows), 'Export coverage differs')
    for source, generation, judged in zip(rows, generated, joined):
        core.validate_generated([generation], policy, [source], eos)
        require(set(judged) == set(generation) | {'judge'}
                and {key: value for key, value in judged.items() if key != 'judge'} == generation,
                'Joined export changed a frozen generated response or its metadata')
        core.validate_judge_output(judged['judge'])


def distribution_rows(policy, metric):
    groups = [('overall', '', '', metric['overall_labels'])]
    groups += [('class', name, '', value) for name, value in metric['by_class'].items()]
    groups += [('type', '', value['type'], value) for value in metric['by_type']]
    groups += [('class_and_type', value['benchmark_class'], value['type'], value)
               for value in metric['by_class_and_type']]
    return [{'policy': policy, 'scope': scope, 'benchmark_class': benchmark_class,
        'type': prompt_type, 'label': label, 'count': value['counts'][label],
        'denominator': value['n'], 'percentage': value['percentages'][label]}
        for scope, benchmark_class, prompt_type, value in groups for label in value['counts']]


def main():
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    attempt = ROOT / 'aggregation_attempts' / stamp
    attempt.mkdir(parents=True, exist_ok=False)
    report = {'format': 'ATML_TASK4_CPU_AGGREGATION_ATTEMPT_V1', 'status': 'RUNNING',
        'attempt_utc': stamp, 'source_sha256': sha(__file__), 'artifact_sha256': {},
        'models_loaded': 0, 'generation_calls': 0, 'judge_calls': 0, 'package_changes': 0,
        'training_performed': False, 'human_labels_assigned': 0, 'human_annotations_inspected': False,
        'historical_contracts_rewritten': False, 'manual_agreement_computed': False}

    def save(name, value):
        immutable(attempt / name, value)

    try:
        save('AGGREGATION_SOURCE.py', Path(__file__).read_bytes())
        print('Step 1/3: verifying saved judgment verification, completion and frozen evidence.', flush=True)
        verification_path = ROOT / 'judgment_verification_attempts' / VERIFY_ATTEMPT / 'REPORT.json'
        verification = read(verification_path)
        require(verification['status'] == 'ALL_SAVED_JUDGMENTS_AND_EXPORTS_VERIFIED'
                and verification['judgments'] == 1800 and verification['exports_verified'] == 4
                and verification['remaining_judgments'] == 0
                and verification['policy_judgment_counts'] == {p: 450 for p in POLICIES}
                and verification['inference_or_training_performed'] is False,
                'Saved successful judgment verification differs')
        judging_folder = ROOT / 'judging_attempts' / JUDGE_ATTEMPT
        require(sha(judging_folder / 'REPORT.json') == verification['judging_report_sha256'],
                'Verified successful judging report differs')
        judging = read(judging_folder / 'REPORT.json')
        require(judging['status'] == 'ALL_FOUR_POLICIES_JUDGED_AND_SAVED'
                and judging['source_sha256'] == JUDGE_SOURCE_SHA
                and sha(judging_folder / 'JUDGING_SOURCE.py') == JUDGE_SOURCE_SHA,
                'Original judging execution source/status differs')
        completion_path = ROOT / 'JUDGING_COMPLETION_CONTRACT.json'
        require(sha(completion_path) == verification['judging_completion_contract_sha256']
                == judging['judging_completion_contract_sha256'], 'Verified judgment completion differs')
        completion = read(completion_path)
        require(completion['format'] == 'ATML_TASK4_JUDGING_COMPLETE_V1'
                and completion['source_sha256'] == JUDGE_SOURCE_SHA
                and completion['judgments'] == 1800
                and completion['policy_judgment_counts'] == {p: 450 for p in POLICIES}, 'Judgment completion incomplete')
        execution_path = ROOT / 'JUDGING_EXECUTION_CONTRACT.json'
        require(sha(execution_path) == completion['judging_execution_contract_sha256']
                == judging['judging_execution_contract_sha256'], 'Judging execution contract differs')
        execution = read(execution_path)
        require(execution['source_sha256'] == JUDGE_SOURCE_SHA and execution['policy_order'] == POLICIES
                and execution['confidence_used_for_filtering_or_weighting'] is False,
                'Frozen judging protocol differs')
        for name, expected in execution['historical_metadata_sha256'].items():
            require(sha(safe(ROOT, name)) == expected, 'Historical preparation/generation identity differs: ' + name)
        require(sha(ROOT / 'EVALUATION_EXECUTION_CONTRACT.json') == completion['evaluation_contract_sha256']
                == execution['evaluation_contract_sha256']
                == sha(ROOT / 'evaluation_v1/CONTRACT.json'), 'Frozen evaluation identity differs')
        generation_path = ROOT / 'GENERATION_COMPLETION_CONTRACT.json'
        require(sha(generation_path) == completion['generation_completion_contract_sha256']
                == execution['generation_completion_contract_sha256'], 'Frozen generation completion differs')
        generation = read(generation_path)
        require(generation['responses'] == 1800 and generation['policy_response_counts'] == {p: 450 for p in POLICIES},
                'Frozen response coverage differs')
        source_folder = ROOT / 'orchestration_releases' / CORE_SOURCE
        require(sha(source_folder / 'MANIFEST.json') == CORE_SOURCE, 'Approved aggregation source manifest differs')
        for relative, expected in read(source_folder / 'MANIFEST.json').items():
            require(sha(safe(source_folder, relative)) == expected, 'Approved aggregation source payload differs: ' + relative)
        spec = importlib.util.spec_from_file_location('task4_cpu_summary_core', source_folder / 'core.py')
        core = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(core)
        finally:
            sys.dont_write_bytecode = previous
        approval_path = ROOT / 'preflight_releases/9a121f29750b5b09c5feafeb491630ba4b88b6c91dc2ed5ee94ebdb989793b65/docs/approval.json'
        require(sha(approval_path) == execution['task4_approval_sha256'] and read(approval_path)['approved'] is True,
                'Original aggregation approval differs')
        frozen = read(ROOT / 'EVALUATION_EXECUTION_CONTRACT.json')
        require(frozen['settings'] == core.SETTINGS and frozen['policies'] == core.POLICIES == POLICIES,
                'Frozen core settings differ')
        data = read(ROOT / 'INPUT_IDENTITY_CONTRACT.json')['inputs']['xstest']
        require(sha(data['path']) == data['sha256'], 'Frozen XSTest file differs')
        with Path(data['path']).open(newline='', encoding='utf-8-sig') as stream:
            rows = [{key: int(row[key]) if key == 'xstest_id' else str(row[key]) for key in core.META}
                    for row in csv.DictReader(stream)]
        require(core.validate_rows(rows) == frozen['ordered_prompt_ids']
                and core.checksum(rows) == frozen['prompt_metadata_sha256']
                and Counter(row['benchmark_class'] for row in rows) == {'SAFE': 250, 'UNSAFE': 200},
                'Frozen XSTest IDs/order/metadata/class coverage differs')
        print('Step 2/3: checking four response/judgment export pairs and applying the original aggregation.', flush=True)
        require(set(completion['exports']) == set(generation['exports']) == set(POLICIES), 'Export policy coverage differs')
        metrics, export_hashes = {}, {}
        for policy in POLICIES:
            pair = {}
            for kind, contract, count_field in [('generation', generation, 'responses'), ('judgment', completion, 'judgments')]:
                entry = contract['exports'][policy]
                path = ROOT / 'evaluation_v1/exports' / kind / (policy + '.jsonl')
                require(entry['path'] == str(path) and entry[count_field] == 450 and sha(path) == entry['sha256'],
                        'Frozen export identity/hash differs: ' + kind + '/' + policy)
                pair[kind] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                export_hashes[kind + '/' + policy] = entry['sha256']
            check_joined(policy, rows, pair['generation'], pair['judgment'], core, frozen['policy_eos_token_id'])
            metric = core.aggregate_policy(policy, rows, pair['judgment'])
            require(metric['responses'] == 450 and metric['confidence_used_for_rates'] is False
                    and metric['ambiguous_excluded_from_denominators'] is False,
                    'Fixed aggregation semantics differ')
            for name, rate in metric['rates'].items():
                require(rate['denominator'] == (250 if name in {'safe_answer', 'over_refusal'} else 200),
                        'Fixed metric denominator differs')
            metrics[policy] = metric
            save('metrics/' + policy + '.json', metric)
            print(policy + ': 450/450 response/judgment pairs checked; required summaries calculated.', flush=True)
        print('Step 3/3: saving complete JSON/CSV summaries and provenance. Human audit remains pending.', flush=True)
        evidence = {'verification_report_sha256': sha(verification_path), 'judging_completion_contract_sha256': sha(completion_path),
            'judging_execution_contract_sha256': sha(execution_path), 'generation_completion_contract_sha256': sha(generation_path),
            'input_identity_contract_sha256': sha(ROOT / 'INPUT_IDENTITY_CONTRACT.json'), 'xstest_sha256': data['sha256'],
            'orchestration_source_manifest_sha256': CORE_SOURCE, 'aggregation_core_sha256': sha(source_folder / 'core.py'),
            'export_sha256': export_hashes, 'task4_approval_sha256': sha(approval_path)}
        save('SUMMARY.json', {'format': 'ATML_TASK4_CPU_SUMMARIES_V1', 'policy_order': POLICIES,
            'metrics': metrics, 'evidence': evidence, 'human_audit': {'status': 'PENDING_INDEPENDENT_HUMAN_LABELS',
                'required_prompt_ids_per_policy': 60, 'required_labels_total': 240,
                'labels_last_confirmed': 0, 'annotations_inspected': False, 'manual_agreement_computed': False}})
        rate_rows = [{'policy': policy, 'metric': name, **rate}
                     for policy in POLICIES for name, rate in metrics[policy]['rates'].items()]
        save('rates.csv', csv_bytes(['policy', 'metric', 'count', 'denominator', 'rate', 'percentage'], rate_rows))
        save('response_lengths.csv', csv_bytes(['policy', 'responses', 'mean', 'population_sd', 'min', 'max'],
            [{'policy': p, 'responses': 450, **metrics[p]['response_tokens']} for p in POLICIES]))
        ambiguity_rows = []
        for policy in POLICIES:
            ambiguity_rows.append({'policy': policy, 'scope': 'overall', **metrics[policy]['ambiguous']['overall']})
            ambiguity_rows += [{'policy': policy, 'scope': name, **value}
                               for name, value in metrics[policy]['ambiguous']['by_class'].items()]
        save('ambiguity.csv', csv_bytes(['policy', 'scope', 'count', 'denominator', 'percentage'], ambiguity_rows))
        save('label_distributions.csv', csv_bytes(
            ['policy', 'scope', 'benchmark_class', 'type', 'label', 'count', 'denominator', 'percentage'],
            [row for policy in POLICIES for row in distribution_rows(policy, metrics[policy])]))
        save('AGGREGATION_PROVENANCE.json', evidence)
        for name, expected in execution['historical_metadata_sha256'].items():
            require(sha(safe(ROOT, name)) == expected, 'Historical contract changed during aggregation: ' + name)
        require(sha(completion_path) == evidence['judging_completion_contract_sha256'], 'Judgment completion changed')
        report.update(status='CPU_SUMMARIES_COMPLETE_HUMAN_AUDIT_PENDING', policies_summarized=4,
            responses_summarized=1800, judgments_summarized=1800,
            summary_sha256=sha(attempt / 'SUMMARY.json'), evidence=evidence,
            human_labels_last_confirmed=0, required_human_labels=240)
        print('Observed rates from the saved AI judgments (counts retain every class prompt):', flush=True)
        for policy in POLICIES:
            metric = metrics[policy]
            print(policy, '; '.join(name + '=' + str(value['count']) + '/' + str(value['denominator'])
                + ' (' + format(value['percentage'], '.2f') + '%)' for name, value in metric['rates'].items()), flush=True)
            print('  response tokens: mean=' + format(metric['response_tokens']['mean'], '.3f')
                + '; population SD=' + format(metric['response_tokens']['population_sd'], '.3f')
                + '; ambiguous=' + str(metric['ambiguous']['overall']['count']) + '/450', flush=True)
    except BaseException as error:
        import traceback
        report.update(status='CPU_AGGREGATION_STOPPED_EVIDENCE_PRESERVED',
            error={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
    for path in sorted(attempt.rglob('*')):
        if path.is_file() and not path.name.startswith('.pending-') and path.name != 'REPORT.json':
            report['artifact_sha256'][path.relative_to(attempt).as_posix()] = sha(path)
    save('REPORT.json', report)
    print('Saved report:', attempt / 'REPORT.json', flush=True)
    print('Status:', report['status'], flush=True)
    if 'error' in report:
        print(json.dumps(report['error'], indent=2), flush=True)
    print('Responses/judgments last verified: 1800/1800 each; human labels last confirmed: 0/240.', flush=True)
    print('No model loading, inference, installation, training, human labeling or manual-agreement calculation.', flush=True)
    print('Return the full output before preparing the independent human audit.', flush=True)


if __name__ == '__main__':
    main()
