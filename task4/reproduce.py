"""Recompute the frozen Task 4 results on CPU using only the standard library."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'evidence'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def recompute(output=None):
    from verify_package import verify_integrity
    verify_integrity()
    exporter = runpy.run_path(str(EVIDENCE / 'exporter/Task4_CPU_Export_GitHub.py'))
    core = runpy.run_path(str(EVIDENCE / 'source/orchestration/core.py'))
    aggregation = runpy.run_path(str(EVIDENCE / 'summaries/AGGREGATION_SOURCE.py'))
    policies = exporter['POLICIES']
    def records(kind):
        return {p: [json.loads(line) for line in (EVIDENCE / 'experiment/evaluation_v1/exports' / kind / (p + '.jsonl')).read_text().splitlines()] for p in policies}
    generations, judgments = records('generation'), records('judgment')
    ids = read('experiment/AUDIT_ID_CONTRACT.json')['ordered_audit_prompt_ids']
    audit_rows = read('audit/PRIVATE_AUDIT_COMPARISON_ROWS.json')
    exporter['check_exports'](core, read('experiment/EVALUATION_EXECUTION_CONTRACT.json'),
        generations, judgments, read('summaries/SUMMARY.json')['metrics'], ids, audit_rows,
        {p: read('audit/agreement/' + p + '.json') for p in policies},
        read('audit/Task4_Completed_Audit_Labels.json'))
    metrics, agreements = {}, {}
    for p in policies:
        metrics[p] = core['aggregate_policy'](p, [{k: row[k] for k in core['META']} for row in generations[p]], judgments[p])
        agreements[p] = core['manual_agreement'](judgments[p], ids, {r['xstest_id']: r['audit_label'] for r in audit_rows if r['policy'] == p})
        agreements[p].update(confusion_rows='audit', confusion_columns='original_AI_judge', audit_label_origin=exporter['ORIGIN'], independent_human_validation=False)
        require(metrics[p] == read('summaries/metrics/' + p + '.json'), 'Metric sidecar differs: ' + p)
    csv_bytes = aggregation['csv_bytes']
    tables = {}
    tables['summaries/rates.csv'] = csv_bytes(['policy', 'metric', 'count', 'denominator', 'rate', 'percentage'],
        [{'policy': p, 'metric': n, **v} for p in policies for n, v in metrics[p]['rates'].items()])
    tables['summaries/response_lengths.csv'] = csv_bytes(['policy', 'responses', 'mean', 'population_sd', 'min', 'max'],
        [{'policy': p, 'responses': 450, **metrics[p]['response_tokens']} for p in policies])
    ambiguity = []
    for p in policies:
        ambiguity.append({'policy': p, 'scope': 'overall', **metrics[p]['ambiguous']['overall']})
        ambiguity.extend({'policy': p, 'scope': n, **v} for n, v in metrics[p]['ambiguous']['by_class'].items())
    tables['summaries/ambiguity.csv'] = csv_bytes(['policy', 'scope', 'count', 'denominator', 'percentage'], ambiguity)
    tables['summaries/label_distributions.csv'] = csv_bytes(['policy', 'scope', 'benchmark_class', 'type', 'label', 'count', 'denominator', 'percentage'],
        [row for p in policies for row in aggregation['distribution_rows'](p, metrics[p])])
    tables['audit/agreement.csv'] = csv_bytes(['policy', 'n', 'matching_labels', 'agreement_percentage'],
        [{'policy': p, **{k: agreements[p][k] for k in ['n', 'matching_labels', 'agreement_percentage']}} for p in policies])
    tables['audit/confusion_counts.csv'] = csv_bytes(['policy', 'audit_label', 'original_judge_label', 'count'],
        [{'policy': p, 'audit_label': a, 'original_judge_label': j, 'count': n}
         for p in policies for a, values in agreements[p]['confusion_counts'].items() for j, n in values.items()])
    for name, data in tables.items():
        require(data == (EVIDENCE / name).read_bytes(), 'Recomputed CSV differs: ' + name)
    if output is not None:
        output = Path(output).absolute()
        require(not output.exists() and not output.is_symlink(), 'Choose a fresh output directory')
        require(not output.resolve().is_relative_to(ROOT), 'Write recomputed results outside the frozen task4 package')
        output.mkdir(parents=True, exist_ok=False)
        for name, data in tables.items():
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        for group, values in [('metrics', metrics), ('agreement', agreements)]:
            folder = output / group
            folder.mkdir()
            for p, value in values.items():
                (folder / (p + '.json')).write_bytes(exporter['encoded'](value))
        report = {'status': 'REPRODUCED', 'responses': 1800, 'judgments': 1800, 'audit_labels': 240,
                  'csv_tables_byte_equal': 6, 'metrics_recomputed': 4, 'confusion_matrices_recomputed': 4,
                  'independent_human_validation': False, 'models_loaded': 0,
                  'package_manifest_sha256': hashlib.sha256((ROOT / 'PACKAGE_SHA256.json').read_bytes()).hexdigest()}
        (output / 'VERIFICATION.json').write_bytes(exporter['encoded'](report))
    print('TASK 4 REPRODUCED: 1800 response/judgment pairs, 240 audit labels, 6 byte-identical CSV tables. No GPU used.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='Fresh output directory outside task4; omit for read-only verification')
    args = parser.parse_args()
    recompute(args.output)
