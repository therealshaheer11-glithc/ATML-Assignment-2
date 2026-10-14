# Task 5 - Block 6: CPU exact-checking of supplied diagnostic answers only.
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, types, uuid

VERIFIER_SOURCE = r'''from __future__ import annotations

import re

_FINAL_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def extract_designated_final(text: str):
    """Extract the final answer only from the designated `#### <number>` field.

    The last designated final is used. Numbers that appear only in intermediate reasoning are ignored.
    """
    matches = _FINAL_RE.findall(str(text))
    if not matches:
        return None
    return matches[-1].replace(",", "")


def numerically_equal(a, b, tol=1e-9):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def exact_reward(response: str, gold_final: str) -> float:
    pred = extract_designated_final(response)
    return float(pred is not None and numerically_equal(pred, gold_final))
'''
SCORER_SOURCE = r'''"""CPU-only scoring of the unchanged course Task 5 diagnostic responses."""
import hashlib
import json
from collections import Counter

VARIANTS = (
    'clean_correct', 'corrupt_reasoning_correct_final',
    'good_reasoning_wrong_final', 'persuasive_filler_correct',
    'gold_distractor_wrong_final',
)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def group_rows(rows, problems=20):
    require(len(rows) == problems * 5, 'Unexpected diagnostic response count.')
    groups = {}
    for row in rows:
        require(all(k in row for k in ('problem_id', 'variant_type', 'question',
                    'gold_final', 'response', 'expected_exact_reward')),
                'Required diagnostic field missing.')
        require(isinstance(row['question'], str) and isinstance(row['response'], str),
                'Question/response must be text.')
        require(row['expected_exact_reward'] in (0, 1), 'Unexpected staff reward label.')
        pid, variant = str(row['problem_id']), row['variant_type']
        require(variant in VARIANTS, 'Unexpected diagnostic variant.')
        group = groups.setdefault(pid, {})
        require(variant not in group, 'Duplicate problem/variant: ' + pid + '/' + variant)
        group[variant] = row
    require(len(groups) == problems, 'Unexpected diagnostic problem count.')
    for pid, group in groups.items():
        require(set(group) == set(VARIANTS), 'Incomplete problem: ' + pid)
        require(len({r['question'] for r in group.values()}) == 1,
                'Question differs across variants: ' + pid)
        require(len({json.dumps(r['gold_final'], sort_keys=True) for r in group.values()}) == 1,
                'Gold differs across variants: ' + pid)
        require(all(r['response'].count('####') == 1 for r in group.values()),
                'Fixed diagnostic marker structure differs: ' + pid)
        clean = group['clean_correct']['response'].rsplit('####', 1)[0].strip()
        wrong = group['good_reasoning_wrong_final']['response'].rsplit('####', 1)[0].strip()
        require(clean == wrong, 'Outcome-only pair has different reasoning: ' + pid)
    return groups


def score_one(row, index, verifier):
    final = verifier.extract_designated_final(row['response'])
    score = verifier.exact_reward(row['response'], row['gold_final'])
    return {
        'source_row_index': index, 'problem_id': str(row['problem_id']),
        'variant_type': row['variant_type'],
        'question_sha256': text_sha(row['question']),
        'response_sha256': text_sha(row['response']),
        'gold_final': row['gold_final'], 'extracted_final': final,
        'format_compliant': final is not None, 'exact_reward': score,
        'course_expected_exact_reward': row['expected_exact_reward'],
        'matches_course_expected_reward': score == row['expected_exact_reward'],
    }


def controlled_pairs(scores):
    groups = {}
    for score in scores:
        group = groups.setdefault(score['problem_id'], {})
        require(score['variant_type'] not in group, 'Duplicate scored problem/variant.')
        group[score['variant_type']] = score
    pairs = []
    for pid, group in groups.items():
        require(set(group) == set(VARIANTS), 'Incomplete scored group: ' + pid)
        clean = group['clean_correct']['exact_reward']
        for variant in VARIANTS[1:]:
            modified = group[variant]['exact_reward']
            pref = 'A' if clean > modified else 'B' if clean < modified else 'TIE'
            pairs.append({'problem_id': pid, 'reference_variant': 'clean_correct',
                          'modified_variant': variant, 'verifier_preference': pref,
                          'reference_exact_reward': clean, 'modified_exact_reward': modified})
    return pairs


def preference_rates(pairs):
    require(bool(pairs), 'Cannot summarize an empty pair set.')
    counts = Counter(p['verifier_preference'] for p in pairs)
    n = len(pairs)
    return {'pairs': n, 'clean_preferred': counts['A'], 'ties': counts['TIE'],
            'modified_preferred': counts['B'],
            'clean_preference_rate': counts['A'] / n, 'tie_rate': counts['TIE'] / n,
            'modified_preference_rate': counts['B'] / n,
            'clean_pairwise_score_with_half_ties': (counts['A'] + .5 * counts['TIE']) / n}


def summarize(scores, pairs):
    require(bool(scores), 'Cannot summarize an empty score set.')
    require(pairs == controlled_pairs(scores), 'Pairs differ from scored responses.')
    by_variant = {}
    for variant in VARIANTS:
        selected = [s for s in scores if s['variant_type'] == variant]
        require(bool(selected), 'Missing scored variant: ' + variant)
        by_variant[variant] = {
            'responses': len(selected),
            'exact_accuracy': sum(s['exact_reward'] for s in selected) / len(selected),
            'format_compliance': sum(s['format_compliant'] for s in selected) / len(selected),
            'staff_reward_mismatches': sum(not s['matches_course_expected_reward'] for s in selected),
        }
    pair_categories = {variant: preference_rates(
        [p for p in pairs if p['modified_variant'] == variant]) for variant in VARIANTS[1:]}
    return {
        'responses_scored': len(scores), 'unique_controlled_pairs': len(pairs),
        'scorer': 'unchanged released exact verifier', 'by_variant': by_variant,
        'clean_reference_all_contrasts': preference_rates(pairs),
        'contrasts': pair_categories,
        'verifier_S_reason': pair_categories['corrupt_reasoning_correct_final']['clean_preference_rate'],
        'verifier_S_outcome': pair_categories['good_reasoning_wrong_final']['clean_preference_rate'],
        'category_denominators_overlap': True,
        'interpretation': 'Reasoning-only verifier ties are expected. Filler ties are acceptable. '
                          'These are verifier preferences, not AI judgments or policy accuracies.',
        'ai_judgments': 0, 'ai_metrics': None, 'generated_response_token_lengths': None,
    }
'''
APPROVAL = {'format': 'ATML_TASK5_PROTOCOL_APPROVAL_V1',
 'approved': True,
 'user_message': 'okay all approved',
 'approved_proposal_sha256': 'e020dc3d0f29d766953fd56dbfd49ae75b2d34378ebcb87878556e1e03fa0a8e',
 'approved_scope': ['Approve proposed evaluation choices',
                    'Implement modular Task 5 evaluation/diagnostic/aggregation code and '
                    'audit/resume helpers in a Task 5-specific copy',
                    'Run CPU syntax and synthetic semantic/consistency/resume checks',
                    'Compute released exact-verifier scores for the 100 existing fixed diagnostic '
                    'responses on CPU and save evidence'],
 'excluded_scope': ['Pretrained model loading',
                    'Policy response generation',
                    'AI judging',
                    'Training',
                    'Publishing or pushing code',
                    'Changing course data, frozen weights, released verifier or judge decisions'],
 'block6_scope': 'CPU released exact-verifier scoring of the 100 unchanged supplied diagnostic '
                 'responses only',
 'later_pretrained_execution_approval_required': True,
 'a100_environment_setup_already_approved_but_deferred_until_cpu_readiness': True}

project = Path('/content/drive/MyDrive/ATML-Assignment-2')
root = project / 'task5/experiment_v1'
previous = root / 'diagnostic_structure_attempts/20261012T175457.377462Z_e7e14416'
diagnostics = project / 'task3/experiment_v1/course_assets/data/task5_controlled_reward_diagnostics.jsonl'
diagnostic_sha = '9008b96595bb1419f1326bf02beefd689f09028c2aa6598d7c427394602f036c'

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

smi = shutil.which('nvidia-smi')
gpu = subprocess.run([smi, '--query-gpu=name', '--format=csv,noheader'],
                     capture_output=True, text=True, timeout=15) if smi else None
require(not ((gpu and gpu.returncode == 0 and gpu.stdout.strip()) or
             os.environ.get('COLAB_TPU_ADDR') or os.environ.get('TPU_NAME')),
        'Keep the same CPU / None runtime for this block.')
require(root.is_dir() and not root.is_symlink(), 'Task 5 Drive storage unavailable.')
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid.uuid4().hex[:8]
attempt = root / 'exact_diagnostic_attempts' / stamp
attempt.mkdir(parents=True, exist_ok=False)
report = {'status': 'RUNNING', 'errors': [], 'installs': 0, 'downloads': 0,
    'environments_created': 0, 'model_loads': 0, 'policy_responses_generated': 0,
    'ai_judgments': 0, 'diagnostic_scores_computed': 0,
    'diagnostic_scores_saved': 0, 'source_data_modified': False,
    'network_access_in_this_block': False}

def save_bytes(name, payload, update=False):
    path = attempt / name
    path.parent.mkdir(parents=True, exist_ok=True)
    require(update or not path.exists(), 'Evidence already exists: ' + name)
    pending = path.with_name(path.name + '.pending')
    with pending.open('xb') as stream:
        stream.write(payload)
    os.replace(pending, path)
    require(path.read_bytes() == payload, 'Evidence readback failed: ' + name)

def save_json(name, value, update=False):
    save_bytes(name, (json.dumps(value, indent=2, sort_keys=True) + '\n').encode(), update)

try:
    save_json('REPORT.json', report)
    prior_path = previous / 'REPORT.json'
    require(prior_path.is_file() and not prior_path.is_symlink(), 'Prior report unavailable.')
    prior = json.loads(prior_path.read_text())
    require(prior['status'] == 'CPU_DIAGNOSTIC_STRUCTURE_SAVED_PROTOCOL_APPROVAL_PENDING'
            and prior['rows'] == 100 and prior['problems'] == 20
            and prior['identical_reasoning_prefix_clean_vs_wrong_final'] == 20
            and prior['all_course_manual_validation_flags_true'] is True
            and not prior['errors'], 'Prior structural checkpoint differs.')
    report['prior_report_sha256'] = digest(prior_path)
    save_json('EVALUATION_PROTOCOL_APPROVAL.json', APPROVAL)
    for name, text in [('rlvr.py', VERIFIER_SOURCE), ('cpu_diagnostics_v1.py', SCORER_SOURCE)]:
        save_bytes('code/' + name, text.encode())
    require(digest(attempt / 'code/rlvr.py') ==
            'f8865f7d6c67dc4d0565b182411383f0ec2ed9d01a0f99198ad2db0c31ec1bff',
            'Released exact verifier identity differs.')
    verifier, scorer = types.ModuleType('course_rlvr'), types.ModuleType('task5_cpu_diagnostics')
    exec(compile(VERIFIER_SOURCE, 'saved_course_rlvr.py', 'exec'), verifier.__dict__)
    exec(compile(SCORER_SOURCE, 'saved_task5_cpu_diagnostics.py', 'exec'), scorer.__dict__)
    print('1/3: approval and CPU scorer saved; no model loading.', flush=True)
    require(diagnostics.is_file() and not diagnostics.is_symlink()
            and digest(diagnostics) == diagnostic_sha, 'Fixed diagnostic identity differs.')
    rows = [json.loads(line) for line in diagnostics.read_text().splitlines() if line.strip()]
    groups = scorer.group_rows(rows)
    require(all(r.get('manual_validation') is True and r.get('staff_edit_required') is False
                for r in rows), 'Course validation metadata differs.')
    report.update({'diagnostic_sha256': diagnostic_sha, 'responses': len(rows),
                   'problems': len(groups), 'variant_counts': dict(scorer.Counter(
                       r['variant_type'] for r in rows))})
    scores = []
    for index, row in enumerate(rows):
        score = scorer.score_one(row, index, verifier)
        report['diagnostic_scores_computed'] += 1
        save_json('scores/' + str(index).zfill(3) + '.json', score)
        scores.append(score)
        report['diagnostic_scores_saved'] += 1
        save_json('REPORT.json', report, update=True)
        if (index + 1) % 25 == 0:
            print('Exact scores saved:', index + 1, '/ 100', flush=True)
    pairs = scorer.controlled_pairs(scores)
    summary = scorer.summarize(scores, pairs)
    save_json('EXACT_SCORES.json', scores)
    save_json('VERIFIER_CONTROLLED_PAIRS.json', pairs)
    save_json('VERIFIER_SUMMARY.json', summary)
    report.update({'summary': summary, 'unique_verifier_pairs': len(pairs),
        'staff_reward_mismatches': sum(not s['matches_course_expected_reward'] for s in scores)})
    require(report['staff_reward_mismatches'] == 0,
            'Exact scores disagree with supplied metadata; preserve evidence and review.')
    require(digest(diagnostics) == diagnostic_sha, 'Diagnostic source changed during scoring.')
    manifest = {str(p.relative_to(attempt)): digest(p) for p in sorted(attempt.rglob('*'))
                if p.is_file() and p.name != 'REPORT.json'}
    save_json('EVIDENCE_MANIFEST.json', manifest)
    report['evidence_manifest_sha256'] = digest(attempt / 'EVIDENCE_MANIFEST.json')
    report['status'] = 'CPU_EXACT_DIAGNOSTICS_SAVED_REVIEW_PENDING'
    print('2/3: supplied answers checked; source file unchanged.', flush=True)
except Exception as exc:
    report['status'] = 'CPU_EXACT_DIAGNOSTICS_STOPPED_REVIEW_PENDING'
    report['errors'].append(type(exc).__name__ + ': ' + str(exc))
finally:
    save_json('REPORT.json', report, update=True)
    print('3/3: checkpoint saved.')
    print('Status:', report['status'])
    print('Exact scores computed / saved:', report['diagnostic_scores_computed'],
          '/', report['diagnostic_scores_saved'])
    print('Unique verifier comparisons:', report.get('unique_verifier_pairs'))
    print('Verifier summary:', json.dumps(report.get('summary', {}), indent=2))
    print('Errors:', report['errors'])
    print('Saved report:', attempt / 'REPORT.json')
    print('Report SHA-256:', digest(attempt / 'REPORT.json'))
    print('0 installs; 0 downloads; 0 environments created; 0 model loads; '
          '0 responses generated; 0 AI judgments.')
    print('Send the printed output before continuing.')
