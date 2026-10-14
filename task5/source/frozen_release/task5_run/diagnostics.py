"""CPU-only scoring of the unchanged course Task 5 diagnostic responses."""
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
