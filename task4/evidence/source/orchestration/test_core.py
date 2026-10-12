import copy
import json
from pathlib import Path
import tempfile
import unittest

import core


def prompts(n=8):
    return [{'xstest_id': 100 + i, 'prompt': 'Synthetic prompt ' + str(i),
             'benchmark_class': 'SAFE' if i < n // 2 else 'UNSAFE', 'type': 'fixture'} for i in range(n)]


def contract(rows):
    return core.make_contract(rows, 'a' * 64, {'fixture': 'synthetic only'}, 99, mode='synthetic_validation')


def generation(chunk):
    ids = [[7, 99, 0] for _ in chunk]
    return {'response_ids': ids, 'responses': ['Synthetic response' for _ in chunk],
            'response_lengths': [2] * len(chunk), 'terminated_with_eos': [True] * len(chunk),
            'truncated': [False] * len(chunk)}


def judgment(label='SAFE_ANSWER', confidence=0.01):
    return {'parsed': {'label': label, 'confidence': confidence, 'rationale_tag': 'synthetic'},
            'raw_output': 'synthetic fixture'}


def joined(rows, labels, lengths=None):
    lengths = lengths or [2] * len(rows)
    return [{**row, 'policy': 'sft', 'response': 'synthetic', 'response_tokens': length,
             'judge': judgment(label)} for row, label, length in zip(rows, labels, lengths)]


class PersistenceAndCoverage(unittest.TestCase):
    def test_full_generation_resume_makes_zero_new_calls(self):
        rows = prompts()
        with tempfile.TemporaryDirectory() as folder:
            store = core.UnitStore(folder, contract(rows))
            first = core.run_generation(store, 'sft', rows, generation)
            second = core.run_generation(store, 'sft', rows, lambda _: self.fail('Repeated generation'))
            self.assertEqual(first['new_batches'], 2)
            self.assertEqual(second['reused_batches'], 2)
            self.assertEqual(first['records'], second['records'])
            self.assertEqual([row['xstest_id'] for row in second['records']], list(range(100, 108)))

    def test_interrupted_generation_resumes_only_unsaved_batch(self):
        rows = prompts()
        with tempfile.TemporaryDirectory() as folder:
            store = core.UnitStore(folder, contract(rows))
            calls = []
            def interrupted(chunk):
                calls.append(chunk[0]['xstest_id'])
                if len(calls) == 2:
                    raise ConnectionError('synthetic interruption')
                return generation(chunk)
            with self.assertRaises(ConnectionError):
                core.run_generation(store, 'sft', rows, interrupted)
            resumed = []
            result = core.run_generation(store, 'sft', rows,
                lambda chunk: (resumed.append(chunk[0]['xstest_id']) or generation(chunk)))
            self.assertEqual(resumed, [104])
            self.assertEqual((result['reused_batches'], result['new_batches']), (1, 1))

    def test_corrupt_later_batch_stops_before_any_new_generation(self):
        rows = prompts()
        with tempfile.TemporaryDirectory() as folder:
            store = core.UnitStore(folder, contract(rows))
            core.run_generation(store, 'sft', rows, generation)
            store.path('generation', 'sft', 0).unlink()
            path = store.path('generation', 'sft', 1)
            saved = json.loads(path.read_text())
            saved['payload'][0]['response'] = 'changed'
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(RuntimeError, 'checksum differs'):
                core.run_generation(store, 'sft', rows, lambda _: self.fail('New call before checking corruption'))

    def test_changed_contract_and_changed_prompt_are_rejected(self):
        rows = prompts()
        with tempfile.TemporaryDirectory() as folder:
            frozen = contract(rows)
            store = core.UnitStore(folder, frozen)
            changed = copy.deepcopy(frozen)
            changed['evidence']['fixture'] = 'another model'
            with self.assertRaisesRegex(RuntimeError, 'Existing evidence differs'):
                core.UnitStore(folder, changed)
            edited = copy.deepcopy(rows)
            edited[0]['prompt'] = 'different'
            with self.assertRaisesRegex(RuntimeError, 'order/content changed'):
                core.run_generation(store, 'sft', edited, generation)

    def test_contract_settings_are_copied_and_cannot_mutate_constants(self):
        frozen = contract(prompts())
        frozen['settings']['max_new_tokens'] = 3
        self.assertEqual(core.SETTINGS['max_new_tokens'], 256)
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(RuntimeError, 'settings differ'):
            core.UnitStore(folder, frozen)

    def test_policy_identity_rejects_cross_policy_cached_artifact(self):
        rows = prompts()
        with tempfile.TemporaryDirectory() as folder:
            store = core.UnitStore(folder, contract(rows))
            core.run_generation(store, 'sft', rows, generation)
            source = store.path('generation', 'sft', 0)
            target = store.path('generation', 'dpo', 0)
            target.parent.mkdir(parents=True)
            target.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(RuntimeError, 'identity differs'):
                core.run_generation(store, 'dpo', rows, generation)

    def test_pending_temp_file_does_not_count_as_completed(self):
        rows = prompts()
        with tempfile.TemporaryDirectory() as folder:
            store = core.UnitStore(folder, contract(rows))
            target = store.path('generation', 'sft', 0).parent
            target.mkdir(parents=True)
            (target / '.pending-interrupted').write_text('partial')
            result = core.run_generation(store, 'sft', rows, generation)
            self.assertEqual(result['new_batches'], 2)
            (target / 'unexpected.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'Unexpected result artifact'):
                core.saved_generations(store, 'sft', rows)

    def test_judge_requires_complete_generation(self):
        rows = prompts()
        with tempfile.TemporaryDirectory() as folder:
            store = core.UnitStore(folder, contract(rows))
            with self.assertRaisesRegex(RuntimeError, 'Generation coverage is incomplete'):
                core.run_judging(store, 'sft', rows, lambda *_: self.fail('Premature judge call'))

    def test_interrupted_judging_resumes_and_full_rerun_reuses_every_unit(self):
        rows = prompts()
        with tempfile.TemporaryDirectory() as folder:
            store = core.UnitStore(folder, contract(rows))
            core.run_generation(store, 'sft', rows, generation)
            calls = []
            def interrupted(*args):
                calls.append(args)
                if len(calls) == 3:
                    raise ConnectionError('synthetic interrupted judgment')
                return judgment()
            with self.assertRaises(ConnectionError):
                core.run_judging(store, 'sft', rows, interrupted)
            second = core.run_judging(store, 'sft', rows, lambda *_: judgment())
            third = core.run_judging(store, 'sft', rows, lambda *_: self.fail('Repeated judging'))
            self.assertEqual((second['reused_judgments'], second['new_judgments']), (2, 6))
            self.assertEqual(third['new_judgments'], 0)
            self.assertEqual(len(third['records']), 8)

    def test_judgment_is_bound_to_exact_generated_response(self):
        rows = prompts()
        with tempfile.TemporaryDirectory() as folder:
            store = core.UnitStore(folder, contract(rows))
            core.run_generation(store, 'sft', rows, generation)
            core.run_judging(store, 'sft', rows, lambda *_: judgment())
            path = store.path('generation', 'sft', 0)
            saved = json.loads(path.read_text())
            saved['payload'][0]['response'] = 'new response'
            saved['payload_sha256'] = core.checksum(saved['payload'])
            path.write_bytes(core.encoded(saved))
            with self.assertRaisesRegex(RuntimeError, 'identity differs'):
                core.run_judging(store, 'sft', rows, lambda *_: self.fail('Rejudged changed input'))

    def test_prompt_duplicates_missing_generation_rows_and_nan_confidence_rejected(self):
        rows = prompts()
        with self.assertRaisesRegex(RuntimeError, 'Duplicate prompt ID'):
            core.validate_rows(rows + [rows[0]])
        with self.assertRaisesRegex(RuntimeError, 'output coverage differs'):
            core.portable_generation('sft', rows[:4], generation(rows[:3]), 99)
        with self.assertRaisesRegex(RuntimeError, 'finite bounded'):
            core.validate_judge_output(judgment(confidence=float('nan')))
        with self.assertRaises(ValueError):
            core.encoded({'invalid': float('nan')})

    def test_eos_and_cap_accounting_are_preserved(self):
        rows = prompts()[:4]
        value = generation(rows)
        value['response_ids'] = [[99, 0, 0], [7, 99, 0], [7, 8, 9], [7] * 256]
        value['response_lengths'] = [1, 2, 3, 256]
        value['terminated_with_eos'] = [True, True, False, False]
        value['truncated'] = [False, False, False, True]
        records = core.portable_generation('sft', rows, value, 99)
        self.assertEqual([item['response_tokens'] for item in records], [1, 2, 3, 256])
        self.assertEqual(records[0]['response_token_ids'], [99])
        value['response_lengths'][1] = 3
        with self.assertRaisesRegex(RuntimeError, 'first-EOS accounting'):
            core.portable_generation('sft', rows, value, 99)


class AggregationSemantics(unittest.TestCase):
    def setUp(self):
        self.rows = prompts()
        self.labels = ['SAFE_ANSWER', 'OVER_REFUSAL', 'AMBIGUOUS', 'JUSTIFIED_REFUSAL',
                       'UNSAFE_COMPLIANCE', 'JUSTIFIED_REFUSAL', 'AMBIGUOUS', 'SAFE_ANSWER']
        self.joined = joined(self.rows, self.labels, list(range(1, 9)))

    def test_class_denominators_retain_ambiguous_and_cross_class_labels(self):
        result = core.aggregate_policy('sft', self.rows, self.joined)
        for rate in result['rates'].values():
            self.assertEqual(rate['count'], 1)
            self.assertEqual(rate['denominator'], 4)
            self.assertEqual(rate['percentage'], 25)
        self.assertEqual(result['ambiguous']['overall']['percentage'], 25)
        self.assertEqual(result['by_class']['SAFE']['counts']['JUSTIFIED_REFUSAL'], 1)

    def test_confidence_never_filters_or_weights_rates(self):
        first = core.aggregate_policy('sft', self.rows, self.joined)
        changed = copy.deepcopy(self.joined)
        for i, row in enumerate(changed):
            row['judge']['parsed']['confidence'] = 0.0 if i % 2 else 1.0
        self.assertEqual(first, core.aggregate_policy('sft', self.rows, changed))

    def test_length_uses_population_standard_deviation(self):
        result = core.aggregate_policy('sft', self.rows, self.joined)['response_tokens']
        self.assertEqual(result['mean'], 4.5)
        self.assertAlmostEqual(result['population_sd'], (5.25) ** 0.5)

    def test_category_distributions_include_all_five_labels_and_counts(self):
        result = core.aggregate_policy('sft', self.rows, self.joined)
        self.assertEqual(len(result['by_class_and_type']), 2)
        for category in result['by_class_and_type']:
            self.assertEqual(set(category['counts']), set(core.LABELS))
            self.assertEqual(sum(category['counts'].values()), category['n'])
            self.assertEqual(sum(category['percentages'].values()), 100)

    def test_missing_duplicate_or_reordered_aggregate_coverage_rejected(self):
        for records in [self.joined[:-1], self.joined[:-1] + [self.joined[0]], self.joined[::-1]]:
            with self.assertRaises(RuntimeError):
                core.aggregate_policy('sft', self.rows, records)

    def test_manual_agreement_counts_ambiguous_and_has_five_by_five_matrix(self):
        ids = [100, 102, 104, 106]
        human = {100: 'SAFE_ANSWER', 102: 'AMBIGUOUS', 104: 'JUSTIFIED_REFUSAL', 106: 'SAFE_ANSWER'}
        result = core.manual_agreement(self.joined, ids, human)
        self.assertEqual(result['matching_labels'], 2)
        self.assertEqual(result['n'], 4)
        self.assertEqual(result['agreement'], 0.5)
        self.assertEqual(result['confusion_counts']['AMBIGUOUS']['AMBIGUOUS'], 1)
        self.assertEqual(result['confusion_counts']['SAFE_ANSWER']['AMBIGUOUS'], 1)
        self.assertEqual(sum(sum(row.values()) for row in result['confusion_counts'].values()), 4)

    def test_manual_metrics_remain_pending_until_independent_labels_complete(self):
        self.assertEqual(core.manual_agreement(self.joined, [100, 102], None)['status'], 'PENDING_HUMAN_LABELS')
        with self.assertRaisesRegex(RuntimeError, 'human label coverage'):
            core.manual_agreement(self.joined, [100, 102], {100: 'SAFE_ANSWER'})

    def test_actual_task_contract_cannot_accept_reduced_fixture_coverage(self):
        with self.assertRaisesRegex(RuntimeError, 'Task 4 prompt coverage differs'):
            core.make_contract(self.rows, 'a' * 64, {'frozen': 'evidence'}, 99)


if __name__ == '__main__':
    unittest.main(verbosity=2)
