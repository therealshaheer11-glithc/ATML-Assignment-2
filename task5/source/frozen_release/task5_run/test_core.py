import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from . import core
from .diagnostics import VARIANTS, group_rows, score_one, controlled_pairs, summarize
from task5_feedback import rlvr


def synthetic_diagnostics():
    responses = ['2 + 2 = 4.\n#### 4', '2 + 2 = 5.\n#### 4', '2 + 2 = 4.\n#### 5',
                 '2 + 2 = 4. Definitely!\n#### 4', 'Reject 4.\n#### 5']
    return [{'problem_id': p, 'variant_type': v, 'question': 'Synthetic four?',
             'gold_final': 4, 'response': text, 'expected_exact_reward': reward}
            for p in range(20) for v, text, reward in zip(VARIANTS, responses, [1, 1, 0, 1, 0])]


def synthetic_eval():
    return {d: [{'source_index': i, 'question': 'Synthetic?', 'gold_final': 4,
                 'messages': [{'role': 'user', 'content': 'Synthetic?'}]} for i in range(n)]
            for d, n in core.COUNTS.items()}


class ScheduleTests(unittest.TestCase):
    def test_1200_unique_units_and_fixed_order(self):
        schedule = core.generation_schedule(synthetic_eval())
        self.assertEqual(len(schedule), 1200)
        self.assertEqual(len(set(schedule)), 1200)
        self.assertEqual(schedule[0][:3], ('sft', 'gsm', 0))
        self.assertEqual(schedule[300][:3], ('sft', 'transfer', 0))
        self.assertEqual(schedule[400][:3], ('rlvr', 'gsm', 0))
        self.assertEqual(schedule[-1][:3], ('rlaif', 'transfer', 99))

    def test_duplicate_source_id_rejected(self):
        rows = synthetic_eval()
        rows['gsm'][1]['source_index'] = 0
        with self.assertRaises(RuntimeError):
            core.generation_schedule(rows)

    def test_wrong_dataset_count_rejected(self):
        rows = synthetic_eval()
        rows['transfer'].pop()
        with self.assertRaises(RuntimeError):
            core.generation_schedule(rows)

    def test_all_880_comparison_events(self):
        from .run import comparison_specs
        rows = synthetic_eval()
        def fake_load(root, policy, domain, source, expected_release_sha=None):
            return ([{'response': f'{policy}:{domain}:{i}', 'exact_reward': int(policy != 'sft')}
                     for i in range(len(source))], None)
        with patch('task5_run.run.load_responses', side_effect=fake_load):
            specs, _ = comparison_specs(Path('/unused'), rows, synthetic_diagnostics(), VARIANTS)
        self.assertEqual(len(specs), 880)
        self.assertEqual(sum(s['kind'] == 'policy' for s in specs), 800)
        self.assertEqual(sum(s['kind'] == 'diagnostic' for s in specs), 80)
        self.assertEqual(specs[0]['verifier_preference'], 'A')


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / 'store'
        self.binding = {'protocol': 'synthetic', 'seed': 6304}

    def tearDown(self):
        self.temporary.cleanup()

    def test_resume_preserves_result_and_rng(self):
        journal = core.Journal(self.path, self.binding)
        journal.append({'answer': 'one'}, {'state': [1, 2]})
        resumed = core.Journal(self.path, self.binding)
        self.assertEqual(resumed.records()[0]['payload'], {'answer': 'one'})
        self.assertEqual(resumed.records()[0]['rng_after'], {'state': [1, 2]})
        resumed.append({'answer': 'two'}, {'state': [3, 4]})
        self.assertEqual(len(core.Journal(self.path, self.binding).records()), 2)

    def test_changed_binding_rejected(self):
        core.Journal(self.path, self.binding)
        with self.assertRaises(RuntimeError):
            core.Journal(self.path, {'seed': 1})

    def test_unknown_unbound_directory_preserved(self):
        self.path.mkdir()
        file = self.path / 'unrelated.txt'
        file.write_text('preserve')
        with self.assertRaises(RuntimeError):
            core.Journal(self.path, self.binding)
        self.assertEqual(file.read_text(), 'preserve')

    def test_partial_write_not_committed_or_deleted(self):
        j = core.Journal(self.path, self.binding)
        j.append({'answer': 'one'})
        pending = j.units / '.pending_interrupted'
        pending.write_text('incomplete')
        resumed = core.Journal(self.path, self.binding)
        self.assertEqual(len(resumed.records()), 1)
        resumed.append({'answer': 'two'})
        self.assertEqual(pending.read_text(), 'incomplete')
        self.assertEqual(len(core.Journal(self.path, self.binding).records()), 2)

    def test_final_record_corruption_rejected(self):
        j = core.Journal(self.path, self.binding)
        j.append({'answer': 'one'})
        path = j.units / '000000.json'
        data = core.read_json(path)
        data['payload']['answer'] = 'changed'
        path.write_bytes(core.json_bytes(data))
        with self.assertRaises(RuntimeError):
            core.Journal(self.path, self.binding)

    def test_chain_gap_rejected(self):
        j = core.Journal(self.path, self.binding)
        j.append({'answer': 'one'})
        (j.units / '000000.json').rename(j.units / '000001.json')
        with self.assertRaises(RuntimeError):
            core.Journal(self.path, self.binding)

    def test_committed_write_not_overwritten(self):
        target = self.path / 'result.json'
        core.write_once(target, {'a': 1})
        with self.assertRaises(RuntimeError):
            core.write_once(target, {'a': 2})
        self.assertEqual(core.read_json(target), {'a': 1})

    def test_rng_stream_resumes_exactly_on_cpu(self):
        import numpy as np
        import torch
        from .inference import capture_rng, restore_rng
        random.seed(6304)
        np.random.seed(6304)
        torch.manual_seed(6304)
        saved = json.loads(json.dumps(capture_rng()))
        expected = (random.random(), np.random.rand(), torch.rand(3).tolist())
        for _ in range(7):
            random.random(); np.random.rand(); torch.rand(3)
        restore_rng(saved)
        actual = (random.random(), np.random.rand(), torch.rand(3).tolist())
        self.assertEqual(actual, expected)


class MetricTests(unittest.TestCase):
    def test_half_ties_and_three_way_agreement(self):
        pairs = [{'verifier_preference': v, 'ai_preference': a, 'unparseable': u}
                 for v, a, u in [('A', 'A', False), ('TIE', 'TIE', True), ('B', 'A', False), ('A', 'B', False)]]
        out = core.preference_summary(pairs)
        self.assertEqual(out['ai_win_rate'], .625)
        self.assertEqual(out['verifier_judge_agreement'], .5)
        self.assertEqual(out['unparseable_ai_outputs'], 1)
        self.assertEqual(out['decisive_verifier_pairs'], 3)
        self.assertAlmostEqual(out['secondary_agreement_decisive_verifier'], 1/3)
        self.assertEqual(sum(sum(r.values()) for r in out['verifier_by_ai_confusion_matrix'].values()), 4)

    def test_all_ties_have_no_secondary_denominator(self):
        out = core.preference_summary([{'verifier_preference': 'TIE', 'ai_preference': 'TIE', 'unparseable': False}])
        self.assertIsNone(out['secondary_agreement_decisive_verifier'])
        self.assertEqual(out['ai_win_rate'], .5)

    def test_population_sd_and_wrong_but_formatted(self):
        rows = [{'response_tokens': n, 'generation_seconds': 1., 'exact_reward': score,
                 'format_compliant': True, 'truncated': False} for n, score in [(2, 0), (4, 1)]]
        out = core.policy_summary(rows)
        self.assertEqual(out['length_mean'], 3)
        self.assertEqual(out['length_population_sd'], 1)
        self.assertEqual(out['exact_accuracy'], .5)
        self.assertEqual(out['format_compliance'], 1)
        self.assertIsNone(out['ai_comparison'])

    def test_transfer_drop_percentage_points(self):
        a = {'exact_accuracy': .8, 'length_mean': 30, 'ai_comparison': {'ai_win_rate': .7}}
        b = {'exact_accuracy': .6, 'length_mean': 20, 'ai_comparison': {'ai_win_rate': .5}}
        out = core.transfer_drop(a, b)
        self.assertAlmostEqual(out['exact_accuracy_drop_percentage_points'], 20)
        self.assertAlmostEqual(out['ai_win_rate_drop_percentage_points'], 20)

    def test_diagnostic_rows_overlap_not_160_pairs(self):
        pairs = [{'problem_id': str(i), 'variant': v, 'verifier_preference': 'TIE',
                  'ai_preference': 'A', 'unparseable': False} for i in range(20) for v in VARIANTS[1:]]
        out = core.diagnostic_summary(pairs, VARIANTS)
        self.assertEqual(out['unique_pairs'], 80)
        self.assertEqual(out['clean_reference_all_contrasts']['pairs'], 80)
        self.assertEqual(out['S_reason'], 1)
        self.assertEqual(out['S_outcome'], 1)


class VerifierDiagnosticTests(unittest.TestCase):
    def test_gold_distractor_not_rewarded(self):
        self.assertEqual(rlvr.exact_reward('Reject 4.\n#### 5', 4), 0)

    def test_last_final_and_commas(self):
        self.assertEqual(rlvr.exact_reward('#### 9\n#### -1,234.5', -1234.5), 1)

    def test_missing_format(self):
        self.assertIsNone(rlvr.extract_designated_final('The answer is 4.'))

    def test_numeric_tolerance(self):
        self.assertEqual(rlvr.exact_reward('#### 1.0000000005', 1), 1)
        self.assertEqual(rlvr.exact_reward('#### 1.000000002', 1), 0)

    def test_diagnostic_exact_metrics(self):
        rows = synthetic_diagnostics()
        group_rows(rows)
        scores = [score_one(r, i, rlvr) for i, r in enumerate(rows)]
        out = summarize(scores, controlled_pairs(scores))
        self.assertEqual(out['unique_controlled_pairs'], 80)
        self.assertEqual(out['verifier_S_reason'], 0)
        self.assertEqual(out['verifier_S_outcome'], 1)
        self.assertIsNone(out['ai_metrics'])

    def test_duplicate_diagnostic_rejected(self):
        rows = synthetic_diagnostics()
        rows[-1] = copy.deepcopy(rows[0])
        with self.assertRaises(RuntimeError):
            group_rows(rows)

    def test_outcome_reasoning_mismatch_rejected(self):
        rows = synthetic_diagnostics()
        rows[2]['response'] = 'Changed reasoning.\n#### 5'
        with self.assertRaises(RuntimeError):
            group_rows(rows)

    def test_wrong_metadata_does_not_override_score(self):
        row = synthetic_diagnostics()[0]
        row['expected_exact_reward'] = 0
        out = score_one(row, 0, rlvr)
        self.assertEqual(out['exact_reward'], 1)
        self.assertFalse(out['matches_course_expected_reward'])


class GateTests(unittest.TestCase):
    def test_no_execution_without_approval_file(self):
        from .runtime import execution_gate
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'approval has not been recorded'):
                execution_gate(Path(folder), Path(folder), 'generation')


if __name__ == '__main__':
    unittest.main()
