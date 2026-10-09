"""CPU regressions for original archived records and JSON metadata compatibility."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / 'work/chunk3-validation'
sys.path.insert(0, str(VALIDATION))
sys.path.insert(0, str(VALIDATION / 'tests'))
sys.path.insert(0, str(Path(__file__).parent))
from task2_resume_json import canonical_contract, evaluation_wrapper, guard_original, replacement_command
from task2_ppo.evaluation_complete import evaluate_condition
from task2_ppo.session_budget import UnlimitedTestBudget
from task2_ppo.storage import atomic_json, sha256
from test_task2_complete import metric_row
from transformers import Qwen2Config


class ResumeJSONTests(unittest.TestCase):
    def test_transformers_integer_label_key_reproduces_and_is_normalized(self):
        live = {'reward_loading': {'effective_config': Qwen2Config(num_labels=1).to_dict()}}
        saved = json.loads(json.dumps(live))
        self.assertNotEqual(live, saved)
        self.assertIn(0, live['reward_loading']['effective_config']['id2label'])
        self.assertEqual(canonical_contract(live), saved)

    def test_original_archive_resumes_without_generation_or_record_changes(self):
        entry = metric_row()['schedule']
        live = {'reward_loading': {'effective_config': {'id2label': {0: 'LABEL_0'}, 'rope_theta': 1000000.}},
                'generation': {'max_response_length': 768}, 'adapter': {'weights': 'fixed'}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = (None, None, None, [{}], {'evaluation': [entry]}, {}, 'midpoint', live,
                    root / 'local', root / 'drive', UnlimitedTestBudget())
            with mock.patch('task2_ppo.evaluation_complete.evaluate_prompt', return_value=metric_row()) as generate:
                evaluate_condition(*args)  # Original first session writes integer map through JSON.
                self.assertEqual(generate.call_count, 1)
                archive = root / 'drive/midpoint/prompt_0000.zip'
                before = sha256(archive)
                with self.assertRaisesRegex(RuntimeError, 'Evaluation contract changed'):
                    evaluate_condition(*args)  # Original resume bug.
                result = evaluation_wrapper(evaluate_condition)(*args)
                self.assertEqual(generate.call_count, 1)
                self.assertEqual(result['metrics']['prompts'], 1)
                self.assertEqual(sha256(archive), before)
                record = json.loads((root / 'local/midpoint/prompt_0000/record.json').read_text())
                self.assertEqual(record['contract'], canonical_contract(live))

    def test_real_setting_changes_still_stop_before_original_function(self):
        original = mock.Mock()
        base = {'generation': {'max_response_length': 768}, 'adapter': {'weights': 'fixed'},
                'job': {'source': {'commit': 'fixed'}},
                'reward_loading': {'effective_config': {'id2label': {0: 'LABEL_0'}, 'rope_theta': 1000000.}}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_json(root / 'drive/midpoint/contract.json', base)
            mutations = [('cap', lambda c: c['generation'].update(max_response_length=512)),
                ('adapter', lambda c: c['adapter'].update(weights='changed')),
                ('source', lambda c: c['job']['source'].update(commit='changed')),
                ('theta', lambda c: c['reward_loading']['effective_config'].update(rope_theta=10000.))]
            before = (root / 'drive/midpoint/contract.json').read_bytes()
            for name, mutate in mutations:
                with self.subTest(name=name):
                    value = copy.deepcopy(base)
                    mutate(value)
                    with self.assertRaisesRegex(RuntimeError, 'really differ'):
                        evaluation_wrapper(original)(None,None,None,[],{}, {}, 'midpoint', value,
                            root / 'local',root / 'drive', UnlimitedTestBudget())
            original.assert_not_called()
            self.assertEqual((root / 'drive/midpoint/contract.json').read_bytes(), before)

    def test_keyword_contract_invocation_preserves_every_field(self):
        original = mock.Mock(return_value='retained')
        live = {'labels': {0: 'x'}, 'temperature': .7, 'caps': [256, 768]}
        with tempfile.TemporaryDirectory() as temporary:
            result = evaluation_wrapper(original)(name='midpoint', contract=live,
                output_root=Path(temporary)/'local', persistent_root=Path(temporary)/'drive')
        self.assertEqual(result, 'retained')
        self.assertEqual(original.call_args.kwargs['contract'], canonical_contract(live))
        self.assertEqual(live['labels'], {0: 'x'})

    def test_original_source_guard_accepts_exact_hash_and_rejects_change(self):
        guard_original(VALIDATION)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'task2_ppo').mkdir()
            (root/'task2_ppo/evaluation_complete.py').write_text('changed')
            with self.assertRaises(RuntimeError):
                guard_original(root)

    def test_only_pipeline_subprocess_gets_compatibility_entry(self):
        original=['python','-u','-m','task2_ppo.pipeline','--session-minutes','120','--models-dir','fixed']
        fixed=replacement_command(original)
        self.assertEqual(fixed,['python','-u','-m','scripts.task2_resume_json','--pipeline-entry',
                               '--session-minutes','120','--models-dir','fixed'])
        self.assertEqual(original[3], 'task2_ppo.pipeline')
        pip=['python','-m','pip','install','-r','requirements.txt']
        self.assertEqual(replacement_command(pip),pip)

    def test_nonfinite_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            canonical_contract({'bad': float('nan')})


if __name__ == '__main__':
    unittest.main(verbosity=2)
