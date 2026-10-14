import contextlib
import io
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from . import core, inference, run
from .test_inference import SyntheticModel, SyntheticTokenizer


class GenerationConfig:
    def __init__(self):
        self.values = {'num_return_sequences': 1, 'top_k': 20}

    def update(self, **kwargs):
        self.values.update(kwargs)

    def to_dict(self):
        return dict(self.values)


class RandomSyntheticModel(SyntheticModel):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(max_position_embeddings=1000)
        self.generation_config = GenerationConfig()

    def generate(self, *args, **kwargs):
        self.tail = [int(torch.randint(10, 100, (1,)).item()),
                     int(np.random.randint(10, 100)), int(random.random() * 100), 2]
        return super().generate(*args, **kwargs)


class OrchestrationChecks(unittest.TestCase):
    def execute(self, project, interrupt_after=None):
        rows = {d: [{'source_index': i, 'question': 'Synthetic?', 'gold_final': 4,
                     'messages': [{'role': 'user', 'content': 'Synthetic?'}]}
                    for i in range(n)] for d, n in [('gsm', 3), ('transfer', 2)]}
        bindings = {'snapshots': {'policy': {'path': 'unused'}},
                    'adapters': {'rlvr': 'unused_rlvr', 'rlaif': 'unused_rlaif'},
                    'datasets': {'gsm': 'fake_gsm', 'transfer': 'fake_svamp'},
                    'asset_sha256': {'fake_gsm': core.checksum(rows['gsm']),
                                     'fake_svamp': core.checksum(rows['transfer'])}}
        models = types.ModuleType('common.models')
        models.load_policy = lambda *a, **k: RandomSyntheticModel().eval()
        models.load_tokenizer = lambda *a, **k: SyntheticTokenizer()
        calls = 0
        original = inference.generate_one
        def observed(*args):
            nonlocal calls
            if interrupt_after is not None and calls >= interrupt_after:
                raise RuntimeError('SYNTHETIC_INTERRUPTION')
            calls += 1
            return original(*args)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict('sys.modules', {'common.models': models}))
            stack.enter_context(patch('task5_run.runtime.execution_gate', return_value={'hardware': 'synthetic CPU double'}))
            stack.enter_context(patch('task5_run.run.verify_source', return_value='synthetic_release'))
            stack.enter_context(patch('task5_run.run.inputs', return_value=(bindings, rows)))
            stack.enter_context(patch('task5_run.run.validate_policy'))  # CUDA/device checks require the later A100 stage.
            stack.enter_context(patch('task5_run.run.verify_adapter_loaded'))  # No pretrained adapter is loaded in this test.
            stack.enter_context(patch('task5_run.inference.generate_one', side_effect=observed))
            stack.enter_context(patch('torch.cuda.synchronize'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            run.generate(project, Path('/synthetic_release'))

    def saved(self, project):
        root = project / 'task5/experiment_v1/evaluation_v1/generations'
        saved = []
        for p in core.POLICIES:
            for d in core.DOMAINS:
                folder = root / p / d
                journal = core.Journal(folder, core.read_json(folder / 'BINDING.json'))
                for r in journal.records():
                    payload = dict(r['payload'])
                    payload.pop('generation_seconds')
                    saved.append((payload, r['rng_after']))
        return saved

    def test_interrupted_runner_resumes_to_same_tokens_and_rng_as_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, resumed = Path(tmp) / 'reference', Path(tmp) / 'resumed'
            self.execute(reference)
            with self.assertRaisesRegex(RuntimeError, 'SYNTHETIC_INTERRUPTION'):
                self.execute(resumed, interrupt_after=3)
            self.execute(resumed)
            self.assertEqual(len(self.saved(reference)), 15)
            self.assertEqual(self.saved(reference), self.saved(resumed))

    def test_completed_runner_resume_generates_no_additional_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            self.execute(project)
            before = self.saved(project)
            self.execute(project, interrupt_after=0)
            self.assertEqual(self.saved(project), before)

    def test_generation_defaults_preserve_inherited_top_k(self):
        model, tok = RandomSyntheticModel(), SyntheticTokenizer()
        effective = inference.effective_generation(model, tok)
        self.assertEqual(effective['effective_requested_generation']['top_k'], 20)
        self.assertEqual(model.generation_config.to_dict(), {'num_return_sequences': 1, 'top_k': 20})


if __name__ == '__main__':
    unittest.main()
