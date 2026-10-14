"""Use real CPU tensors and unchanged course helpers with synthetic model/tokenizer doubles."""
import ast
import hashlib
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import torch
from . import core, inference


class SyntheticTokenizer:
    pad_token_id, eos_token_id = 0, 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        if kwargs.get('return_tensors') == 'pt':
            return torch.tensor([[11, 12, 13]])
        return 'full unchanged prompt'

    def __call__(self, rendered, return_tensors, padding, truncation, max_length=None):
        ids = [11, 12, 13]
        if truncation:
            ids = ids[:max_length]
        return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones(1, len(ids), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return 'Synthetic reasoning.\n#### 4'


class SyntheticModel(torch.nn.Module):
    def __init__(self, tail=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.tail = tail if tail is not None else [7, 2, 0, 0]
        self.received = None

    def generate(self, *args, **kwargs):
        self.received = kwargs.copy()
        ids = args[0] if args else kwargs['input_ids']
        return torch.cat([ids, torch.tensor([self.tail])], dim=1)


def synthetic_judge(raw):
    source = Path(__file__).resolve().parents[1] / 'task5_feedback/rlaif.py'
    tree = ast.parse(source.read_text())
    # Execute the EXACT released class/rubric with CPU doubles; never run its model-loading initializer.
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef)
                or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PAIRWISE_RUBRIC' for t in n.targets)]
    namespace = {'torch': torch, 'hashlib': hashlib, 'json': json, 're': re, 'Path': Path}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    judge = namespace['PairwiseAIJudge'].__new__(namespace['PairwiseAIJudge'])
    judge.cfg = {'ai_judge_model': 'Qwen/Qwen2.5-3B-Instruct'}
    judge.cache, judge.cache_path = {}, inference.MemoryCachePath()
    tokenizer = SyntheticTokenizer()
    tokenizer.decode = lambda *a, **k: raw
    judge.tokenizer = inference.CaptureTokenizer(tokenizer)
    judge.model = inference.TimedModel(SyntheticModel([7, 2]), 1000)
    return judge


class GenerationTests(unittest.TestCase):
    def test_full_prompt_actual_course_helper_and_frozen_kwargs(self):
        model = inference.TimedModel(SyntheticModel(), 1000)
        model.model.eval()
        row = {'question': 'Synthetic?', 'messages': [{'role': 'user', 'content': 'Synthetic?'}]}
        with patch('torch.cuda.synchronize'):
            out = inference.generate_one(model, SyntheticTokenizer(), row)
        self.assertEqual(out['prompt_tokens'], 3)
        self.assertEqual(out['response_tokens'], 2)
        self.assertEqual(out['generated_token_ids'], [7, 2])
        self.assertTrue(out['terminated_with_eos'])
        self.assertFalse(out['truncated'])
        self.assertEqual(model.model.received['max_new_tokens'], 512)
        self.assertEqual(model.model.received['temperature'], .7)
        self.assertEqual(model.model.received['top_p'], .9)
        self.assertTrue(model.model.received['do_sample'])
        self.assertEqual(model.calls[0]['prompt_ids'], [11, 12, 13])

    def test_context_overflow_stops_before_generate(self):
        model = inference.TimedModel(SyntheticModel(), 514)
        with self.assertRaisesRegex(RuntimeError, 'context budget'):
            inference.generate_one(model, SyntheticTokenizer(), {'question': 'Synthetic?'})
        self.assertEqual(model.calls, [])

    def test_first_eos_counted_padding_excluded(self):
        self.assertEqual(inference.response_length([9, 2, 0, 0], 2), 2)
        self.assertEqual(inference.response_length([2, 2, 2], 2), 1)

    def test_no_eos_keeps_all_generated_tokens(self):
        self.assertEqual(inference.response_length([8, 9], 2), 2)
        self.assertEqual(inference.response_length([8, 9], None), 2)

    def test_helper_marks_completion_cap_without_eos(self):
        model = inference.TimedModel(SyntheticModel([7] * 512), 1000)
        with patch('torch.cuda.synchronize'):
            out = inference.generate_one(model, SyntheticTokenizer(), {'question': 'Synthetic?'})
        self.assertEqual(out['response_tokens'], 512)
        self.assertTrue(out['truncated'])


class JudgeTests(unittest.TestCase):
    def test_canonical_key_matches_released_helper(self):
        judge = synthetic_judge('A')
        self.assertEqual(judge._key('question', 'answer_a', 'answer_b'),
                         core.judge_key('question', 'answer_a', 'answer_b'))

    def test_local_path_would_change_key_and_is_rejected(self):
        judge = synthetic_judge('A')
        judge.cfg['ai_judge_model'] = '/local/snapshot'
        with self.assertRaisesRegex(RuntimeError, 'Canonical'):
            inference.judge_one(judge, {}, 'q', 'a', 'b')

    def test_both_swap_orientations_and_four_token_limit(self):
        orientations = set()
        for i in range(20):
            judge = synthetic_judge('A')
            with patch('torch.cuda.synchronize'):
                out = inference.judge_one(judge, {}, str(i), 'a', 'b')
            orientations.add(out['swapped'])
            self.assertEqual(out['ai_preference'], 'B' if out['swapped'] else 'A')
            self.assertEqual(judge.model.model.received['max_new_tokens'], 4)
            self.assertFalse(judge.model.model.received['do_sample'])
        self.assertEqual(orientations, {True, False})

    def test_unparseable_is_logged_and_preserved_as_tie(self):
        judge = synthetic_judge('unclear')
        with patch('torch.cuda.synchronize'):
            out = inference.judge_one(judge, {}, 'q', 'a', 'b')
        self.assertEqual(out['ai_preference'], 'TIE')
        self.assertTrue(out['unparseable'])
        self.assertEqual(out['raw_judge_output'], 'unclear')

    def test_explicit_tie_is_not_ambiguous(self):
        judge = synthetic_judge(' tie ')
        with patch('torch.cuda.synchronize'):
            out = inference.judge_one(judge, {}, 'q', 'a', 'b')
        self.assertEqual(out['ai_preference'], 'TIE')
        self.assertFalse(out['unparseable'])

    def test_cache_reuse_adds_no_generate_call_and_keeps_raw_output(self):
        judge, captured = synthetic_judge('A'), {}
        with patch('torch.cuda.synchronize'):
            first = inference.judge_one(judge, captured, 'q', 'a', 'b')
            second = inference.judge_one(judge, captured, 'q', 'a', 'b')
        self.assertEqual(len(judge.model.calls), 1)
        self.assertTrue(second['cache_hit'])
        self.assertEqual(second['judge_seconds'], 0)
        self.assertEqual(second['raw_judge_output'], first['raw_judge_output'])

    def test_cache_without_raw_evidence_is_rejected(self):
        judge = synthetic_judge('A')
        judge.cache[core.judge_key('q', 'a', 'b')] = 'TIE'
        with self.assertRaisesRegex(RuntimeError, 'raw-output evidence'):
            inference.judge_one(judge, {}, 'q', 'a', 'b')

    def test_course_first_label_regex_is_preserved(self):
        judge = synthetic_judge('A or B')
        with patch('torch.cuda.synchronize'):
            out = inference.judge_one(judge, {}, 'q', 'a', 'b')
        self.assertFalse(out['unparseable'])
        self.assertEqual(out['ai_preference'], 'B' if out['swapped'] else 'A')


if __name__ == '__main__':
    unittest.main()
