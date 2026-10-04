"""CPU boundary checks; all responses and tokenizer examples are synthetic."""
import copy
import unittest
from unittest.mock import patch

import torch
from common.data import load_yaml
from task1_dpo import evaluation_support as support


class TokenizerFixture:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        return {"input_ids": list(text.encode())}


class EvaluationSupportTests(unittest.TestCase):
    def test_scopes_keep_beta_runs_matched_and_baseline_untouched(self):
        with patch.object(support, "validate_config"):
            cfg = {"seed": 6304, "generation_batch_size": 1}
            for name in ("beta_003", "beta_010", "beta_030"):
                plan = support.evaluation_plan(cfg, name)
                self.assertEqual(plan["pair_pools"], ["dpo_standard_eval"])
                self.assertEqual(plan["generation_pools"], ["dpo_standard_eval"])
            for name in ("standard", "length_balanced"):
                plan = support.evaluation_plan(cfg, name)
                self.assertEqual(plan["pair_pools"], ["dpo_standard_eval", "dpo_length_eval"])
                self.assertEqual(plan["generation_pools"], ["dpo_standard_eval", "word_limit_prompts"])
            baseline = support.evaluation_plan(cfg, "sft")
            self.assertEqual(baseline["pair_pools"], [])
            self.assertIsNone(baseline["beta"])
            self.assertEqual(baseline["generation_pools"], ["dpo_standard_eval", "word_limit_prompts"])

    def test_prompt_boundary_preserves_complete_input(self):
        tok, messages = TokenizerFixture(), [{"role": "user", "content": "question"}]
        text, ids = support.generation_input(tok, messages, 100)
        self.assertEqual(support.generation_input(tok, messages, len(ids)), (text, ids))
        with self.assertRaises(ValueError):
            support.generation_input(tok, messages, len(ids) - 1)

    def test_reward_boundary_keeps_prompt_answer_and_template(self):
        tok, messages = TokenizerFixture(), [{"role": "user", "content": "question"}]
        original = copy.deepcopy(messages)
        text, ids = support.reward_input(tok, messages, "answer", 100)
        self.assertIn("question", text)
        self.assertIn("answer", text)
        self.assertEqual(support.reward_input(tok, messages, "answer", len(ids)), (text, ids))
        with self.assertRaises(ValueError):
            support.reward_input(tok, messages, "answer", len(ids) - 1)
        self.assertEqual(messages, original)

    def test_overlong_reward_stops_before_the_model_is_called(self):
        with patch.object(support, "score_reward_pairs", side_effect=AssertionError("Model should not run")):
            with self.assertRaises(ValueError):
                support.checked_reward_scores(None, TokenizerFixture(),
                    [[{"role": "user", "content": "question"}]], ["answer"], 1)

    def test_nonfinite_rewards_cannot_be_reported(self):
        with patch.object(support, "score_reward_pairs", return_value=torch.tensor([float("nan")])):
            with self.assertRaises(FloatingPointError):
                support.checked_reward_scores(None, TokenizerFixture(),
                    [[{"role": "user", "content": "question"}]], ["answer"], 100)

    def test_generated_eos_is_counted_later_padding_is_excluded(self):
        generated = {"response_ids": torch.tensor([[5, 9, 0]]),
                     "response_mask": torch.tensor([[1., 1., 0.]]),
                     "response_lengths": [2], "terminated_with_eos": [True],
                     "truncated": [False], "responses": ["fixture"]}
        record = support.generated_record(generated, 9, 3)
        self.assertEqual(record["response_token_ids"], [5, 9])
        self.assertEqual(record["response_length"], 2)
        generated["response_mask"] = torch.ones(1, 3)
        with self.assertRaises(ValueError):
            support.generated_record(generated, 9, 3)

    def test_generation_ceiling_is_distinct_from_eos_termination(self):
        generated = {"response_ids": torch.tensor([[1, 2, 3]]),
                     "response_mask": torch.ones(1, 3), "response_lengths": [3],
                     "terminated_with_eos": [False], "truncated": [True], "responses": ["fixture"]}
        record = support.generated_record(generated, 9, 3)
        self.assertTrue(record["reached_generation_ceiling"])
        self.assertFalse(record["terminated_with_eos"])

    def test_word_rule_uses_course_count_and_inclusive_limit(self):
        result = support.word_result([{"role": "user", "content": "Answer in under 3 words."}], "one two three")
        self.assertEqual(result, {"word_limit": 3, "word_count": 3, "word_limit_compliance": 1.})
        with self.assertRaises(ValueError):
            support.word_result([{"role": "user", "content": "No specified limit"}], "answer")

    def test_population_statistics_and_supplied_strata_preserve_rows(self):
        self.assertEqual(support.numeric_summary([1, 3]), {"count": 2, "mean": 2., "population_std": 1.})
        with self.assertRaises(ValueError):
            support.numeric_summary([float("nan")])
        rows = [{"prompt_id": "duplicate", "length_stratum": "preferred_longer"},
                {"prompt_id": "duplicate", "length_stratum": "length_matched"},
                {"prompt_id": "other", "length_stratum": "preferred_longer"}]
        self.assertEqual(support.partition_strata(rows), {"preferred_longer": [0, 2], "length_matched": [1], "rejected_longer": []})


if __name__ == "__main__":
    unittest.main()
