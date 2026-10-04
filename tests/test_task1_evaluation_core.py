"""Synthetic CPU integration checks; no public weights or assignment results."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from task1_dpo import evaluation_core as core

class TokenizerFixture:
    eos_token_id, pad_token_id = 1, 0
    def ids(self, text):
        return [2 + (ord(char) % 30) for char in text]
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + ("<assistant>" if add_generation_prompt else "")
        return self.ids(text) if tokenize else text
    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            return {"input_ids": self.ids(text)}
        ids = [self.ids(item) for item in text]
        width = max(map(len, ids))
        return {"input_ids": torch.tensor([[0] * (width - len(x)) + x for x in ids]),
                "attention_mask": torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids])}
    def decode(self, ids, skip_special_tokens=True):
        return " ".join("word" for token in ids if token not in (0, 1))

class RewardFixture(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.values = iter(values)
        self.calls = 0
    def forward(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(logits=torch.tensor([[next(self.values)]], dtype=torch.float32))

CFG = {"batch_size": 2, "max_sequence_length": 256, "seed": 6304,
       "max_prompt_length": 256, "max_generation_tokens": 3, "reward_max_length": 256,
       "generation_batch_size": 1,
       "generation": {"do_sample": True, "temperature": .7, "top_p": .9}}

def rows():
    return [{"prompt_id": "duplicate", "prompt": "q", "chosen": "yes", "rejected": "",
             "length_stratum": name} for name in core.STRATA]

class EvaluationCoreTests(unittest.TestCase):
    def test_pair_tail_strata_and_duplicate_ids_keep_full_denominator(self):
        scores = [(torch.zeros(2), torch.zeros(2)), (torch.tensor([2., -2.]), torch.zeros(2)),
                  (torch.zeros(1), torch.zeros(1)), (torch.zeros(1), torch.zeros(1))]
        saved = []
        with patch.object(core, "pair_scores", side_effect=scores):
            result = core.evaluate_pair_pool(RewardFixture([]), TokenizerFixture(), rows(), CFG, .1, saved.append, stratified=True)
        expected = torch.nn.functional.softplus(torch.tensor([-.2, .2, 0.])).mean().item()
        self.assertAlmostEqual(result["mean_dpo_loss"], expected, places=6)
        self.assertEqual((result["pairs"], result["correct_pairs"], result["ties"]), (3, 1, 1))
        self.assertEqual(result["preference_accuracy"], 1 / 3)
        self.assertEqual([x["row_index"] for x in saved], [0, 1, 2])
        self.assertEqual([x["prompt_id"] for x in saved], ["duplicate"] * 3)
        self.assertTrue(all(x["rejected_response_tokens"] == 1 for x in saved))
        self.assertEqual({k: v["pairs"] for k, v in result["strata"].items()}, {k: 1 for k in core.STRATA})

    def test_actual_tiny_qwen_generation_resets_seed_and_baseline_kl_is_zero(self):
        torch.manual_seed(5)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=512, eos_token_id=1, pad_token_id=0)).eval().requires_grad_(False)
        before = {n: p.clone() for n, p in model.named_parameters()}
        output, repeated = [], []
        first = core.generate_pool(model, TokenizerFixture(), rows()[:2], CFG, "dpo_standard_eval", output.append)
        torch.manual_seed(999)
        second = core.generate_pool(model, TokenizerFixture(), rows()[:2], CFG, "dpo_standard_eval", repeated.append)
        self.assertEqual(first["sampled_kl"], 0.)
        self.assertEqual(first["responses"], 2)
        self.assertEqual(output, repeated)
        self.assertEqual(first, second)
        self.assertEqual(first["valid_response_tokens"], sum(x["response_length"] for x in output))
        word_rows = [{"prompt_id": "duplicate", "prompt": "at most 1 words"}] * 2
        word_records = []
        word_summary = core.generate_pool(model, TokenizerFixture(), word_rows, CFG,
                                          "word_limit_prompts", word_records.append)
        expected_compliance = sum(float(x["word_count"] <= 1) for x in word_records) / 2
        self.assertEqual(word_summary["word_limit_compliance"], expected_compliance)
        self.assertTrue(all(x["word_limit"] == 1 for x in word_records))
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)

    def test_reward_replay_membership_and_population_mean(self):
        tok, source = TokenizerFixture(), rows()[:2]
        records = [{"row_index": i, "prompt_id": row["prompt_id"],
                    "messages": core.pool_messages(row, "dpo_standard_eval"),
                    "response_token_ids": [2], "response_length": 1, "response": "word"}
                   for i, row in enumerate(source)]
        saved = []
        result = core.score_reward_pool(RewardFixture([1.5, -.5]), tok, records, source, CFG, "dpo_standard_eval", saved.append)
        self.assertEqual(result["reward_score"], {"count": 2, "mean": .5, "population_std": 1.})
        model = RewardFixture([])
        bad = copy.deepcopy(records)
        bad[1]["row_index"] = 0
        with self.assertRaisesRegex(ValueError, "row/order/prompt"):
            core.score_reward_pool(model, tok, bad, source, CFG, "dpo_standard_eval", saved.append)
        self.assertEqual(model.calls, 0)

    def test_overlength_pair_stops_before_model_scoring(self):
        cfg = dict(CFG, max_sequence_length=1)
        with patch.object(core, "pair_scores") as score:
            with self.assertRaisesRegex(ValueError, "shortened"):
                core.evaluate_pair_pool(RewardFixture([]), TokenizerFixture(), rows(), cfg, .1, lambda x: None)
            score.assert_not_called()

if __name__ == "__main__":
    unittest.main()
