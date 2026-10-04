"""Independent small-number checks of the released scoring helpers."""
import math
import unittest
from types import SimpleNamespace

import torch
from common.data import encode_prompt_response, pad_batch
from common.generation import (
    _response_mask, response_sequence_logprobs, response_token_logprobs,
)


class ToyTokenizer:
    pad_token_id = 0
    eos_token = " <eos>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 2, 3, 4, 5]

    def __call__(self, text, add_special_tokens=False):
        return {
            "input_ids": [
                9 if x == "<eos>" else int(x) for x in text.split()
            ]
        }


class FixedLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        assert self.logits.shape[:2] == input_ids.shape
        assert use_cache is False and return_dict is True
        return SimpleNamespace(logits=self.logits)


def fixture():
    # Both responses are [3, 4]; 4 represents EOS.
    batch = pad_batch(ToyTokenizer(), [
        ([1, 2, 3, 4], [0, 0, 1, 1]),
        ([1, 2, 5, 3, 4], [0, 0, 0, 1, 1]),
    ])
    probabilities = torch.full((2, 5, 6), 1 / 6)
    for row, position, token, probability in (
        (0, 2, 3, 0.2), (0, 3, 4, 0.4),
        (1, 2, 3, 0.6), (1, 3, 4, 0.3),
    ):
        probabilities[row, position] = (1 - probability) / 5
        probabilities[row, position, token] = probability
    return batch, probabilities.log()


class ResponseScoringTests(unittest.TestCase):
    def test_next_token_shift_and_response_sum(self):
        batch, logits = fixture()
        scores, _, mask = response_sequence_logprobs(
            FixedLogits(logits), batch
        )
        # Independent answer: log P(first response token) + log P(EOS).
        expected = torch.tensor([
            math.log(.2) + math.log(.4),
            math.log(.6) + math.log(.3),
        ])
        torch.testing.assert_close(scores, expected, rtol=0, atol=1e-6)
        self.assertEqual(mask.sum(-1).tolist(), [2., 2.])

    def test_prompt_and_padding_predictions_do_not_affect_score(self):
        batch, logits = fixture()
        a, _, _ = response_sequence_logprobs(FixedLogits(logits), batch)
        logits[:, :2, :] = torch.arange(6).float() * 10
        b, _, _ = response_sequence_logprobs(FixedLogits(logits), batch)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_only_response_predictions_receive_gradients(self):
        batch, logits = fixture()
        model = FixedLogits(logits)
        scores, _, _ = response_sequence_logprobs(model, batch)
        scores.sum().backward()
        grad = model.logits.grad.abs().sum(-1)
        self.assertTrue((grad[:, 2:4] > 0).all().item())
        self.assertEqual(grad[:, [0, 1, 4]].sum().item(), 0)

    def test_left_padding_preserves_both_masks(self):
        batch, _ = fixture()
        self.assertEqual(batch["input_ids"][0].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(
            batch["attention_mask"][0].tolist(), [0, 1, 1, 1, 1]
        )
        self.assertEqual(
            batch["response_mask"][0].tolist(), [0, 0, 0, 1, 1]
        )
        self.assertEqual(
            batch["attention_mask"][1].tolist(), [1, 1, 1, 1, 1]
        )

    def test_generated_token_scores_match_sequence_scores(self):
        batch, logits = fixture()
        model = FixedLogits(logits)
        sequence_score, _, _ = response_sequence_logprobs(model, batch)
        token_scores, _ = response_token_logprobs(
            model, batch["input_ids"], batch["attention_mask"],
            prompt_width=3, response_ids=batch["input_ids"][:, 3:],
        )
        torch.testing.assert_close(token_scores.sum(-1), sequence_score)

    def test_eos_included_and_later_positions_excluded(self):
        ids = torch.tensor([
            [3, 4, 0, 0], [4, 0, 0, 0], [3, 5, 3, 5]
        ])
        mask = _response_mask(ids, eos_id=4)
        self.assertEqual(
            mask.tolist(),
            [[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]],
        )
        self.assertEqual(mask.sum(-1).tolist(), [2., 1., 4.])
        self.assertTrue(
            (_response_mask(ids, eos_id=None) == 1).all().item()
        )

    def test_response_encoding_appends_eos(self):
        ids, mask = encode_prompt_response(
            ToyTokenizer(), [], "6 7", max_length=20
        )
        self.assertEqual(ids, [1, 2, 3, 4, 5, 6, 7, 9])
        self.assertEqual(mask, [0, 0, 0, 0, 0, 1, 1, 1])

    def test_prompt_truncation_preserves_response(self):
        ids, mask = encode_prompt_response(
            ToyTokenizer(), [], "6 7 8", max_length=6
        )
        self.assertEqual(ids, [4, 5, 6, 7, 8, 9])
        self.assertEqual(mask, [0, 0, 1, 1, 1, 1])

    def test_overlong_response_release_boundary(self):
        ids, mask = encode_prompt_response(
            ToyTokenizer(), [], "1 2 3 4 5 6 7", max_length=6,
        )
        # Released behavior retains the response tail above the cap.
        self.assertEqual(ids, [3, 4, 5, 6, 7, 9])
        self.assertEqual(mask, [1, 1, 1, 1, 1, 1])
        # First retained token has no preceding position to predict it.
        batch = pad_batch(ToyTokenizer(), [(ids, mask)])
        score, _, shifted_mask = response_sequence_logprobs(
            FixedLogits(torch.zeros((1, 6, 10))), batch,
        )
        self.assertEqual(shifted_mask.sum().item(), 5)
        self.assertAlmostEqual(score.item(), -5 * math.log(10), places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
