"""Streaming Task 1 metrics using the released scoring/generation helpers.

Caller supplies verified original pools, a frozen policy and durable emitters.
No model loading, data selection, tuning or GPU allocation occurs on import.
"""
import copy
import hashlib
import torch
from common.data import (encode_prompt_response, pad_batch, preference_responses,
                         prompt_messages, prompt_messages_from_preference)
from common.generation import batch_generate, response_token_logprobs
from common.logging_utils import set_seed
from common.metrics import word_count
from common.models import reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.runtime import PairMetrics, SampledKL, device_batch, pair_scores
from task1_dpo.evaluation_support import (STRATA, checked_reward_scores, generated_record,
    generation_input, numeric_summary, partition_strata, word_result)


def pool_messages(row, pool_key):
    if pool_key == "dpo_standard_eval":
        return prompt_messages_from_preference(row)
    if pool_key == "word_limit_prompts":
        return prompt_messages(row)
    raise ValueError("Unexpected generation/reward pool.")


def complete_pair_batches(tokenizer, rows, max_length):
    """Require exact audited prompt + complete answer + EOS before scoring."""
    chosen, rejected = [], []
    for row in rows:
        messages = prompt_messages_from_preference(row)
        prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        for destination, response in zip((chosen, rejected), preference_responses(row)):
            answer_ids = tokenizer(response, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            if len(prompt_ids) + len(answer_ids) > max_length:
                raise ValueError("A held-out response would be shortened; stop instead of truncating.")
            encoded = encode_prompt_response(tokenizer, messages, response, max_length)
            expected = (prompt_ids + answer_ids, [0] * len(prompt_ids) + [1] * len(answer_ids))
            if encoded != expected:
                raise ValueError("Active encoder changed the complete prompt/answer boundary.")
            destination.append(encoded)
    return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)


@torch.no_grad()
def evaluate_pair_pool(model, tokenizer, rows, cfg, beta, emit, *, stratified=False):
    if not rows:
        raise ValueError("Empty pair evaluation pool.")
    if stratified:
        partition_strata(rows)
    model.eval()
    device = next(model.parameters()).device
    overall = PairMetrics(beta)
    groups = {name: PairMetrics(beta) for name in STRATA} if stratified else {}
    width = int(cfg["batch_size"])
    if width <= 0:
        raise ValueError("Invalid pair evaluation batch size.")
    for start in range(0, len(rows), width):
        subset = rows[start:start + width]
        cpu_batches = complete_pair_batches(tokenizer, subset, cfg["max_sequence_length"])
        batches = tuple(device_batch(batch, device) for batch in cpu_batches)
        with reference_mode(model):
            reference = pair_scores(model, batches)
        policy = pair_scores(model, batches)
        overall.add(*policy, *reference)
        margin = (policy[0] - policy[1]) - (reference[0] - reference[1])
        for offset, row in enumerate(subset):
            scores = tuple(t[offset:offset + 1] for t in (*policy, *reference))
            loss, _ = dpo_loss(*scores, beta)
            if stratified:
                groups[row["length_stratum"]].add(*scores)
            emit({"row_index": start + offset, "prompt_id": row["prompt_id"],
                  "length_stratum": row.get("length_stratum"),
                  "policy_chosen_logp": float(scores[0].item()),
                  "policy_rejected_logp": float(scores[1].item()),
                  "reference_chosen_logp": float(scores[2].item()),
                  "reference_rejected_logp": float(scores[3].item()),
                  "reference_adjusted_margin": float(margin[offset].item()),
                  "dpo_loss": float(loss.item()), "correct": bool(margin[offset].item() > 0),
                  "chosen_response_tokens": int(cpu_batches[0]["response_mask"][offset].sum().item()),
                  "rejected_response_tokens": int(cpu_batches[1]["response_mask"][offset].sum().item())})
    result = overall.result()
    if result["pairs"] != len(rows):
        raise RuntimeError("Incomplete pair evaluation.")
    if stratified:
        result["strata"] = {name: metric.result() for name, metric in groups.items()}
    return result


@torch.no_grad()
def generate_pool(model, tokenizer, rows, cfg, pool_key, emit):
    if not rows:
        raise ValueError("Empty generation pool.")
    if cfg["generation_batch_size"] != 1:
        raise ValueError("Generation must use the approved batch size of one.")
    model.eval()
    inherited = model.generation_config.to_dict()
    effective = copy.deepcopy(model.generation_config)
    effective.update(max_new_tokens=cfg["max_generation_tokens"],
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, **cfg["generation"])
    set_seed(int(cfg["seed"]))  # Reset once per model and fixed prompt pool.
    kl = SampledKL()
    lengths, words, compliance = [], [], []
    for index, row in enumerate(rows):
        messages = pool_messages(row, pool_key)
        text, prompt_ids = generation_input(tokenizer, messages, cfg["max_prompt_length"])
        generated = batch_generate(model, tokenizer, [messages],
            max_prompt_length=cfg["max_prompt_length"], max_new_tokens=cfg["max_generation_tokens"],
            **cfg["generation"])
        if (generated["prompt_width"] != len(prompt_ids) or
                generated["sequences"][0, :len(prompt_ids)].cpu().tolist() != prompt_ids):
            raise ValueError("Generation helper altered the complete prompt.")
        record = generated_record(generated, tokenizer.eos_token_id, cfg["max_generation_tokens"])
        args = (generated["sequences"], generated["attention_mask"],
                generated["prompt_width"], generated["response_ids"])
        policy_logp, policy_logits = response_token_logprobs(model, *args)
        del policy_logits
        with reference_mode(model):
            reference_logp, reference_logits = response_token_logprobs(model, *args)
        del reference_logits
        mask = generated["response_mask"]
        kl.add(policy_logp, reference_logp, mask)
        active = mask[0].bool()
        differences = (policy_logp - reference_logp)[0, active]
        record.update({"row_index": index, "prompt_id": row["prompt_id"], "messages": messages,
            "prompt_tokens": len(prompt_ids), "rendered_prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "policy_token_logp": policy_logp[0, active].cpu().tolist(),
            "reference_token_logp": reference_logp[0, active].cpu().tolist(),
            "token_logp_difference": differences.cpu().tolist(),
            "logp_difference_sum": float(differences.double().sum().item()),
            "word_count": word_count(record["response"])})
        if pool_key == "word_limit_prompts":
            record.update(word_result(messages, record["response"]))
            compliance.append(record["word_limit_compliance"])
        lengths.append(record["response_length"])
        words.append(record["word_count"])
        emit(record)
        del generated, args, policy_logp, reference_logp, mask, differences
    result = {"responses": len(lengths), **kl.result(),
        "response_length_tokens": numeric_summary(lengths), "response_length_words": numeric_summary(words),
        "inherited_generation_config": inherited, "effective_generation_config": effective.to_dict(),
        "seed_per_pool": cfg["seed"]}
    if result["responses"] != len(rows):
        raise RuntimeError("Incomplete generation pool.")
    if compliance:
        result["word_limit_compliance"] = sum(compliance) / len(compliance)
    return result


@torch.no_grad()
def score_reward_pool(model, tokenizer, records, rows, cfg, pool_key, emit):
    if not rows or len(records) != len(rows):
        raise ValueError("Generated response membership is incomplete.")
    # Check membership before invoking the reward model; duplicate IDs stay separate.
    for index, (record, row) in enumerate(zip(records, rows)):
        if (record["row_index"] != index or record["prompt_id"] != row["prompt_id"] or
                record["messages"] != pool_messages(row, pool_key)):
            raise ValueError("Generated response row/order/prompt differs from the fixed pool.")
        if (len(record["response_token_ids"]) != record["response_length"] or
                tokenizer.decode(record["response_token_ids"], skip_special_tokens=True) != record["response"]):
            raise ValueError("Saved response text differs from its generated tokens.")
    model.eval()
    scores = []
    for index, record in enumerate(records):
        reward, lengths = checked_reward_scores(model, tokenizer, [record["messages"]],
                                               [record["response"]], cfg["reward_max_length"])
        score = float(reward[0].item())
        scores.append(score)
        emit({"row_index": index, "prompt_id": record["prompt_id"],
              "reward_score": score, "reward_input_tokens": lengths[0]})
    return {"responses": len(scores), "reward_score": numeric_summary(scores)}
