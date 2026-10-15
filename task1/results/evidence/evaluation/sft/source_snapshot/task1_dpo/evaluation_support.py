"""Fixed Task 1 evaluation scopes and guards around released metric helpers."""
import math

import torch
from common.generation import score_reward_pairs
from common.metrics import parse_word_limit, word_count, word_limit_compliance
from task1_dpo.support import RUNS, validate_config

STRATA = ("preferred_longer", "length_matched", "rejected_longer")


def evaluation_plan(cfg, name):
    validate_config(cfg)
    if name not in (*RUNS, "sft"):
        raise ValueError("Unknown evaluation condition.")
    pairs = [] if name == "sft" else ["dpo_standard_eval"]
    generation = ["dpo_standard_eval"]
    if name in {"standard", "length_balanced"}:
        pairs.append("dpo_length_eval")
    if name in {"standard", "length_balanced", "sft"}:
        generation.append("word_limit_prompts")
    return {"name": name, "beta": None if name == "sft" else RUNS[name][2],
            "pair_pools": pairs, "generation_pools": generation,
            "seed_per_generation_pool": cfg["seed"],
            "generation_batch_size": cfg["generation_batch_size"]}


def generation_input(tokenizer, messages, max_length):
    """Use the same rendering/tokenization as batch_generate, without cropping."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, truncation=False)["input_ids"]
    if max_length <= 0 or not ids or len(ids) > max_length:
        raise ValueError("Generation prompt exceeds its limit or is empty; no truncation allowed.")
    return text, ids


def reward_input(tokenizer, messages, response, max_length):
    """Check the complete course reward input before calling its scoring helper."""
    text = tokenizer.apply_chat_template(
        list(messages) + [{"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False,
    )
    ids = tokenizer(text, truncation=False)["input_ids"]
    if max_length <= 0 or not ids or len(ids) > max_length:
        raise ValueError("Complete reward input exceeds its limit or is empty; stop instead of cropping.")
    return text, ids


@torch.no_grad()
def checked_reward_scores(model, tokenizer, prompts, responses, max_length):
    if not prompts or len(prompts) != len(responses):
        raise ValueError("Reward prompt/response counts differ or are empty.")
    lengths = [len(reward_input(tokenizer, p, r, max_length)[1])
               for p, r in zip(prompts, responses)]
    scores = score_reward_pairs(model, tokenizer, prompts, responses, max_length=max_length)
    if scores.shape != (len(prompts),) or not torch.isfinite(scores).all().item():
        raise FloatingPointError("Reward scores are nonfinite or have an unexpected shape.")
    return scores, lengths


def numeric_summary(values):
    """Mean and population SD over individual responses, not batch means."""
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Summary requires nonempty finite values.")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return {"count": len(values), "mean": mean, "population_std": math.sqrt(variance)}


def word_result(messages, response):
    prompt = "\n".join(str(message.get("content", "")) for message in messages)
    limit = parse_word_limit(prompt)
    if limit is None:
        raise ValueError("Fixed word-limit prompt has no limit recognized by the course helper.")
    return {"word_limit": limit, "word_count": word_count(response),
            "word_limit_compliance": word_limit_compliance(prompt, response)}


def partition_strata(rows):
    groups = {name: [] for name in STRATA}
    for index, row in enumerate(rows):
        name = row.get("length_stratum")
        if name not in groups:
            raise ValueError("Missing or unknown supplied length stratum.")
        groups[name].append(index)
    return groups


def generated_record(generated, eos_id, max_new_tokens):
    """Save exactly the tokens selected by the released response mask, including EOS."""
    ids, mask = generated["response_ids"], generated["response_mask"]
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape != mask.shape:
        raise ValueError("Approved generation requires one response per batch.")
    if ids.shape[1] == 0 or ids.shape[1] > max_new_tokens:
        raise ValueError("Generated response violates the token ceiling.")
    tokens = ids[0].detach().cpu().tolist()
    first_eos = tokens.index(eos_id) if eos_id in tokens else None
    count = len(tokens) if first_eos is None else first_eos + 1
    expected_mask = [1.] * count + [0.] * (len(tokens) - count)
    if mask[0].detach().cpu().tolist() != expected_mask:
        raise ValueError("Response mask does not match first-EOS boundaries.")
    terminated = first_eos is not None
    capped = count >= max_new_tokens and not terminated
    if generated["response_lengths"] != [count] or generated["terminated_with_eos"] != [terminated] or generated["truncated"] != [capped]:
        raise ValueError("Released generation metadata disagrees with its response tokens.")
    if len(generated["responses"]) != 1:
        raise ValueError("Generated response text count differs.")
    return {"response": generated["responses"][0], "response_token_ids": tokens[:count],
            "response_length": count, "terminated_with_eos": terminated,
            "reached_generation_ceiling": capped}
