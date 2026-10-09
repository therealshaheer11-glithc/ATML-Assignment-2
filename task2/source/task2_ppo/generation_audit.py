"""Document inherited pinned decoding defaults; do not alter the course sampler."""


def generation_protocol(model, tokenizer, response_cap, cfg):
    inherited = model.generation_config.to_dict()
    # The approved pinned snapshot supplies a generation_config.json. Preserve
    # its top-k/repetition defaults, which the course helper does not override.
    if (inherited.get("top_k") != 20 or inherited.get("repetition_penalty") != 1.1
            or inherited.get("use_cache") is not True
            or getattr(model.generation_config, "_from_model_config", False)):
        raise RuntimeError("Pinned checkpoint generation defaults differ; stop rather than silently replace them")
    return {"inherited_generation_config": inherited,
        "course_generate_overrides": {"max_new_tokens": response_cap, **cfg["generation"],
            "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id},
        "max_prompt_length": cfg["max_prompt_length"],
        "model_forward_use_cache": model.config.use_cache,
        "note": "KV caching is inherited for generate(); scoring/optimization explicitly pass use_cache=False. top_k=20 and repetition_penalty=1.1 are checkpoint defaults retained by the released helper, not new choices. Log-probability statistics use raw model logits as in the release."}
