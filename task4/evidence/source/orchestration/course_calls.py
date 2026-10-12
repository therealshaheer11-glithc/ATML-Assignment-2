"""Thin adapters around unchanged course inference helpers."""


def generate_batch(model, tokenizer, rows):
    from common.generation import batch_generate
    messages = [[{'role': 'user', 'content': row['prompt']}] for row in rows]
    return batch_generate(model, tokenizer, messages, max_prompt_length=256,
                          max_new_tokens=256, temperature=0.0, top_p=1.0, do_sample=False)


class CaptureDecode:
    """Record the original decoded judge text without changing tokenization."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.decoded = []

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def decode(self, *args, **kwargs):
        text = self.tokenizer.decode(*args, **kwargs)
        self.decoded.append(text)
        return text


def judge_response(tokenizer, model, prompt, response):
    from task4_safety.judge_responses import judge_one
    from core import require
    capture = CaptureDecode(tokenizer)
    parsed = judge_one(capture, model, prompt, response, max_new_tokens=64)
    require(len(capture.decoded) == 1, 'Course judge decode coverage differs')
    return {'parsed': parsed, 'raw_output': capture.decoded[0]}
