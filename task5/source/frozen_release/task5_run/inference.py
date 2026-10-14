"""Thin observers around the unchanged course generation and judge helpers."""
import copy
import json
import random
import time
from .core import checksum, decode_preference, judge_key, require


def capture_rng():
    import numpy as np
    import torch
    n = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]],
            'torch_cpu': torch.get_rng_state().tolist(),
            'torch_cuda': [s.tolist() for s in torch.cuda.get_rng_state_all()]}


def restore_rng(value):
    import numpy as np
    import torch
    def tuples(x):
        return tuple(tuples(v) for v in x) if isinstance(x, list) else x
    random.setstate(tuples(value['python']))
    n = value['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(torch.tensor(value['torch_cpu'], dtype=torch.uint8))
    require(len(value['torch_cuda']) == torch.cuda.device_count(), 'CUDA RNG device count differs.')
    torch.cuda.set_rng_state_all([torch.tensor(s, dtype=torch.uint8) for s in value['torch_cuda']])


def response_length(ids, eos):
    return ids.index(eos) + 1 if eos is not None and eos in ids else len(ids)


class TimedModel:
    def __init__(self, model, context_limit):
        self.model, self.context_limit, self.calls = model, context_limit, []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def generate(self, *args, **kwargs):
        import torch
        ids = args[0] if args else kwargs['input_ids']
        require(ids.shape[1] + kwargs['max_new_tokens'] <= self.context_limit,
                'Full prompt and generation budget exceed model context; no truncation/fallback.')
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = self.model.generate(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.calls.append({'seconds': elapsed, 'prompt_ids': ids[0].tolist(),
                           'generated_ids': out[0, ids.shape[1]:].tolist()})
        return out


class CaptureTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer, self.decoded = tokenizer, []

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def decode(self, *args, **kwargs):
        text = self.tokenizer.decode(*args, **kwargs)
        self.decoded.append(text)
        return text


class MemoryCachePath:
    """The outer atomic journal persists the course cache together with its raw output."""
    @property
    def parent(self):
        return self

    def mkdir(self, **kwargs):
        pass

    def write_text(self, *args, **kwargs):
        pass


def prompt_preview(tokenizer, messages, context_limit):
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer([rendered], return_tensors='pt', padding=True, truncation=False)
    width = int(encoded['input_ids'].shape[1])
    require(width > 0 and width + 512 <= context_limit, 'Unmodified full prompt exceeds context budget.')
    return width, encoded['input_ids'][0].tolist()


def generate_one(model, tokenizer, row):
    from common.data import prompt_messages
    from common.generation import batch_generate
    messages = prompt_messages(row)
    width, preview = prompt_preview(tokenizer, messages, model.context_limit)
    before = len(model.calls)
    batch = batch_generate(model, tokenizer, [messages], max_prompt_length=width,
                           max_new_tokens=512, temperature=.7, top_p=.9, do_sample=True)
    require(len(model.calls) == before + 1, 'Unexpected generation call count.')
    call = model.calls[-1]
    require(call['prompt_ids'] == preview, 'Course helper changed/truncated the full prompt.')
    length = int(batch['response_lengths'][0])
    require(length == response_length(call['generated_ids'], tokenizer.eos_token_id)
            and 0 <= length <= 512, 'Released length mask differs from generated IDs.')
    return {'response': batch['responses'][0], 'response_tokens': length,
            'generated_token_ids': call['generated_ids'][:length],
            'prompt_tokens': width, 'prompt_token_ids_sha256': checksum(preview),
            'terminated_with_eos': bool(batch['terminated_with_eos'][0]),
            'truncated': bool(batch['truncated'][0]), 'generation_seconds': call['seconds']}


def effective_generation(model, tokenizer):
    value = copy.deepcopy(model.generation_config)
    value.update(max_new_tokens=512, do_sample=True, temperature=.7, top_p=.9,
                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return {'checkpoint_generation_defaults': model.generation_config.to_dict(),
            'effective_requested_generation': value.to_dict(),
            'tokenizer_pad_token_id': tokenizer.pad_token_id,
            'tokenizer_eos_token_id': tokenizer.eos_token_id}


def load_judge(local_snapshot, temporary_cache, previous_records):
    from task5_feedback.rlaif import PairwiseAIJudge
    judge = PairwiseAIJudge({'ai_judge_model': str(local_snapshot), 'dtype': 'float16',
                             'quantize_frozen_models': True}, temporary_cache)
    import torch
    from bitsandbytes.nn import Linear4bit
    quantized = [m for m in judge.model.modules() if isinstance(m, Linear4bit)]
    require(quantized and all(m.weight.device.type == 'cuda' and m.weight.quant_type == 'nf4'
                             and m.compute_dtype == torch.float16
                             and m.weight.quant_state is not None for m in quantized),
            'Loaded judge does not have the released CUDA NF4/float16 configuration.')
    require(all(p.device.type == 'cuda' for p in judge.model.parameters()),
            'Judge CPU/disk offload would change the approved runtime.')
    # Load locally, then restore the course's canonical identity used in _key and A/B swap.
    judge.cfg['ai_judge_model'] = 'Qwen/Qwen2.5-3B-Instruct'
    judge.cache_path = MemoryCachePath()
    judge.cache = {}
    captured = {}
    for record in previous_records:
        p = record['payload']
        key = p['judge_key']
        if key in captured:
            require(captured[key]['ai_preference'] == p['ai_preference']
                    and captured[key]['raw_judge_output'] == p['raw_judge_output'],
                    'Committed cache records disagree.')
        captured[key] = p
        judge.cache[key] = p['ai_preference']
    judge.model = TimedModel(judge.model, int(judge.model.config.max_position_embeddings))
    judge.tokenizer = CaptureTokenizer(judge.tokenizer)
    return judge, captured


def judge_one(judge, captured, problem, a, b):
    key = judge_key(problem, a, b)
    require(judge._key(problem, a, b) == key, 'Canonical course judge key differs.')
    cache_hit = key in judge.cache
    if cache_hit:
        require(key in captured, 'Cached preference lacks committed raw-output evidence.')
        result = judge.compare(problem, a, b)
        old = captured[key]
        require(result == old['ai_preference'], 'Course cached preference differs.')
        return {k: old[k] for k in ('judge_key', 'swapped', 'ai_preference', 'raw_judge_output',
                                    'unparseable', 'judge_prompt_token_ids_sha256', 'judge_output_tokens')} | {
                                        'judge_seconds': 0.0, 'cache_hit': True}
    decoded_before, calls_before = len(judge.tokenizer.decoded), len(judge.model.calls)
    result = judge.compare(problem, a, b)
    require(len(judge.tokenizer.decoded) == decoded_before + 1
            and len(judge.model.calls) == calls_before + 1, 'Unexpected fixed judge call/decode count.')
    raw, call = judge.tokenizer.decoded[-1], judge.model.calls[-1]
    swapped = int(key[:8], 16) % 2 == 1
    expected, unparseable = decode_preference(raw, swapped)
    require(result == expected, 'Observed preference differs from unchanged course parser.')
    value = {'judge_key': key, 'swapped': swapped, 'ai_preference': result,
             'raw_judge_output': raw, 'unparseable': unparseable, 'cache_hit': False,
             'judge_prompt_token_ids_sha256': checksum(call['prompt_ids']),
             'judge_output_tokens': response_length(call['generated_ids'], judge.tokenizer.eos_token_id),
             'judge_seconds': call['seconds']}
    captured[key] = value
    return value
