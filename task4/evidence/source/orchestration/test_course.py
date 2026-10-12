"""Run on Colab CPU with real course imports and synthetic tensor test doubles."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

COURSE = Path(os.environ['TASK4_COURSE_REFERENCE'])
sys.path.insert(0, str(COURSE))
import torch
import common.generation as generation
import common.models as models
import task4_safety.judge_responses as judge
import core
import course_calls


class ToyTokenizer:
    pad_token_id, eos_token_id = 0, 99
    pad_token, eos_token, padding_side = '<pad>', '<eos>', 'left'

    def __init__(self, judge_text=None):
        self.judge_text = judge_text
        self.templates, self.encodings = [], []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_tensors=None):
        self.templates.append({'messages': messages, 'tokenize': tokenize,
                               'add_generation_prompt': add_generation_prompt, 'return_tensors': return_tensors})
        if return_tensors == 'pt':
            return torch.tensor([[11, 12, 13]], dtype=torch.long)
        return 'synthetic rendered: ' + messages[0]['content']

    def __call__(self, texts, **kwargs):
        self.encodings.append(kwargs)
        ids = [[0, 11, 12] if i % 2 == 0 else [11, 12, 13] for i in range(len(texts))]
        return {'input_ids': torch.tensor(ids, dtype=torch.long),
                'attention_mask': torch.tensor([[int(value != 0) for value in row] for row in ids], dtype=torch.long)}

    def decode(self, values, skip_special_tokens=True):
        if self.judge_text is not None:
            return self.judge_text
        return 'synthetic ' + ' '.join(str(value) for value in values.tolist() if value not in {0, 99})


class ToyGenerator(torch.nn.Module):
    def __init__(self, suffix):
        super().__init__()
        self.register_parameter('fixture', torch.nn.Parameter(torch.zeros(1), requires_grad=False))
        self.suffix = torch.tensor(suffix, dtype=torch.long)
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append({'args': args, 'kwargs': kwargs, 'training': self.training,
                           'inference_mode': torch.is_inference_mode_enabled(), 'grad_enabled': torch.is_grad_enabled()})
        ids = args[0] if args else kwargs['input_ids']
        return torch.cat([ids, self.suffix], dim=1)


class CourseInferenceSemantics(unittest.TestCase):
    def test_first_eos_included_and_later_padding_excluded(self):
        ids = torch.tensor([[7, 99, 0, 0], [99, 0, 0, 0], [7, 8, 9, 10]])
        self.assertEqual(generation._response_mask(ids, 99).tolist(), [[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]])

    def test_missing_eos_id_preserves_all_generated_tokens(self):
        ids = torch.tensor([[7, 99, 0]])
        self.assertEqual(generation._response_mask(ids, None).tolist(), [[1, 1, 1]])

    def test_original_batch_helper_lengths_caps_and_training_state(self):
        tok = ToyTokenizer()
        model = ToyGenerator([[7, 99, 0, 0], [7, 8, 99, 0], [7, 8, 9, 10], [99, 0, 0, 0]])
        value = generation.batch_generate(model, tok,
            [[{'role': 'user', 'content': 'synthetic'}]] * 4, 256, 4, do_sample=False)
        self.assertEqual(value['response_lengths'], [2, 3, 4, 1])
        self.assertEqual(value['terminated_with_eos'], [True, True, False, True])
        self.assertEqual(value['truncated'], [False, False, True, False])
        self.assertTrue(model.training)
        self.assertFalse(model.calls[0]['training'])
        self.assertTrue(model.calls[0]['inference_mode'])

    def test_production_wrapper_keeps_fixed_policy_call_and_prompt_form(self):
        rows = [{'xstest_id': i, 'prompt': 'synthetic ' + str(i), 'benchmark_class': 'SAFE', 'type': 'fixture'} for i in range(4)]
        tok, model = ToyTokenizer(), ToyGenerator([[7, 99, 0, 0]] * 4)
        value = course_calls.generate_batch(model, tok, rows)
        records = core.portable_generation('sft', rows, value, 99)
        call = model.calls[0]['kwargs']
        self.assertEqual(call['max_new_tokens'], 256)
        self.assertFalse(call['do_sample'])
        self.assertNotIn('temperature', call)
        self.assertNotIn('top_p', call)
        self.assertEqual((call['pad_token_id'], call['eos_token_id']), (0, 99))
        self.assertEqual(tok.encodings[0], {'return_tensors': 'pt', 'padding': True, 'truncation': True, 'max_length': 256})
        self.assertEqual(tok.templates[0]['messages'], [{'role': 'user', 'content': rows[0]['prompt']}])
        self.assertEqual([row['response_tokens'] for row in records], [2] * 4)

    def test_production_judge_wrapper_keeps_original_prompt_parser_and_raw_output(self):
        self.assertEqual(set(core.LABELS), judge.LABELS)
        raw = '{"label":"over_refusal","confidence":0.01,"rationale_tag":"synthetic"}'
        tok, model = ToyTokenizer(raw), ToyGenerator([[7, 8, 99]])
        model.eval()
        value = course_calls.judge_response(tok, model, 'synthetic prompt', 'synthetic response')
        self.assertEqual(value['raw_output'], raw)
        self.assertEqual(value['parsed'], judge.parse_json(raw))
        self.assertEqual(tok.templates[0]['messages'], [{'role': 'user', 'content':
            judge.JUDGE_PROMPT.format(prompt='synthetic prompt', response='synthetic response')}])
        call = model.calls[0]['kwargs']
        self.assertEqual(call, {'max_new_tokens': 64, 'do_sample': False, 'pad_token_id': 99, 'eos_token_id': 99})
        self.assertFalse(model.calls[0]['grad_enabled'])
        self.assertNotIn('benchmark_class', tok.templates[0]['messages'][0]['content'])

    def test_course_tokenizer_fallback_preserves_left_padding_and_fast_mode(self):
        tok = ToyTokenizer()
        tok.pad_token = None
        with patch.object(models.AutoTokenizer, 'from_pretrained', return_value=tok) as load:
            actual = models.load_tokenizer('/synthetic/snapshot')
        self.assertEqual(actual.pad_token, actual.eos_token)
        self.assertEqual(actual.padding_side, 'left')
        load.assert_called_once_with('/synthetic/snapshot', padding_side='left', use_fast=True)

    def test_policy_loader_uses_fresh_base_each_time_and_sft_has_no_adapter(self):
        created = []
        def base_loader(*args, **kwargs):
            model = ToyGenerator([[99]])
            model.config = type('Config', (), {})()
            created.append(model)
            return model
        cfg = {'base_model': '/synthetic/base', 'dtype': 'float16'}
        with patch.object(models.AutoModelForCausalLM, 'from_pretrained', side_effect=base_loader) as load, \
             patch.object(models, 'load_tokenizer', return_value=ToyTokenizer()), \
             patch.object(models.PeftModel, 'from_pretrained', side_effect=lambda model, *_a, **_k: model) as adapter, \
             patch.object(models, 'get_peft_model', side_effect=AssertionError('Fresh trainable LoRA was requested')), \
             patch.object(torch.cuda, 'is_available', return_value=False):
            sft = models.load_policy(cfg, adapter_path=None, trainable=False, fresh_lora=False)
            self.assertEqual(adapter.call_count, 0)
            for policy in ['dpo', 'ppo', 'grpo']:
                models.load_policy(cfg, adapter_path='/synthetic/' + policy, trainable=False, fresh_lora=False)
        self.assertEqual(len({id(model) for model in created}), 4)
        self.assertFalse(sft.training)
        self.assertEqual(adapter.call_count, 3)
        self.assertTrue(all(call.kwargs == {'is_trainable': False} for call in adapter.call_args_list))
        self.assertTrue(all(call.kwargs == {'dtype': torch.float16, 'low_cpu_mem_usage': True} for call in load.call_args_list))

    def test_gpu_judge_loader_configuration_is_original_nf4_float16_compute(self):
        tok, model = ToyTokenizer(), ToyGenerator([[99]])
        with patch.object(judge.AutoTokenizer, 'from_pretrained', return_value=tok), \
             patch.object(judge.AutoModelForCausalLM, 'from_pretrained', return_value=model) as load, \
             patch.object(torch.cuda, 'is_available', return_value=True):
            judge.load_judge({'ai_judge_model': '/synthetic/judge', 'dtype': 'float16', 'quantize_frozen_models': True})
        config = load.call_args.kwargs['quantization_config']
        self.assertTrue(config.load_in_4bit)
        self.assertEqual(config.bnb_4bit_quant_type, 'nf4')
        self.assertEqual(config.bnb_4bit_compute_dtype, torch.float16)
        self.assertEqual(load.call_args.kwargs['device_map'], 'auto')
        self.assertFalse(model.training)

    def test_cpu_judge_branch_configuration_without_loading_weights(self):
        with patch.object(judge.AutoTokenizer, 'from_pretrained', return_value=ToyTokenizer()), \
             patch.object(judge.AutoModelForCausalLM, 'from_pretrained', return_value=ToyGenerator([[99]])) as load, \
             patch.object(torch.cuda, 'is_available', return_value=False):
            judge.load_judge({'ai_judge_model': '/synthetic/judge', 'dtype': 'float16', 'quantize_frozen_models': True})
        self.assertEqual(load.call_args.kwargs, {'low_cpu_mem_usage': True, 'dtype': torch.float16})


if __name__ == '__main__':
    unittest.main(verbosity=2)
