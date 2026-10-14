"""Fresh isolated imports, tiny CUDA/NF4 checks and tokenizer-only prompt checks."""
import hashlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path
import sys
import unittest

release, project, output = map(Path, sys.argv[1:])
sys.path.insert(0, str(release))
from task5_run import core
from task5_run.run import verify_source, inputs
from task5_run.inference import prompt_preview

verify_source(release)
bindings, rows = inputs(project, release)

import torch, transformers, tokenizers, trl, peft
import datasets, accelerate, bitsandbytes, huggingface_hub
import safetensors, yaml, pandas, numpy, sklearn, matplotlib, tqdm
from common.data import load_yaml, prompt_messages
from common.models import load_tokenizer
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge
from task5_run.diagnostics import group_rows, VARIANTS

core.require(torch.cuda.is_available() and torch.cuda.device_count() == 1
             and 'A100' in torch.cuda.get_device_name(0)
             and torch.cuda.get_device_capability(0) == (8, 0), 'One A100 with capability 8.0 is required.')
matrix = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float16, device='cuda:0')
product = matrix @ torch.eye(2, dtype=torch.float16, device='cuda:0')
core.require(torch.equal(matrix, product), 'Tiny float16 CUDA multiplication failed.')
fixture = torch.linspace(-1, 1, 256, dtype=torch.float16, device='cuda:0').reshape(16, 16)
packed, state = bitsandbytes.functional.quantize_4bit(fixture, quant_type='nf4')
restored = bitsandbytes.functional.dequantize_4bit(packed, quant_state=state)
core.require(restored.shape == fixture.shape and restored.dtype == torch.float16
             and state.quant_type == 'nf4' and bool(torch.isfinite(restored).all()),
             'Tiny NF4 CUDA round-trip check failed.')
torch.cuda.synchronize()
del matrix, product, fixture, packed, state, restored
print('Fresh imports and synthetic float16 CUDA / NF4 checks: PASS.', flush=True)

configuration = load_yaml(str(release / 'configs/feedback.yaml'))
core.require(configuration['seed'] == 6304 and configuration['dtype'] == 'float16'
             and configuration['math_max_new_tokens'] == 512 and configuration['judge_max_new_tokens'] == 64
             and configuration['generation'] == {'temperature': .7, 'top_p': .9, 'do_sample': True},
             'Released configuration differs.')

for role in ('policy', 'judge'):
    folder = project / bindings['snapshots'][role]['path']
    for name, record in bindings['snapshots'][role]['files'].items():
        path = folder / name
        core.require(path.is_file() and not path.is_symlink() and path.stat().st_size == record['bytes'],
                     'Snapshot file availability/size differs: ' + name)
        if not name.endswith('.safetensors'):
            core.require(core.file_sha(path) == record['sha256'], 'Tokenizer/model metadata identity differs.')

policy_folder = project / bindings['snapshots']['policy']['path']
policy_tokenizer = load_tokenizer(str(policy_folder))
policy_context = core.read_json(policy_folder / 'config.json')['max_position_embeddings']
prompt_checks = {}
for domain, domain_rows in rows.items():
    prompt_checks[domain] = []
    for index, row in enumerate(domain_rows):
        width, ids = prompt_preview(policy_tokenizer, prompt_messages(row), policy_context)
        prompt_checks[domain].append({'row_index': index, 'row_sha256': core.checksum(row),
                                     'prompt_tokens': width, 'token_ids_sha256': core.checksum(ids)})

judge_folder = project / bindings['snapshots']['judge']['path']
judge_tokenizer = load_tokenizer(str(judge_folder))
judge_context = core.read_json(judge_folder / 'config.json')['max_position_embeddings']
groups = group_rows(core.read_rows(project / bindings['diagnostics']))
diagnostic_checks = []
for pid, group in groups.items():
    clean = group['clean_correct']
    for variant in VARIANTS[1:]:
        a, b = clean['response'], group[variant]['response']
        key = core.judge_key(clean['question'], a, b)
        swap = int(key[:8], 16) % 2 == 1
        a, b = (b, a) if swap else (a, b)
        text = PAIRWISE_RUBRIC.format(problem=clean['question'], a=a, b=b)
        ids = judge_tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                                  return_tensors='pt', add_generation_prompt=True)
        core.require(ids.shape[1] + 4 <= judge_context, 'Full diagnostic judge prompt exceeds context.')
        diagnostic_checks.append({'problem_id': pid, 'variant': variant, 'judge_key': key,
                                  'prompt_tokens': ids.shape[1], 'token_ids_sha256': core.checksum(ids.tolist())})

core.write_once(output / 'POLICY_PROMPT_CHECKS.json', prompt_checks)
core.write_once(output / 'DIAGNOSTIC_JUDGE_PROMPT_CHECKS.json', diagnostic_checks)
print('Pinned tokenizers: 400 full policy prompts and 80 diagnostic judge prompts fit.', flush=True)

suite = unittest.defaultTestLoader.loadTestsFromNames([
    'task5_run.test_core', 'task5_run.test_inference', 'task5_run.test_orchestration', 'task5_run.test_preflight'])
log = io.StringIO()
result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
(output / 'ISOLATED_TESTS.txt').write_text(log.getvalue())
core.require(result.wasSuccessful() and result.testsRun == 44, 'Isolated 44-test suite did not pass.')
probe = {'fresh_imports_passed': True, 'tiny_float16_cuda_passed': True, 'tiny_nf4_cuda_passed': True,
         'tests_run': result.testsRun, 'tests_passed': True,
         'gpu': torch.cuda.get_device_name(0), 'compute_capability': list(torch.cuda.get_device_capability(0)),
         'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
         'policy_prompts_checked': sum(len(v) for v in prompt_checks.values()),
         'diagnostic_judge_prompts_checked': len(diagnostic_checks),
         'tokenizers_loaded': 2, 'pretrained_model_weights_loaded': 0,
         'policy_answers_generated': 0, 'ai_judgments': 0,
         'policy_prompt_checks_sha256': core.file_sha(output / 'POLICY_PROMPT_CHECKS.json'),
         'diagnostic_prompt_checks_sha256': core.file_sha(output / 'DIAGNOSTIC_JUDGE_PROMPT_CHECKS.json')}
core.write_once(output / 'GPU_PROBE.json', probe)
print('Isolated tests: 44/44 PASS. Pretrained model weights loaded: 0.', flush=True)
