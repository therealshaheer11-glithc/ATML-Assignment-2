"""Verify entered Drive state and run synthetic orchestration checks on CPU."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
EXPERIMENT = Path('/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1')
STAGE1 = '9a121f29750b5b09c5feafeb491630ba4b88b6c91dc2ed5ee94ebdb989793b65'
MODEL_INPUTS = '50696b4becd13a8f1594896d384e718fa8d3984e4fb5785bec977a6b35eeac7d'
MODEL_INPUT_ATTEMPT = '20261011T121238.845030Z'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise RuntimeError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    stage1 = EXPERIMENT / 'preflight_releases' / STAGE1
    require(digest(stage1 / 'MANIFEST.json') == STAGE1, 'Preflight source identity differs')
    manifest = read_json(stage1 / 'MANIFEST.json')
    require(digest(stage1 / 'scripts/preflight.py') == manifest['scripts/preflight.py'], 'Preflight helper differs')
    preflight = load_module('task4_orchestration_preflight', stage1 / 'scripts/preflight.py')
    preflight.verify_source(stage1)
    source = preflight.verify_source(ROOT)
    preflight.require_cpu()
    require(read_json(EXPERIMENT / 'OWNER.json')['approval_sha256'] == digest(stage1 / 'docs/approval.json'),
            'Task 4 ownership/approval identity differs')
    save, save_bytes = preflight.immutable_json, preflight.immutable_bytes
    release = EXPERIMENT / 'orchestration_releases' / source['manifest_sha256']
    for name in [*read_json(ROOT / 'MANIFEST.json'), 'MANIFEST.json']:
        save_bytes(release / name, (ROOT / name).read_bytes())
    attempt_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    attempt = EXPERIMENT / 'orchestration_attempts' / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    report = {'format': 'ATML_TASK4_CPU_ORCHESTRATION_ATTEMPT_V1', 'status': 'RUNNING',
        'attempt_utc': attempt_id, 'source_manifest_sha256': source['manifest_sha256'],
        'pretrained_model_weights_loaded': False, 'synthetic_test_doubles_only': True,
        'task4_generation_performed': False, 'task4_judging_performed': False,
        'training_performed': False, 'package_changes_performed': False, 'model_downloads_performed': False}
    try:
        save(attempt / 'START.json', report)
        inputs_source = EXPERIMENT / 'model_inputs_releases' / MODEL_INPUTS
        require(digest(inputs_source / 'MANIFEST.json') == MODEL_INPUTS, 'Model-input source identity differs')
        preflight.verify_source(inputs_source)
        preparation = load_module('task4_orchestration_model_inputs', inputs_source / 'prepare.py')
        folder = EXPERIMENT / 'model_inputs_attempts' / MODEL_INPUT_ATTEMPT
        previous = read_json(folder / 'REPORT.json')
        require(previous['status'] == 'CPU_ADAPTERS_TOKENIZERS_READY' and
                previous['source_manifest_sha256'] == MODEL_INPUTS and previous['prepared_adapters'] == 3 and
                previous['tokenizers_checked'] == 2 and previous['prompts_tokenized'] == 450 and
                previous['generation_batches_checked'] == 113 and previous['prompts_exceeding_fixed_cap'] == 0,
                'Completed model-input result differs')
        for name, expected in previous['artifact_sha256'].items():
            require(Path(name).name == name and name not in {'.', '..'}, 'Unsafe preparation artifact path')
            require(digest(folder / name) == expected, 'Model-input artifact changed: ' + name)
        contract_path = EXPERIMENT / 'CPU_MODEL_INPUT_CONTRACT.json'
        require(digest(contract_path) == previous['cpu_model_input_contract_sha256'], 'Model-input contract changed')
        model_inputs = read_json(contract_path)
        require(model_inputs['source_manifest_sha256'] == MODEL_INPUTS and model_inputs['sft_adapter'] is None and
                model_inputs['policy_order'] == ['sft', 'dpo', 'ppo', 'grpo'], 'Frozen policy selection differs')
        for name, key in [('CPU_ENVIRONMENT_CONTRACT.json', 'environment_contract_sha256'),
                          ('INPUT_IDENTITY_CONTRACT.json', 'input_contract_sha256'),
                          ('AUDIT_ID_CONTRACT.json', 'audit_id_contract_sha256'),
                          ('BASE_MODEL_REFERENCE.json', 'base_model_reference_sha256'),
                          ('JUDGE_SNAPSHOT_CONTRACT.json', 'judge_snapshot_contract_sha256')]:
            require(digest(EXPERIMENT / name) == model_inputs[key], 'Frozen preparation contract changed: ' + name)
        env = read_json(EXPERIMENT / 'CPU_ENVIRONMENT_CONTRACT.json')
        require(env['format'] == 'ATML_TASK4_CPU_ENVIRONMENT_V2' and env['required_dependency_chain_passed'] is True
                and env['required_distributions'] == 64 and env['global_dependency_check_passed'] is False,
                'Approved CPU environment scope differs')
        require(platform.python_version() == env['python'] and len(env['all_required_package_versions']) == 64,
                'Python/required package inventory changed')
        for name, version in env['all_required_package_versions'].items():
            require(importlib.metadata.version(name) == version, 'Required package changed: ' + name)
        for policy, adapter in model_inputs['prepared_adapters'].items():
            require(policy in ['dpo', 'ppo', 'grpo'], 'Unexpected prepared policy')
            path = Path(adapter['directory'])
            require(path == EXPERIMENT / 'prepared_adapters' / policy, 'Prepared adapter path differs')
            for name, expected in adapter['files_sha256'].items():
                require(name in preparation.ADAPTER_FILES and digest(path / name) == expected,
                        'Prepared adapter differs: ' + policy + '/' + name)
        require(set(model_inputs['prepared_adapters']) == {'dpo', 'ppo', 'grpo'}, 'Adapter coverage differs')
        course = stage1 / 'course_reference'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['TASK4_COURSE_REFERENCE'] = str(course)
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(course))
        import torch
        require(not torch.cuda.is_available(), 'CPU semantic checks require the CPU runtime')
        import core
        import run_checks
        print('Running 37 synthetic CPU semantic checks; no pretrained model is loaded.', flush=True)
        checks, transcript = run_checks.run()
        save_bytes(attempt / 'CPU_CHECKS.txt', transcript.encode())
        save(attempt / 'CPU_CHECKS.json', checks)
        print(transcript, end='', flush=True)
        require(checks['passed'], 'CPU orchestration/course checks failed; evidence is preserved')
        inputs = read_json(EXPERIMENT / 'INPUT_IDENTITY_CONTRACT.json')['inputs']['xstest']
        csv = preflight.verify_csv(Path(inputs['path']), inputs['sha256'])
        require(csv['ordered_prompt_ids'] == inputs['ordered_prompt_ids'] and
                csv['class_counts'] == {'SAFE': 250, 'UNSAFE': 200}, 'Fixed prompt identity/order differs')
        import pandas as pd
        rows = pd.read_csv(csv['path']).to_dict('records')
        rows = [{key: int(row[key]) if key == 'xstest_id' else str(row[key]) for key in core.META} for row in rows]
        require(len(core.chunks(rows)) == 113, 'Fixed batch coverage differs')
        cfg = model_inputs['course_configuration']
        require(cfg['seed'] == core.SETTINGS['seed'] and cfg['safety_max_new_tokens'] == 256 and
                cfg['judge_max_new_tokens'] == 64 and cfg['dtype'] == 'float16' and
                cfg['quantize_frozen_models'] is True, 'Course settings differ from the tested execution core')
        frozen = core.make_contract(rows, source['manifest_sha256'],
            {'cpu_model_input_contract_sha256': digest(contract_path),
             'audit_id_contract_sha256': digest(EXPERIMENT / 'AUDIT_ID_CONTRACT.json'),
             'gpu_runtime_validation_pending': True},
            model_inputs['tokenizer_and_generation_metadata']['policy']['tokenizer']['eos_token_id'])
        audit = read_json(EXPERIMENT / 'AUDIT_ID_CONTRACT.json')
        require(len(audit['ordered_audit_prompt_ids']) == 60 and audit['manual_labels_assigned'] == 0,
                'Fixed audit contract differs')
        validation = {'format': 'ATML_TASK4_CPU_ORCHESTRATION_VALIDATION_V1',
            'source_manifest_sha256': source['manifest_sha256'],
            'model_input_contract_sha256': digest(contract_path),
            'fixed_evaluation_protocol': frozen, 'tests_passed': checks['tests_run'],
            'tested_scripts_sha256': checks['tested_scripts_sha256'],
            'course_helper_sha256': {name: digest(course / name) for name in
                ['common/models.py', 'common/generation.py', 'task4_safety/judge_responses.py',
                 'task4_safety/make_audit_sheet.py']},
            'synthetic_cpu_validation_passed': True, 'pretrained_model_weights_loaded': False,
            'gpu_inference_validation_pending': True, 'manual_audit_export_pending': True,
            'planned_units': {'prompts_per_policy': 450, 'policies': 4, 'generation_batches_per_policy': 113,
                              'responses': 1800, 'judgments': 1800, 'audit_prompt_ids': 60, 'human_labels': 240}}
        save(EXPERIMENT / 'CPU_ORCHESTRATION_VALIDATION_CONTRACT.json', validation)
        report.update({'status': 'CPU_ORCHESTRATION_VALIDATED_NO_INFERENCE', 'cpu_semantic_tests_passed': checks['tests_run'],
            'cpu_orchestration_validation_contract_sha256': digest(EXPERIMENT / 'CPU_ORCHESTRATION_VALIDATION_CONTRACT.json'),
            'model_input_contract_sha256': digest(contract_path), 'gpu_inference_validation_pending': True})
        print('CPU semantic checks: 37/37 PASS. Fixed prompt IDs/order: 450/450; batches: 113.', flush=True)
        print('Atomic generation-batch and judgment caching, identity checks, and resume behavior validated synthetically.', flush=True)
        print('Class denominators, ambiguous retention, population SD and five-label manual agreement validated synthetically.', flush=True)
        print('Remaining: generation 1800; judging 1800; human labels 240.', flush=True)
        print('GPU environment/model validation remains pending. Stay on CPU until the next block.', flush=True)
    except BaseException as error:
        report['status'] = 'CPU_ORCHESTRATION_FAILED_PRESERVED'
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        report['artifact_sha256'] = {p.name: digest(p) for p in attempt.iterdir() if p.is_file()}
        save(attempt / 'REPORT.json', report)
        print('Saved CPU orchestration report:', attempt / 'REPORT.json', flush=True)
        print('Status:', report['status'], flush=True)


if __name__ == '__main__':
    main()
