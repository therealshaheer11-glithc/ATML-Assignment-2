"""Repair JSON metadata comparison in memory; preserve all sealed experiment files."""
from __future__ import annotations
import functools
import hashlib
import json
from pathlib import Path
import sys

# Exactly the original module included in the sealed 48,316-byte Chunk 3 ZIP.
EXPECTED_EVALUATION_SHA256 = 'ddad6b34bd279026fe486e4e2ecba62dafcd47e5eaa893fc258b6efa5ebe9700'
FIX_ID = 'TASK2_JSON_CONTRACT_V1'


def canonical_contract(value):
    """Use precisely the representation already written by atomic_json.

    JSON object keys are strings: Transformers' {0: 'LABEL_0'} id2label map
    otherwise differs from the saved {'0': 'LABEL_0'} map after every restart.
    No field is removed, and every stored setting/value is still compared.
    """
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def differences(before, after, path='$'):
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            child = path + '.' + key
            if key not in before or key not in after:
                result.append(child + ' [missing key]')
            else:
                result.extend(differences(before[key], after[key], child))
        return result
    if before != after:
        return [path + ': saved=' + repr(before)[:180] + ', current=' + repr(after)[:180]]
    return []


def evaluation_wrapper(original):
    @functools.wraps(original)
    def evaluate(*args, **kwargs):
        args = list(args)
        if len(args) > 7:
            args[7] = canonical_contract(args[7])
            contract = args[7]
        else:
            kwargs['contract'] = canonical_contract(kwargs['contract'])
            contract = kwargs['contract']
        # Helpful diagnostics for any genuinely changed setting. No bypass.
        name = args[6] if len(args) > 6 else kwargs['name']
        local = args[8] if len(args) > 8 else kwargs['output_root']
        saved = args[9] if len(args) > 9 else kwargs['persistent_root']
        for root in (local, saved):
            path = Path(root) / name / 'contract.json'
            if path.exists():
                observed = json.loads(path.read_text())
                mismatch = differences(observed, contract)
                if mismatch:
                    print('REAL EVALUATION CONTRACT DIFFERENCES:', flush=True)
                    for item in mismatch[:20]:
                        print(item, flush=True)
                    raise RuntimeError('Evaluation settings really differ; existing evidence retained. See field differences above.')
        return original(*args, **kwargs)
    return evaluate


def guard_original(repo):
    source = Path(repo) / 'task2_ppo/evaluation_complete.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_EVALUATION_SHA256:
        raise RuntimeError('Resume compatibility fix expects the exact sealed Chunk 3 evaluation module')


def pipeline_entry():
    from task2_ppo import pipeline
    guard_original(Path(pipeline.__file__).resolve().parents[1])
    pipeline.evaluate_condition = evaluation_wrapper(pipeline.evaluate_condition)
    print(FIX_ID + ': compare all evaluation metadata in its saved JSON representation; numerical settings unchanged.', flush=True)
    pipeline.main()


def replacement_command(command):
    command = list(command)
    if '-m' in command:
        index = command.index('-m')
        if command[index + 1] == 'task2_ppo.pipeline':
            command[index + 1:index + 2] = ['scripts.task2_resume_json', '--pipeline-entry']
    return command


def session_entry():
    from scripts import task2_gpu_session
    from task2_ppo.storage import atomic_json
    repo = Path(task2_gpu_session.__file__).resolve().parents[1]
    guard_original(repo)
    ready_path = Path(sys.argv[sys.argv.index('--ready') + 1])
    ready = json.loads(ready_path.read_text())
    evidence = {
        'fix_id': FIX_ID,
        'original_evaluation_sha256': EXPECTED_EVALUATION_SHA256,
        'compatibility_module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cause': "JSON reload converts Transformers id2label integer key 0 to string key '0'.",
        'change': 'JSON-normalize the live contract before strict comparison; no fields removed.',
        'sealed_sources_modified': False, 'saved_records_modified': False,
        'training_or_generation_settings_changed': False,
        'existing_job_contract_retained': True, 'session_minutes': 120,
    }
    target = Path(ready['experiment_root']) / 'compatibility_fixes/json_contract_v1.json'
    if target.exists() and json.loads(target.read_text()) != evidence:
        raise RuntimeError('Existing compatibility-fix record differs; not replaced')
    atomic_json(target, evidence)
    if json.loads(target.read_text()) != evidence:
        raise RuntimeError('Compatibility-fix record did not persist correctly')
    original = task2_gpu_session.run_visible
    def run(command, log, cwd):
        command = replacement_command(command)
        if '--pipeline-entry' in command:
            note = FIX_ID + ': JSON metadata resume fix active; sealed source hashes preserved.\n'
            print(note, end='', flush=True)
            log.write(note)
            log.flush()
        return original(command, log, cwd)
    task2_gpu_session.run_visible = run
    task2_gpu_session.main()  # Original saving, 120-minute budget and shutdown gate.


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--pipeline-entry':
        del sys.argv[1]
        pipeline_entry()
    else:
        session_entry()
