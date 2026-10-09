from pathlib import Path
from uuid import uuid4
import hashlib
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def safe_extract(z, destination):
    destination = Path(destination).resolve()
    names = set()
    for item in z.infolist():
        target = (destination / item.filename).resolve()
        if destination not in target.parents or item.filename in names or (item.external_attr >> 16) & 0o170000 == 0o120000:
            raise RuntimeError('Unsafe or duplicate archive member')
        names.add(item.filename)
    z.extractall(destination)


def run_visible(command, cwd=None):
    process = subprocess.Popen(list(map(str, command)), cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        code = process.wait()
    except KeyboardInterrupt:
        # The GPU orchestrator stops its child process and saves completed work.
        process.send_signal(signal.SIGINT)
        remaining, _ = process.communicate()
        print(remaining, end='', flush=True)
        code = process.returncode
    except BaseException:
        process.terminate()
        process.wait()
        raise
    if code:
        raise RuntimeError('Process failed; keep the runtime available and inspect the printed error.')


def latest_receipt(pattern, status, override=''):
    choices = [Path(override)] if override else sorted(Path('/content/drive/MyDrive/ATML-Assignment-2/task2').glob(pattern), reverse=True)
    for path in choices:
        record = json.loads(path.read_text())
        if record['status'] == status:
            return path, record
    raise RuntimeError('No completed preparation receipt found. Run the CPU preparation notebook first.')


def restore_project(ready, repo):
    spec = ready['project_archive']
    repo = Path(repo)
    with tempfile.TemporaryDirectory(prefix='task2_bootstrap_') as tmp:
        local = Path(tmp) / 'project.zip'
        shutil.copyfile(spec['path'], local)
        if file_sha(local) != spec['sha256']:
            raise RuntimeError('Saved project archive checksum differs')
        with zipfile.ZipFile(local) as z:
            manifest = json.loads(z.read('TASK2_BACKUP_MANIFEST.json'))['files_sha256']
            expected = set(manifest) | {'TASK2_BACKUP_MANIFEST.json'}
            if set(z.namelist()) != expected or len(z.namelist()) != len(expected):
                raise RuntimeError('Project archive inventory differs')
            for name, digest in manifest.items():
                if hashlib.sha256(z.read(name)).hexdigest() != digest:
                    raise RuntimeError('Project member checksum differs: ' + name)
            extracted = Path(tmp) / 'extracted'
            safe_extract(z, extracted)
        for name, digest in manifest.items():
            target = repo / name
            if target.exists() and file_sha(target) != digest:
                raise RuntimeError('Existing project file differs; use a fresh runtime: ' + name)
        for name in manifest:
            target = repo / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(extracted / name, target)

# Optional exact GPU-ready receipt path. Approved session limit: 120 minutes.
READY_RECEIPT = ''
SESSION_START_EPOCH = time.time()
import torch
from google.colab import drive, runtime
if not torch.cuda.is_available():
    raise RuntimeError('Select Runtime > Change runtime type > A100 GPU first.')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
if 'A100' not in torch.cuda.get_device_name(0):
    raise RuntimeError('This prepared launch expects the previously tested A100. Select A100 before using GPU time.')
drive.mount('/content/drive')
ready_path, ready = latest_receipt('gpu_ready/*/ready.json', 'GPU_JOBS_READY', READY_RECEIPT)
print('Using GPU-ready receipt:', ready_path, flush=True)
repo = Path('/content/ATML-PA2-Task2')
restore_project(ready, repo)
# Metadata compatibility repair; every original source/hash and saved result is retained.
RESUME_MODULE_SOURCE = '"""Repair JSON metadata comparison in memory; preserve all sealed experiment files."""\nfrom __future__ import annotations\nimport functools\nimport hashlib\nimport json\nfrom pathlib import Path\nimport sys\n\n# Exactly the original module included in the sealed 48,316-byte Chunk 3 ZIP.\nEXPECTED_EVALUATION_SHA256 = \'ddad6b34bd279026fe486e4e2ecba62dafcd47e5eaa893fc258b6efa5ebe9700\'\nFIX_ID = \'TASK2_JSON_CONTRACT_V1\'\n\n\ndef canonical_contract(value):\n    """Use precisely the representation already written by atomic_json.\n\n    JSON object keys are strings: Transformers\' {0: \'LABEL_0\'} id2label map\n    otherwise differs from the saved {\'0\': \'LABEL_0\'} map after every restart.\n    No field is removed, and every stored setting/value is still compared.\n    """\n    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))\n\n\ndef differences(before, after, path=\'$\'):\n    if isinstance(before, dict) and isinstance(after, dict):\n        result = []\n        for key in sorted(before.keys() | after.keys()):\n            child = path + \'.\' + key\n            if key not in before or key not in after:\n                result.append(child + \' [missing key]\')\n            else:\n                result.extend(differences(before[key], after[key], child))\n        return result\n    if before != after:\n        return [path + \': saved=\' + repr(before)[:180] + \', current=\' + repr(after)[:180]]\n    return []\n\n\ndef evaluation_wrapper(original):\n    @functools.wraps(original)\n    def evaluate(*args, **kwargs):\n        args = list(args)\n        if len(args) > 7:\n            args[7] = canonical_contract(args[7])\n            contract = args[7]\n        else:\n            kwargs[\'contract\'] = canonical_contract(kwargs[\'contract\'])\n            contract = kwargs[\'contract\']\n        # Helpful diagnostics for any genuinely changed setting. No bypass.\n        name = args[6] if len(args) > 6 else kwargs[\'name\']\n        local = args[8] if len(args) > 8 else kwargs[\'output_root\']\n        saved = args[9] if len(args) > 9 else kwargs[\'persistent_root\']\n        for root in (local, saved):\n            path = Path(root) / name / \'contract.json\'\n            if path.exists():\n                observed = json.loads(path.read_text())\n                mismatch = differences(observed, contract)\n                if mismatch:\n                    print(\'REAL EVALUATION CONTRACT DIFFERENCES:\', flush=True)\n                    for item in mismatch[:20]:\n                        print(item, flush=True)\n                    raise RuntimeError(\'Evaluation settings really differ; existing evidence retained. See field differences above.\')\n        return original(*args, **kwargs)\n    return evaluate\n\n\ndef guard_original(repo):\n    source = Path(repo) / \'task2_ppo/evaluation_complete.py\'\n    digest = hashlib.sha256(source.read_bytes()).hexdigest()\n    if digest != EXPECTED_EVALUATION_SHA256:\n        raise RuntimeError(\'Resume compatibility fix expects the exact sealed Chunk 3 evaluation module\')\n\n\ndef pipeline_entry():\n    from task2_ppo import pipeline\n    guard_original(Path(pipeline.__file__).resolve().parents[1])\n    pipeline.evaluate_condition = evaluation_wrapper(pipeline.evaluate_condition)\n    print(FIX_ID + \': compare all evaluation metadata in its saved JSON representation; numerical settings unchanged.\', flush=True)\n    pipeline.main()\n\n\ndef replacement_command(command):\n    command = list(command)\n    if \'-m\' in command:\n        index = command.index(\'-m\')\n        if command[index + 1] == \'task2_ppo.pipeline\':\n            command[index + 1:index + 2] = [\'scripts.task2_resume_json\', \'--pipeline-entry\']\n    return command\n\n\ndef session_entry():\n    from scripts import task2_gpu_session\n    from task2_ppo.storage import atomic_json\n    repo = Path(task2_gpu_session.__file__).resolve().parents[1]\n    guard_original(repo)\n    ready_path = Path(sys.argv[sys.argv.index(\'--ready\') + 1])\n    ready = json.loads(ready_path.read_text())\n    evidence = {\n        \'fix_id\': FIX_ID,\n        \'original_evaluation_sha256\': EXPECTED_EVALUATION_SHA256,\n        \'compatibility_module_sha256\': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),\n        \'cause\': "JSON reload converts Transformers id2label integer key 0 to string key \'0\'.",\n        \'change\': \'JSON-normalize the live contract before strict comparison; no fields removed.\',\n        \'sealed_sources_modified\': False, \'saved_records_modified\': False,\n        \'training_or_generation_settings_changed\': False,\n        \'existing_job_contract_retained\': True, \'session_minutes\': 120,\n    }\n    target = Path(ready[\'experiment_root\']) / \'compatibility_fixes/json_contract_v1.json\'\n    if target.exists() and json.loads(target.read_text()) != evidence:\n        raise RuntimeError(\'Existing compatibility-fix record differs; not replaced\')\n    atomic_json(target, evidence)\n    if json.loads(target.read_text()) != evidence:\n        raise RuntimeError(\'Compatibility-fix record did not persist correctly\')\n    original = task2_gpu_session.run_visible\n    def run(command, log, cwd):\n        command = replacement_command(command)\n        if \'--pipeline-entry\' in command:\n            note = FIX_ID + \': JSON metadata resume fix active; sealed source hashes preserved.\\n\'\n            print(note, end=\'\', flush=True)\n            log.write(note)\n            log.flush()\n        return original(command, log, cwd)\n    task2_gpu_session.run_visible = run\n    task2_gpu_session.main()  # Original saving, 120-minute budget and shutdown gate.\n\n\nif __name__ == \'__main__\':\n    if len(sys.argv) > 1 and sys.argv[1] == \'--pipeline-entry\':\n        del sys.argv[1]\n        pipeline_entry()\n    else:\n        session_entry()\n'
RESUME_MODULE_SHA256 = 'ffd1841563573eb4c2d3209f40394536512087949a34d567cfcd07e87a0e1b59'
resume_module = repo / 'scripts/task2_resume_json.py'
if hashlib.sha256(RESUME_MODULE_SOURCE.encode()).hexdigest() != RESUME_MODULE_SHA256:
    raise RuntimeError('Embedded compatibility module checksum differs')
if resume_module.exists() and file_sha(resume_module) != RESUME_MODULE_SHA256:
    raise RuntimeError('Existing compatibility module differs; use a fresh runtime')
resume_module.write_text(RESUME_MODULE_SOURCE)
if file_sha(resume_module) != RESUME_MODULE_SHA256:
    raise RuntimeError('Compatibility module installation differs')
print('JSON contract resume repair installed; original experiment sources retained.', flush=True)
finish_path = Path('/content') / ('task2_finish_' + uuid4().hex + '.json')
run_visible([sys.executable, '-u', '-m', 'scripts.task2_resume_json', '--ready', ready_path,
             '--start-epoch', str(SESSION_START_EPOCH), '--finish-receipt', finish_path], cwd=repo)
finished = json.loads(finish_path.read_text())
if not finished['verified']:
    raise RuntimeError('Session backup not verified. Keep this runtime available for recovery.')
# Both process output and the complete session log have now been persisted.
print('SESSION BACKUP VERIFIED. Status:', finished['status'], flush=True)
if finished['status'] == 'SESSION_LIMIT':
    print('After disconnection, resume the same GPU notebook in a fresh A100 runtime.', flush=True)
elif finished['status'] == 'COMPLETE':
    print('GPU work complete. Next run Task2_Results_CPU.ipynb on CPU.', flush=True)
else:
    print('The job failed. The saved session log contains the error; inspect it before retrying.', flush=True)
sys.path.insert(0, str(repo))
from scripts.task2_finish import disconnect_after_verified
try:
    disconnect_after_verified(finished['resource_approval'], finished['verified'],
                              drive.flush_and_unmount, runtime.unassign)
except Exception as error:
    print('Automatic shutdown did not finish:', repr(error), flush=True)
    print('If Drive flush failed, keep this runtime for recovery. If flushing succeeded and only unassign failed, select Runtime > Disconnect and delete runtime.', flush=True)
    raise
