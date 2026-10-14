# Task 5 - Block 7: all remaining CPU checks in one pass.
from google.colab import drive, files
from pathlib import Path, PurePosixPath
import hashlib, io, json, os, shutil, stat, subprocess, sys, zipfile

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

smi = shutil.which('nvidia-smi')
gpu = subprocess.run([smi, '--query-gpu=name', '--format=csv,noheader'],
                     capture_output=True, text=True, timeout=15) if smi else None
require(not ((gpu and gpu.returncode == 0 and gpu.stdout.strip()) or
             os.environ.get('COLAB_TPU_ADDR') or os.environ.get('TPU_NAME')),
        'Use CPU / None for this block.')

if not Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')
project = Path('/content/drive/MyDrive/ATML-Assignment-2')
root = project / 'task5/experiment_v1'
require(root.is_dir() and not root.is_symlink(), 'Existing Task 5 Drive folder unavailable.')

print('Select the downloaded ATML_PA2_Task5_CPU_Ready_v1.zip file.')
uploaded = files.upload()
require(len(uploaded) == 1, 'Select exactly the supplied ZIP file.')
payload = next(iter(uploaded.values()))
require(hashlib.sha256(payload).hexdigest() ==
        '40170f6def4b08d4c4ad0cc8c1e5201b3306acc32a5d83e7a5a76936e24598ca',
        'Uploaded bundle checksum differs; nothing from it has been executed.')

manifest_sha = 'd6dd999e7fdf241c7136ee13f46babdf13338fa6dc12ff9046d2ea6a2e89fd87'
release = root / 'evaluation_releases' / manifest_sha
with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
    entries = bundle.infolist()
    names = [entry.filename for entry in entries]
    require(len(names) == len(set(names)) == 31 and
            sum(entry.file_size for entry in entries) < 2 * 2**20,
            'Unexpected bundle coverage or size.')
    for entry in entries:
        p = PurePosixPath(entry.filename)
        require(not p.is_absolute() and '..' not in p.parts and '\\' not in entry.filename
                and not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16),
                'Unsafe archive entry.')
    manifest_bytes = bundle.read('MANIFEST.json')
    require(hashlib.sha256(manifest_bytes).hexdigest() == manifest_sha,
            'Source manifest identity differs.')
    manifest = json.loads(manifest_bytes)
    require(set(names) == set(manifest) | {'MANIFEST.json'}, 'Unexpected packaged files.')
    require(not release.is_symlink(), 'Linked release directory.')
    release.mkdir(parents=True, exist_ok=True)
    existing = {p.relative_to(release).as_posix() for p in release.rglob('*') if p.is_file()}
    require(existing <= set(names), 'Unknown existing release files; preserve and review.')
    for name in names:
        content = bundle.read(name)
        expected = manifest_sha if name == 'MANIFEST.json' else manifest[name]
        require(hashlib.sha256(content).hexdigest() == expected, 'Source checksum differs: ' + name)
        target = release / name
        require(not target.is_symlink(), 'Linked source file.')
        if target.exists():
            require(target.read_bytes() == content, 'Existing source differs; preserve and review.')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(content)
        require(target.read_bytes() == content, 'Saved source readback failed.')

env = {k: v for k, v in os.environ.items()
       if not k.startswith(('PYTHON', 'PIP_')) and k != 'VIRTUAL_ENV'}
env.update({'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1'})
print('Bundle verified and saved. Running the combined CPU checks...', flush=True)
process = subprocess.Popen([sys.executable, '-s', '-B', '-m', 'task5_run.preflight'],
                           cwd=release, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end='', flush=True)
require(process.wait() == 0, 'Combined checks stopped; send the printed report for review.')
print('Combined CPU preflight completed. Send the printed output before continuing.')
