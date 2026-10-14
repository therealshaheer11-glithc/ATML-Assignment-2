# Task 5 - Block 8: approved isolated A100 setup and synthetic checks.
from google.colab import drive, files
from pathlib import Path, PurePosixPath
import hashlib, io, json, os, shutil, stat, subprocess, sys, zipfile

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

smi = shutil.which('nvidia-smi')
gpu = subprocess.run([smi, '--query-gpu=name', '--format=csv,noheader'],
                     capture_output=True, text=True, timeout=15) if smi else None
require(gpu and gpu.returncode == 0 and 'A100' in gpu.stdout.upper(),
        'Switch Colab to an NVIDIA A100 runtime before running this block.')

if not Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')
project = Path('/content/drive/MyDrive/ATML-Assignment-2')
root = project / 'task5/experiment_v1'
require(root.is_dir() and not root.is_symlink(), 'Existing Task 5 Drive folder unavailable.')

print('Select the downloaded ATML_PA2_Task5_A100_Setup_v1.zip file.')
uploaded = files.upload()
require(len(uploaded) == 1, 'Select exactly the supplied ZIP file.')
payload = next(iter(uploaded.values()))
require(hashlib.sha256(payload).hexdigest() ==
        '0c5474278fa44cb9df26d37a559414e04bd2f444caa4674bf90928f85f90b64f',
        'Uploaded bundle checksum differs; nothing from it has been executed.')

manifest_sha = 'cace5df894e95d9e7d4d7064da06f32edc183500b0df6fb64f7919b36085ff9d'
release = root / 'a100_setup_releases' / manifest_sha
with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
    entries = bundle.infolist()
    names = [entry.filename for entry in entries]
    require(len(names) == len(set(names)) == 10 and
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
print('Bundle verified and saved. Preparing the approved isolated A100 environment...', flush=True)
process = subprocess.Popen([sys.executable, '-s', '-B', str(release / 'prepare_a100.py')],
                           cwd=release, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end='', flush=True)
require(process.wait() == 0,
        'A100 setup stopped; preserve the runtime and send the printed report.')
print('A100 setup completed. Keep this runtime connected and send the printed output.')