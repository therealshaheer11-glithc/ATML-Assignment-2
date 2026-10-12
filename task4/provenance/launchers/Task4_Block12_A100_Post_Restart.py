# Task 4 — Block 12: run AFTER ordinary Restart session/kernel.
from google.colab import drive, files
from pathlib import Path, PurePosixPath
import hashlib, io, json, os, platform, stat, subprocess, sys, tempfile, zipfile

if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive")
uploaded = files.upload()
assert len(uploaded) == 1, "Upload only ATML_PA2_Task4_A100_Post_Restart.zip."
payload = next(iter(uploaded.values()))
assert hashlib.sha256(payload).hexdigest() == "0ee9cff9f74a7f3b2b6fbb82c6c9d364d0540f6014d101a8a28fda9df765a9cf", "ZIP checksum differs."
folder = Path(tempfile.mkdtemp(prefix="task4_post_restart_", dir="/content"))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    names = archive.namelist()
    assert len(names) == len(set(names)), "Duplicate ZIP members."
    for item in archive.infolist():
        relative = PurePosixPath(item.filename)
        assert len(relative.parts) == 1 and item.filename not in {".", ".."}
        assert not relative.is_absolute() and ".." not in relative.parts
        assert "\\" not in item.filename and not item.is_dir()
        assert not stat.S_ISLNK(item.external_attr >> 16)
    manifest = json.loads(archive.read("MANIFEST.json"))
    assert set(names) == set(manifest) | {"MANIFEST.json"}
    for name, expected in manifest.items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
    archive.extractall(folder)

import IPython, jedi
completion_names = {item.name for item in
    jedi.Script("task4_text = 'verified'\ntask4_text.up").complete(2, len("task4_text.up"))}
observed_modules = {}
for module_name, package_name in [
    ("IPython", "ipython"), ("jedi", "jedi"), ("parso", "parso"),
    ("numpy", "numpy"), ("packaging", "packaging"), ("gradio", "gradio"),
    ("diffusers", "diffusers"), ("transformers", "transformers"),
    ("tokenizers", "tokenizers"), ("peft", "peft"),
    ("huggingface_hub", "huggingface-hub"), ("trl", "trl"),
    ("bitsandbytes", "bitsandbytes")]:
    module = sys.modules.get(module_name)
    observed_modules[package_name] = None if module is None else {
        "version": str(getattr(module, "__version__", "unreported")),
        "file": str(getattr(module, "__file__", "unreported"))}
observation = folder / "NOTEBOOK_KERNEL_OBSERVATION.json"
observation.write_text(json.dumps({
    "pid": os.getpid(), "python_executable": sys.executable,
    "python": platform.python_version(), "ipython_version": IPython.__version__,
    "jedi_version": jedi.__version__, "jedi_completion_passed": "upper" in completion_names,
    "already_loaded_modules": observed_modules
}, indent=2) + "\n")
command = [sys.executable, "-B", "-u", str(folder / "recheck_runtime.py"), str(observation)]
process = subprocess.Popen(command, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end="", flush=True)
returncode = process.wait()
if returncode:
    raise subprocess.CalledProcessError(returncode, command)
