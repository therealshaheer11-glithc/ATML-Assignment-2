# Task 4 — Block 11: keep the same A100 runtime connected.
from google.colab import drive, files
from pathlib import Path, PurePosixPath
import hashlib, io, json, os, stat, subprocess, sys, tempfile, zipfile

if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive")
uploaded = files.upload()
assert len(uploaded) == 1, "Upload only ATML_PA2_Task4_A100_Base_Repair.zip."
payload = next(iter(uploaded.values()))
assert hashlib.sha256(payload).hexdigest() == "834d27dfcf0ce9d3c1584cba6de0059083bc6c185a8787291994fcc3be91e99b", "ZIP checksum differs."
folder = Path(tempfile.mkdtemp(prefix="task4_a100_base_repair_", dir="/content"))
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

observed_modules = {}
for name in ["IPython", "jedi", "parso", "gradio", "diffusers",
             "transformers", "tokenizers", "peft", "huggingface_hub", "trl", "bitsandbytes"]:
    module = sys.modules.get(name)
    observed_modules[name] = None if module is None else {
        "version": str(getattr(module, "__version__", "unreported")),
        "file": str(getattr(module, "__file__", "unreported"))}
observation = folder / "NOTEBOOK_KERNEL_OBSERVATION.json"
observation.write_text(json.dumps({"pid": os.getpid(), "python_executable": sys.executable,
                                   "already_loaded_modules": observed_modules}, indent=2) + "\n")
command = [sys.executable, "-B", "-u", str(folder / "repair_base.py"), str(observation)]
process = subprocess.Popen(command, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end="", flush=True)
returncode = process.wait()
if returncode:
    raise subprocess.CalledProcessError(returncode, command)
