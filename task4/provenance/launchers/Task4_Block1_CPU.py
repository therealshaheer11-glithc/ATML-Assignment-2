# Task 4 — Block 1. Colab CPU / hardware accelerator None.
from google.colab import drive, files
from pathlib import Path, PurePosixPath
import hashlib, io, json, stat, subprocess, sys, tempfile, zipfile

expected_sha256 = "3b135537927a0e562a8d5e149a603af021e4231b8a90e5e9e550d82afce694d8"
print("Upload ATML_PA2_Task4_CPU_Preflight.zip")
uploaded = files.upload()
assert len(uploaded) == 1, "Upload exactly the Task 4 CPU preflight ZIP."
payload = next(iter(uploaded.values()))
assert hashlib.sha256(payload).hexdigest() == expected_sha256, "ZIP checksum differs."

runtime_folder = Path(tempfile.mkdtemp(prefix="task4_cpu_preflight_", dir="/content"))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    names = archive.namelist()
    assert len(names) == len(set(names)), "Duplicate ZIP members."
    for item in archive.infolist():
        relative = PurePosixPath(item.filename)
        assert not relative.is_absolute() and ".." not in relative.parts and "\\" not in item.filename
        assert not stat.S_ISLNK(item.external_attr >> 16), "ZIP contains a symlink."
    manifest = json.loads(archive.read("MANIFEST.json"))
    assert set(names) == set(manifest) | {"MANIFEST.json"}, "ZIP file set differs."
    for name, digest in manifest.items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    archive.extractall(runtime_folder)

drive.mount("/content/drive")
subprocess.run(
    [sys.executable, "-u", "scripts/preflight.py", "--drive-root", "/content/drive/MyDrive"],
    cwd=runtime_folder, check=True,
)
