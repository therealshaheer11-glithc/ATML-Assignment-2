# Task 4 — Approved CPU recovery. Use the same runtime.
from google.colab import files
from pathlib import Path, PurePosixPath
import hashlib, io, json, stat, subprocess, sys, tempfile, zipfile

uploaded = files.upload()
assert len(uploaded) == 1, "Upload only ATML_PA2_Task4_CPU_Recovery.zip."
payload = next(iter(uploaded.values()))
assert hashlib.sha256(payload).hexdigest() == "373ed4078f5b1349f46d7b97678bca2736a319bdb61ff10715fb244d4747963d", "ZIP checksum differs."
folder = Path(tempfile.mkdtemp(prefix="task4_cpu_recovery_", dir="/content"))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    names = archive.namelist()
    assert len(names) == len(set(names)), "Duplicate ZIP members."
    for item in archive.infolist():
        relative = PurePosixPath(item.filename)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert not stat.S_ISLNK(item.external_attr >> 16)
    manifest = json.loads(archive.read("MANIFEST.json"))
    assert set(names) == set(manifest) | {"MANIFEST.json"}
    for name, digest in manifest.items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    archive.extractall(folder)

command = [sys.executable, "-u", str(folder / "recover.py")]
process = subprocess.Popen(command, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end="", flush=True)
returncode = process.wait()
if returncode:
    raise subprocess.CalledProcessError(returncode, command)
