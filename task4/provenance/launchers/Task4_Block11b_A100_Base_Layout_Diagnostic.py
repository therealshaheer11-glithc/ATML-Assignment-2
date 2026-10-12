# Task 4 — Block 11b: keep the same A100 runtime connected.
from google.colab import drive, files
from pathlib import Path, PurePosixPath
import hashlib, io, json, stat, subprocess, sys, tempfile, zipfile

if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive")

uploaded = files.upload()
assert len(uploaded) == 1, "Upload only ATML_PA2_Task4_A100_Base_Layout_Diagnostic.zip."
payload = next(iter(uploaded.values()))
assert hashlib.sha256(payload).hexdigest() == "cf19a50266d8ac63cbe581cec4f9b20269231686abb9dc4f0a40e2bbc37b30d8", "ZIP checksum differs."
folder = Path(tempfile.mkdtemp(prefix="task4_a100_layout_diagnosis_", dir="/content"))
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

command = [sys.executable, "-B", "-u", str(folder / "diagnose_layout.py")]
process = subprocess.Popen(command, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end="", flush=True)
returncode = process.wait()
if returncode:
    raise subprocess.CalledProcessError(returncode, command)
