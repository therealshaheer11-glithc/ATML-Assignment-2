from google.colab import files
from pathlib import Path
import hashlib, runpy

uploaded = files.upload()
expected = {
    "Task4_CPU_Audit_Analysis.py": "f911bada1c9562378673b93d28472bf93c4aac3324c6104fa7ab5011d7f8d377",
    "Task4_Completed_Audit_Labels.json": "4d95102c5385b166b291d8579c38986fc4dd9c0112742b8d0874fdc11b1d2d0c",
}
assert set(uploaded) == set(expected), "Upload exactly the analysis script and completed labels JSON"
for name, checksum in expected.items():
    assert hashlib.sha256(uploaded[name]).hexdigest() == checksum, "Upload checksum mismatch: " + name
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == checksum, "Saved upload differs: " + name
_ = runpy.run_path("Task4_CPU_Audit_Analysis.py", run_name="__main__")
