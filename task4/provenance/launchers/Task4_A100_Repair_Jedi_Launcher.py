from google.colab import files
import hashlib, runpy

uploaded = files.upload()
name = "Task4_A100_Repair_Jedi.py"
assert set(uploaded) == {name}, "Upload only Task4_A100_Repair_Jedi.py"
assert hashlib.sha256(uploaded[name]).hexdigest() == "6b1813443b7b61fe9be1bc0ee36e6ac4f73dcba660533d15d884aab4580f95af", "Script checksum mismatch"
_ = runpy.run_path(name, run_name="__main__")
