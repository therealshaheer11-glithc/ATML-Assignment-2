from google.colab import files
import hashlib, runpy

uploaded = files.upload()
name = "Task4_A100_Resume_Preflight.py"
assert set(uploaded) == {name}, "Upload only Task4_A100_Resume_Preflight.py"
assert hashlib.sha256(uploaded[name]).hexdigest() == "b304191c43f791b5c9b742758389027b7c7c0de034d72ed7881ad25747d5a657", "Script checksum mismatch"
_ = runpy.run_path(name, run_name="__main__")
