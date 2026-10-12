from google.colab import files
import hashlib, runpy

uploaded = files.upload()
name = "Task4_A100_Restore_Isolated.py"
assert set(uploaded) == {name}, "Upload only Task4_A100_Restore_Isolated.py"
assert hashlib.sha256(uploaded[name]).hexdigest() == "a3025083ed0094652e39e306c7d2bdaf226d025a651e705cf3f8f37b17ac130e", "Script checksum mismatch"
_ = runpy.run_path(name, run_name="__main__")
