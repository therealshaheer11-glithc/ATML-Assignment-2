from google.colab import files
import hashlib, runpy

uploaded = files.upload()
name = "Task4_A100_Post_Restart_Resume.py"
assert set(uploaded) == {name}, "Upload only Task4_A100_Post_Restart_Resume.py"
assert hashlib.sha256(uploaded[name]).hexdigest() == "0a4fbc85656b74e01c145a2476ae058f9687e6bcf3cc21b3815716a9da67115d", "Script checksum mismatch"
_ = runpy.run_path(name, run_name="__main__")
