from google.colab import files
import hashlib, runpy

uploaded = files.upload()
name = "Task4_CPU_Summaries.py"
assert set(uploaded) == {name}, "Upload only Task4_CPU_Summaries.py"
assert hashlib.sha256(uploaded[name]).hexdigest() == "30e0841b49339db01ca8e2bd45fd37e84a70185c8131f1bb5b973d3b3076941b", "Script checksum mismatch"
_ = runpy.run_path(name, run_name="__main__")
