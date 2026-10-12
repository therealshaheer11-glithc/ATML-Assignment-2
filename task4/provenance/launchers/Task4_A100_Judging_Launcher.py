from google.colab import files
import hashlib, runpy

uploaded = files.upload()
name = "Task4_A100_Judging.py"
assert set(uploaded) == {name}, "Upload only Task4_A100_Judging.py"
assert hashlib.sha256(uploaded[name]).hexdigest() == "b7a51de411113a6a3412176f03b33a58206115b9a60ecaa83e5b21b3d07405b8", "Script checksum mismatch"
_ = runpy.run_path(name, run_name="__main__")
