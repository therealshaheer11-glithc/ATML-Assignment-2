from google.colab import files
import hashlib, runpy

uploaded = files.upload()
name = "Task4_A100_Judge_Load_Resume.py"
assert set(uploaded) == {name}, "Upload only Task4_A100_Judge_Load_Resume.py"
assert hashlib.sha256(uploaded[name]).hexdigest() == "255b62f2e701444bcf05f850b1028c60289f218b6aa38323bfc520a435919904", "Script checksum mismatch"
_ = runpy.run_path(name, run_name="__main__")
