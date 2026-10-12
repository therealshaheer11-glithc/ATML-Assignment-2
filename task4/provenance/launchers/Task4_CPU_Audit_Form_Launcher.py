from google.colab import files
import hashlib, runpy

uploaded = files.upload()
name = "Task4_CPU_Audit_Form.py"
assert set(uploaded) == {name}, "Upload only Task4_CPU_Audit_Form.py"
assert hashlib.sha256(uploaded[name]).hexdigest() == "3d3e28e563ea28fe1c3ca4a0ddccc44ae401784604bce3ce9d228d9dcac235fc", "Script checksum mismatch"
_ = runpy.run_path(name, run_name="__main__")
