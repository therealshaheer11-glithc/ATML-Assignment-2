from google.colab import files
import hashlib, runpy

uploaded = files.upload()
name = "Task4_Resume_Check.py"
assert set(uploaded) == {name}, "Upload only Task4_Resume_Check.py"
assert hashlib.sha256(uploaded[name]).hexdigest() == "818224dbe9f5084839db920aceeb0de785db6bb77adecc8fb76039057637706d", "Script checksum mismatch"
runpy.run_path(name, run_name="__main__")
