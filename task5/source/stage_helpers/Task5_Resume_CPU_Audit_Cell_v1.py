# Task 5: CPU resume audit only. Select the downloaded audit script.
from google.colab import files
import hashlib, types

print('Select Task5_CPU_Resume_Audit_v1.py')
uploaded = files.upload()
if len(uploaded) != 1:
    raise RuntimeError('Select exactly the supplied audit file.')
payload = next(iter(uploaded.values()))
expected = 'df82c0f5fc1014650b05035c89b83634732d446081edda08a2b2c7b08a198267'
if hashlib.sha256(payload).hexdigest() != expected:
    raise RuntimeError('Audit file checksum differs; nothing from it was executed.')
audit = types.ModuleType('verified_task5_cpu_resume_audit')
exec(compile(payload, 'verified_task5_cpu_resume_audit_v1.py', 'exec'), audit.__dict__)
audit.AUDIT_SOURCE_SHA256 = expected
audit.main()
