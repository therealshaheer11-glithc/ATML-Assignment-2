from google.colab import files
import hashlib, types

print('Select Task5_Resume_A100_Setup_v1.py')
uploaded = files.upload()
if len(uploaded) != 1:
    raise RuntimeError('Select exactly the supplied setup file.')

payload = next(iter(uploaded.values()))
expected = 'f1d2b770df093e3168751b39554d0315a2ba6b313ca4614c4e19707449d124d2'
if hashlib.sha256(payload).hexdigest() != expected:
    raise RuntimeError('Setup file checksum differs; nothing was executed.')

resume = types.ModuleType('verified_task5_a100_resume_setup')
exec(compile(payload, 'verified_task5_a100_resume_setup_v1.py', 'exec'),
     resume.__dict__)
resume.WRAPPER_SOURCE_SHA256 = expected
resume.main()
