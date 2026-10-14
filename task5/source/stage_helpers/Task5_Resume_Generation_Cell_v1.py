from google.colab import files
import hashlib, types

print('Select Task5_Resume_Generation_v1.py')
uploaded = files.upload()
if len(uploaded) != 1:
    raise RuntimeError('Select exactly the supplied generation file.')
payload = next(iter(uploaded.values()))
expected = '58443d1ee5d98ed9bde26ea495d107c6a203a7550aaf067d8773c63526593533'
if hashlib.sha256(payload).hexdigest() != expected:
    raise RuntimeError('Generation file checksum differs; nothing was executed.')
resume = types.ModuleType('verified_task5_generation_resume')
exec(compile(payload, 'verified_task5_generation_resume_v1.py', 'exec'),
     resume.__dict__)
resume.WRAPPER_SOURCE_SHA256 = expected
resume.main()
