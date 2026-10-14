from google.colab import files
import hashlib, types

print('Select Task5_Judge_Results_AutoRelease_v1.py')
uploaded = files.upload()
if len(uploaded) != 1:
    raise RuntimeError('Select exactly the supplied Task 5 helper.')

payload = next(iter(uploaded.values()))
expected = 'c57b3094100ea785250de015e15e8250d99fec273268ac6a16b7ef1c0850f166'
if hashlib.sha256(payload).hexdigest() != expected:
    raise RuntimeError('Helper checksum differs; nothing was executed.')

job = types.ModuleType('verified_task5_feedback_batch')
exec(compile(payload, 'verified_task5_feedback_batch_v1.py', 'exec'),
     job.__dict__)
job.SOURCE_BYTES = payload
job.WRAPPER_SOURCE_SHA256 = expected
job.main()