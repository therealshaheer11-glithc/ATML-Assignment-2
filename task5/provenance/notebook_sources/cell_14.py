from google.colab import files
import hashlib, types

print('Select Task5_CPU_Judge_Output_Inspection_v1.py')
uploaded = files.upload()
if len(uploaded) != 1:
    raise RuntimeError('Select exactly the supplied inspection helper.')

payload = next(iter(uploaded.values()))
expected = '4eb2177ca31cd41a7e037ada831e02bb0f8f8a23dc66ce27f7b8d04af87c08b1'
if hashlib.sha256(payload).hexdigest() != expected:
    raise RuntimeError('Inspection helper checksum differs; nothing was executed.')

audit = types.ModuleType('verified_task5_cpu_judge_inspection')
exec(compile(payload, 'verified_task5_cpu_judge_inspection_v1.py', 'exec'),
     audit.__dict__)
audit.SOURCE_BYTES = payload
audit.WRAPPE
R_SOURCE_SHA256 = expected
audit.main()