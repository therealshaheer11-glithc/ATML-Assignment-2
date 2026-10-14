import hashlib

expected = "4eb2177ca31cd41a7e037ada831e02bb0f8f8a23dc66ce27f7b8d04af87c08b1"
audit = globals().get("audit")

if (
    audit is None
    or getattr(audit, "__name__", "") != "verified_task5_cpu_judge_inspection"
    or not callable(getattr(audit, "main", None))
):
    raise RuntimeError("Loaded inspector is missing. Rerun the original upload cell.")

source = getattr(audit, "SOURCE_BYTES", None)
if not isinstance(source, bytes) or hashlib.sha256(source).hexdigest() != expected:
    raise RuntimeError("Inspector verification failed. Stop and send this output.")

setattr(audit, "WRAPPER_SOURCE_SHA256", expected)
audit.main()