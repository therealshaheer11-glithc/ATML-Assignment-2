"""Run deterministic CPU checks; write a truthful machine-readable receipt."""
import argparse
import importlib.metadata
import json
import platform
import sys
import time
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from task3_grpo.storage import atomic_json, sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    root = Path(__file__).resolve().parents[1]
    receipt = {'passed':result.wasSuccessful(),'tests_run':result.testsRun,
        'failures':[{'test':str(t),'traceback':s} for t,s in result.failures],
        'errors':[{'test':str(t),'traceback':s} for t,s in result.errors],
        'skipped':[(str(t),s) for t,s in result.skipped],
        'seconds':time.perf_counter()-start,'python':platform.python_version(),'torch':torch.__version__,
        'packages':{k:importlib.metadata.version(k) for k in ('transformers','tokenizers','peft','numpy')},
        'tested_files_sha256':{p.relative_to(root).as_posix():sha256(p) for p in root.rglob('*')
            if p.is_file() and p.suffix in ('.py','.yaml','.txt') and '__pycache__' not in p.parts},
        'scope':'Tiny synthetic CPU models/records; no course policy weights, real reward inference, CUDA or Colab validation.'}
    atomic_json(args.output,receipt)
    sys.exit(0 if receipt['passed'] else 1)


if __name__=='__main__':
    main()
