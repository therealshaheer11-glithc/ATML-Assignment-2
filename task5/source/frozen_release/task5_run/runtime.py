"""Require the approved isolated A100 environment and explicit execution approval."""
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from .core import checksum, file_sha, read_json, require


def execution_gate(project, release, phase):
    root = project / 'task5/experiment_v1'
    approval_path = root / 'EXECUTION_APPROVAL.json'
    require(approval_path.is_file(), 'Pretrained execution approval has not been recorded.')
    approval = read_json(approval_path)
    protocol_sha = file_sha(release / 'protocol.json')
    require(approval.get('approved') is True and approval.get(phase) is True
            and approval.get('protocol_sha256') == protocol_sha,
            'Execution approval does not cover this phase and frozen protocol.')
    require(sys.prefix == '/content/task5_isolated_runtime_v1'
            and sys.prefix != sys.base_prefix, 'Use the approved isolated interpreter.')
    require(os.environ.get('HF_HUB_OFFLINE') == '1'
            and os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'Offline model loading is required.')
    contract_path = root / 'ISOLATED_ENVIRONMENT_CONTRACT.json'
    require(contract_path.is_file(), 'Task 5 A100 environment validation is incomplete.')
    contract = read_json(contract_path)
    cpu_path = root / 'CPU_READINESS_CONTRACT.json'
    require(cpu_path.is_file(), 'Combined CPU readiness checkpoint is missing.')
    cpu = read_json(cpu_path)
    require(contract.get('complete') is True and contract.get('synthetic_cuda_nf4_passed') is True
            and contract.get('package_versions') == read_json(release / 'package_versions.json')
            and contract.get('cpu_readiness_sha256') == file_sha(cpu_path)
            and cpu.get('release_sha256') == file_sha(release / 'MANIFEST.json')
            and cpu.get('complete') is True,
            'Task 5 environment contract differs.')
    pins = read_json(release / 'package_versions.json')
    from packaging.utils import canonicalize_name
    dists = list(importlib.metadata.distributions())
    actual = {canonicalize_name(d.metadata['Name']): d.version for d in dists}
    require(len(actual) == len(dists) and actual == pins, 'Installed packages differ from approved exact set.')
    require(all(Path(d.locate_file('')).resolve().is_relative_to(Path(sys.prefix).resolve()) for d in dists),
            'Package imported from outside the isolated environment.')
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1
            and 'A100' in torch.cuda.get_device_name(0), 'One A100 is required.')
    smi = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True, timeout=15)
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'driver': smi.stdout.strip(), 'packages_sha256': checksum(actual),
            'environment_contract_sha256': file_sha(contract_path),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}
