"""Dispatch the three fixed beta conditions through the original run commands.

Added after the experiments to replace the unused starter placeholder. Each
condition runs in a separate process; training always initializes from scratch.
The original train/evaluate implementations enforce settings and save backups.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def commands(args):
    result = []
    for name in ("beta_003", "beta_010", "beta_030"):
        root = args.run_root / name
        shared = ["--config", args.config, "--models-dir", str(args.models_dir), "--backup-root", str(args.backup_root)]
        if args.phase == "train":
            cmd = [sys.executable, "-m", "task1_dpo.train", "--run-name", name,
                   "--output", str(root / "checkpoint"), *shared]
        else:
            cmd = [sys.executable, "-m", "task1_dpo.evaluate", "--name", name, "--phase", args.phase,
                   "--output", str(root / args.phase), *shared]
            cmd += (["--adapter", str(root / "checkpoint")] if args.phase == "policy" else
                    ["--policy-results", str(root / "policy")])
        result.append(cmd)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--phase", choices=("train", "policy", "reward"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--plan", action="store_true", help="Print dispatch commands without execution")
    args = parser.parse_args()
    for cmd in commands(args):
        print(shlex.join(cmd), flush=True)
        if not args.plan:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
