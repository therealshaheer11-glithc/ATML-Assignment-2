"""Recompute unchanged Task 1 saved evidence from the organized layout."""
import argparse
from pathlib import Path
import subprocess
import sys

TASK = Path(__file__).resolve().parent
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    subprocess.run([sys.executable, '-B', str(TASK/'source/task1_dpo/analyze_results.py'),
                    '--evidence', str(TASK/'results/evidence'),
                    '--output', str(args.output.absolute())], check=True)
if __name__ == '__main__':
    main()
