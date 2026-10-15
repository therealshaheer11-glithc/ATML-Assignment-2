"""Run the unchanged figure script with a temporary historical input layout."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

TASK = Path(__file__).resolve().parent
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.absolute()
    if out.exists():
        p.error('Choose a new output directory.')
    with tempfile.TemporaryDirectory(prefix='task1-figure-inputs-') as tmp:
        root = Path(tmp)
        shutil.copyfile(TASK.parent/'manifests/TASK1_ORIGINAL_SHA256.json', root/'PACKAGE_SHA256.json')
        target = root/'results/task1/analysis'
        target.mkdir(parents=True)
        for name in ('summary.csv','length_strata.csv','word_limits.csv','audit.json'):
            shutil.copyfile(TASK/'results/analysis'/name, target/name)
        subprocess.run([sys.executable,'-B',str(TASK/'source/task1_dpo/plot_results.py'),
                        '--repo',str(root),'--output',str(out)],check=True)
if __name__ == '__main__':
    main()
