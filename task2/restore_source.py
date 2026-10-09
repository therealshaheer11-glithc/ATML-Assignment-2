"""Create a fresh isolated checkout of the exact Task 2 source, without downloads."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess
HERE=Path(__file__).resolve().parent
COMMIT='1d64ac65acd5e45d1e4e1f415edc80455e21274e'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--destination',type=Path,default=HERE/'.runtime/course')
 args=parser.parse_args();dest=args.destination.expanduser().resolve()
 if dest.exists():raise SystemExit('Destination already exists; choose a fresh --destination. Nothing overwritten.')
 manifest=json.loads((HERE/'provenance/SOURCE_SHA256.json').read_text())
 for rel,expected in manifest.items():
  if sha(HERE/'source'/rel)!=expected:raise SystemExit('Source checksum differs: '+rel)
 # Keep the original Git HEAD required by the historical source guard. The task
 # implementation is an applied, explicitly hashed overlay in this disposable checkout.
 subprocess.run(['git','clone','--no-checkout',str(HERE/'provenance/course.bundle'),str(dest)],check=True)
 subprocess.run(['git','-C',str(dest),'checkout','--detach',COMMIT],check=True)
 for rel in manifest:
  target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(HERE/'source'/rel,target)
 approval=json.loads((dest/'docs/task2_approval.json').read_text())
 for rel,expected in approval['source_sha256'].items():
  if sha(dest/rel)!=expected:raise SystemExit('Restored course/helper checksum differs: '+rel)
 for n in (2,3):
  for rel,expected in json.loads((dest/f'docs/task2_chunk{n}_files.json').read_text()).items():
   if sha(dest/rel)!=expected:raise SystemExit('Restored chunk checksum differs: '+rel)
 print('SOURCE RESTORED AND VERIFIED:',dest)
 print('No weights loaded, assets downloaded, packages installed, or training run.')
if __name__=='__main__':main()
