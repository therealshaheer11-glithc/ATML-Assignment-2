"""Unpack the verified, weights-free audit records for CPU analysis."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,stat,zipfile
HERE=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,default=HERE/'.runtime/records')
 args=parser.parse_args();out=args.output.expanduser().resolve()
 if out.exists():raise SystemExit('Output exists; choose a fresh folder. Nothing overwritten.')
 with zipfile.ZipFile(HERE/'evidence/raw_records_verified.zip') as z:
  names=z.namelist();manifest=json.loads(z.read('FILES_SHA256.json'))
  assert len(names)==len(set(names)) and set(names)==set(manifest)|{'FILES_SHA256.json'}
  for i in z.infolist():
   p=PurePosixPath(i.filename)
   assert not p.is_absolute() and '..' not in p.parts and not stat.S_ISLNK(i.external_attr>>16)
  for name,expected in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
  out.mkdir(parents=True,exist_ok=False);z.extractall(out)
 print('Verified records extracted:',out)
if __name__=='__main__':main()
