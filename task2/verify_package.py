"""Verify this Task 2 Git addition without installing dependencies or using a GPU."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent

def main():
 manifest=json.loads((ROOT/'PACKAGE_SHA256.json').read_text())
 def included(p):
  rel=p.relative_to(ROOT)
  return p.is_file() and not any(part in ('.runtime','__pycache__','.git') for part in rel.parts) and p.suffix!='.pyc' and p.name!='.DS_Store' and rel.as_posix()!='PACKAGE_SHA256.json'
 actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if included(p)}
 if actual!=set(manifest):raise SystemExit('File inventory differs: '+str(actual.symmetric_difference(manifest)))
 for name,expected in manifest.items():
  p=ROOT/name
  if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise SystemExit('Checksum differs: '+name)
 print(f'PACKAGE VERIFIED: {len(manifest)} files. No GPU used.')
 print('Audit:',json.loads((ROOT/'FINAL_AUDIT.json').read_text())['technical_audit'])
 print('Qualitative status:',json.loads((ROOT/'provenance/qualitative_review_approval.json').read_text())['status'])
if __name__=='__main__':main()
