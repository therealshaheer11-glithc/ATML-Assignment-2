"""Rebuild the three required figures from verified saved JSON records on CPU."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'source'))
from task2_ppo.aggregate_cpu import NAMES,render_plots
from task2_ppo.evaluation_complete import summarize_evaluation
from task2_ppo.cache_geometry import geometry_summary

def main():
 import torch
 if torch.cuda.is_available():raise RuntimeError('Use CPU for plotting.')
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=HERE/'.runtime/rebuilt-figures')
 args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 with zipfile.ZipFile(HERE/'evidence/raw_records_verified.zip') as z:
  manifest=json.loads(z.read('FILES_SHA256.json'))
  def record(path):
   b=z.read(path)
   assert hashlib.sha256(b).hexdigest()==manifest[path],path
   return json.loads(b)
  training={n:record(f'training/{n}/final/summary.json') for n in NAMES}
  evaluation={n:summarize_evaluation([record(f'evaluation/{n}/prompt_{i:04d}/record.json') for i in range(200)]) for n in ['midpoint']+NAMES}
  cache=geometry_summary([record(f'cache_geometry/row_{i:04d}/record.json') for i in range(32)])
 render_plots(args.output,training,evaluation,cache)
 print('Rebuilt figures:',args.output.resolve())
if __name__=='__main__':main()
