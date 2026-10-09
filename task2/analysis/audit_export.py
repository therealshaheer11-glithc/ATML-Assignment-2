"""Audit the original CPU export and write separate, corrected presentation tables.

Standard library only. Never changes original evidence or the historical source.
Raw-record verification is a separate required audit, recorded as pending here.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,re,statistics

CONDITIONS=['midpoint','standard','clip_005','central_8','clip_050','kl_000','kl_020']
CACHE_TOKENS=8814  # Independently counted in the hash-verified released 32-row cache.
CACHE_SHA='6d9c28c1cc534b60410640f2f23694c227d4d653bf8088e9b1b2d6bf39be8b6d'
PROMPT_ID='ee7b7003b96f186726812f27c0c689da5283da51a6a682eb1f65046fda17ba65'

def csv_rows(path):
 with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
 with Path(path).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def close(a,b):
 assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)
def title_lengths(text):
 # Numbered titles only; punctuation-only tokens such as '-' and '&' are not words.
 # Apostrophes and internal hyphens stay within one whitespace-delimited word.
 titles=re.findall(r'^\s*(\d+)\.\s*(.*)$',text,re.M)
 assert [int(n) for n,_ in titles]==list(range(1,21))
 return [(int(n),sum(any(c.isalnum() for c in word) for word in title.split())) for n,title in titles]

def audit(original,destination):
 original,destination=Path(original),Path(destination)
 destination.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((original/'FILES_SHA256.json').read_text())
 actual={p.relative_to(original).as_posix() for p in original.rglob('*') if p.is_file()}
 assert actual==set(manifest)|{'FILES_SHA256.json'}
 for name,expected in manifest.items():assert hashlib.sha256((original/name).read_bytes()).hexdigest()==expected,name
 held=csv_rows(original/'held_out_metrics.csv');assert [r['condition'] for r in held]==CONDITIONS
 by_name={r['condition']:r for r in held}
 pairs=csv_rows(original/'qualitative_review.csv');assert len(pairs)==1200
 baseline={};comparison=[]
 for name in CONDITIONS[1:]:
  group=[r for r in pairs if r['condition']==name]
  assert len(group)==200 and len({r['prompt_id'] for r in group})==200
  for row in group:
   value=tuple(row[k] for k in ['source_index','prompt_messages','midpoint_response','midpoint_reward'])
   if row['prompt_id'] in baseline:assert value==baseline[row['prompt_id']]
   baseline[row['prompt_id']]=value
   close(float(row['candidate_reward'])-float(row['midpoint_reward']),row['reward_delta'])
  scores=[float(r['candidate_reward']) for r in group]
  close(statistics.mean(scores),by_name[name]['raw_reward_mean'])
  close(statistics.pstdev(scores),by_name[name]['raw_reward_std_population'])
  comparison.append({'condition':name,'paired_prompts':len(group),
   'mean_raw_reward_change':statistics.mean(float(r['reward_delta']) for r in group),
   'identical_response_count':sum(r['candidate_response']==r['midpoint_response'] for r in group),
   'reward_change_std_population':statistics.pstdev(float(r['reward_delta']) for r in group)})
 values=[float(v[-1]) for v in baseline.values()]
 close(statistics.mean(values),by_name['midpoint']['raw_reward_mean']);close(statistics.pstdev(values),by_name['midpoint']['raw_reward_std_population'])
 for row in held:
  assert int(row['prompts'])==200
  close(float(row['response_length_mean'])*200,row['valid_tokens'])
  close(float(row['raw_reward_mean'])-(1-float(row['eos_rate'])),row['effective_reward_mean'])
  close(float(row['eos_rate'])+float(row['truncation_rate']),1)
  for k,v in row.items():
   if k not in ('condition','aggregation'):assert math.isfinite(float(v)),(k,v)
 steps=csv_rows(original/'standard_optimizer_steps.csv')
 assert [(int(r['rollout_update']),int(r['ppo_epoch'])) for r in steps]==[(i,e) for i in range(1,21) for e in (0,1)]
 for r in steps:
  assert all(math.isfinite(float(v)) for v in r.values())
  close(float(r['value_mse'])*.5,r['weighted_value_loss'])
  if int(r['ppo_epoch'])==0:close(r['ratio_mean'],1);close(r['ratio_max_abs_change'],0)
 timing=json.loads((original/'standard_timing_memory.json').read_text())
 assert timing['wall_clock_complete'] and timing['sessions'][0]['completed_updates']==20
 for s in timing['sessions']:
  a=s['model_loading']
  assert a['forward_mode']=='eval with gradients enabled'
  assert all(p['dtype']==('torch.float32' if p['trainable'] else 'torch.float16') for p in a['critic_precision']['parameters'])
  assert a['reward']['frozen'] and a['reward']['quantization']=='8bit' and a['reward']['loaded_position_frequencies_match']
  for field in ('policy_loading','critic_loading'):
   assert not any(a[field].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
 clipping=csv_rows(original/'clipping_study.csv');pressure=csv_rows(original/'kl_pressure_study.csv')
 corrected=[]
 for r in clipping:
  for k,v in by_name[r['condition']].items():assert r[k]==v,(r['condition'],k)
  close(int(r['affected_token_count'])/CACHE_TOKENS,r['affected_token_fraction'])
  close(int(r['active_clipped_branch_count'])/CACHE_TOKENS,r['active_clipped_branch_fraction'])
  close(-float(r['clipped_surrogate']),r['policy_loss'])
  assert int(r['allocated_tokens'])==4096 and int(r['optimizer_steps'])==16 and int(r['nonfinite_steps'])==0
  # Original dict merge overwrote these cache fields with held-out fields.
  revised={'condition':r['condition'],'epsilon':r['epsilon'],'cached_rows':32,'cached_valid_tokens':CACHE_TOKENS,
   'cache_aggregation':'masked token mean over all fixed cached responses'}
  cache_fields=['clipped_surrogate','policy_loss','unclipped_surrogate','affected_token_count','affected_token_fraction','active_clipped_branch_count','active_clipped_branch_fraction']
  for k in cache_fields:revised['cache_'+k]=r[k]
  for k,v in r.items():
   if k not in ['condition','epsilon']+cache_fields:
    revised[{'valid_tokens':'held_out_valid_tokens','prompts':'held_out_prompts','aggregation':'held_out_aggregation'}.get(k,k)]=v
  corrected.append(revised)
 for r in pressure:
  for k,v in by_name[r['condition']].items():assert r[k]==v
  assert int(r['allocated_tokens'])==4096 and int(r['optimizer_steps'])==16 and int(r['nonfinite_steps'])==0
 approval_file=Path(__file__).resolve().parents[1]/'provenance/qualitative_review_approval.json'
 approval=json.loads(approval_file.read_text()) if approval_file.exists() else {'approved':False}
 approved=approval.get('approved') is True
 chosen=[];lengths=[]
 for name,relation in [('kl_020','agreement'),('kl_000','disagreement')]:
  r=next(x for x in pairs if x['condition']==name and x['prompt_id']==PROMPT_ID)
  old,new=title_lengths(r['midpoint_response']),title_lengths(r['candidate_response'])
  counts=[sum(6<=count<=10 for _,count in values) for values in [old,new]]
  assert counts==([18,20] if name=='kl_020' else [18,17])
  for label,values in [('midpoint',old),(name,new)]:
   for number,count in values:lengths.append({'comparison':name,'response':label,'title_number':number,'word_count':count,'complies_6_to_10':6<=count<=10})
  chosen.append({**r,'quality_direction':'better' if counts[1]>counts[0] else 'worse',
   'criterion':'Instruction compliance: 20 titles, each 6-10 words',
   'reasoning':f'{counts[0]}/20 midpoint titles and {counts[1]}/20 candidate titles meet the stated word-length bound. Both produce 20 titles. Reward increases. This is {relation} for this explicit constraint only; overall factual or stylistic quality is not scored.',
   'verification_sources':'Saved model output plus deterministic title word counts; see qualitative_title_counts.csv',
   'review_status':('Assistant analysis explicitly approved by the user on '+approval['approval_date']+'; see provenance/qualitative_review_approval.json') if approved else 'Proposed assistant analysis; user approval recorded separately'})
 write_csv(destination/'clipping_study_corrected.csv',corrected)
 write_csv(destination/'paired_reward_comparisons.csv',comparison)
 write_csv(destination/('qualitative_examples.csv' if approved else 'qualitative_examples_proposed.csv'),chosen)
 write_csv(destination/'qualitative_title_counts.csv',lengths)
 result={'export_checks_passed':True,'verified_original_files':len(manifest),'conditions':7,'prompts_per_condition':200,
  'qualitative_pair_rows':len(pairs),'standard_optimizer_steps':len(steps),'original_export_modified':False,
  'cached_rows':32,'cached_tokens':CACHE_TOKENS,'course_cache_sha256':CACHE_SHA,
  'presentation_fix':'Separate cache and held-out token counts and aggregation definitions; numerical rewards, objectives and fractions unchanged.',
  'raw_training_and_per_prompt_metric_audit':'Recorded separately in RECORD_AUDIT.json',
  'qualitative_approval_record':'See provenance/qualitative_review_approval.json',
  'scope':'Summary export integrity and presentation corrections; underlying records are audited separately'}
 (destination/'EXPORT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--original',required=True);parser.add_argument('--output',required=True)
 args=parser.parse_args();audit(args.original,args.output)
