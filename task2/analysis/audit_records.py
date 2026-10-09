"""Verify exported Drive records against the frozen source and original result tables.

CPU only; never loads model weights. Requires the recorded analysis dependencies
and locally available hash-verified course prompt/cache assets and pinned tokenizer.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,statistics,sys

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def canonical(x):return json.loads(json.dumps(x,allow_nan=False))
def close(a,b,atol=1e-10):
 assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=atol),(a,b)
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write_csv(p,r):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)

def run(args):
 p,out,source,assets=[Path(x).resolve() for x in [args.records,args.output,args.source,args.assets]]
 sys.path.insert(0,str(source.resolve()))
 import torch
 from transformers import AutoTokenizer
 from common.data import prompt_messages,read_jsonl
 from task2_ppo.preflight import make_schedule,verify_config
 from task2_ppo.cache_geometry import geometry_summary,fixed_cached_advantages,reconstruct_cached
 from task2_ppo.evaluation_complete import summarize_evaluation
 from task2_ppo.continuation import stability
 if torch.cuda.is_available():raise RuntimeError('Use CPU for the audit.')
 out.mkdir(parents=True,exist_ok=True)
 manifest=read(p/'FILES_SHA256.json')
 assert {x.relative_to(p).as_posix() for x in p.rglob('*') if x.is_file()}==set(manifest)|{'FILES_SHA256.json'}
 for n,h in manifest.items():assert digest(p/n)==h,n
 original=Path(args.original);tables={r['condition']:r for r in rows(original/'held_out_metrics.csv')}
 job=read(p/'metadata/job_contract.json');cfg=job['release_config'];verify_config(cfg)
 assert job['protocol']=='TASK2_APPROVED_COMPLETE_V1'
 assert job['source']['head']=='1d64ac65acd5e45d1e4e1f415edc80455e21274e'
 approval=read(source/'docs/task2_approval.json');assert job['approvals'][0]==approval
 assert job['approvals'][1]==read(source/'docs/task2_precision_approval.json')
 assert job['source']['verified_files']==approval['source_sha256']
 for n,h in job['source']['verified_files'].items():assert digest(source/n)==h,n
 for number in (2,3):
  assert job[f'chunk{number}_files']==read(source/f'docs/task2_chunk{number}_files.json')
  for n,h in job[f'chunk{number}_files'].items():assert digest(source/n)==h,n
 resource=read(source/'docs/task2_resource_approval.json');assert job['resource_approval']==resource
 assert resource['session_minutes']==120 and resource['automatic_disconnect'] and resource['approved']
 course_manifest=read(assets/'manifests/sha256.json');assert digest(assets/'manifests/sha256.json')==approval['asset_manifest_sha256']
 for n,h in job['assets'].items():assert course_manifest[n]==h
 for n in ['data/rl_prompt_pool_train.jsonl','data/rl_prompt_pool_eval.jsonl','cached/ppo_rollout.pt']:
  assert digest(assets/n)==job['assets'][n]
 train=read_jsonl(assets/'data/rl_prompt_pool_train.jsonl');evaluation=read_jsonl(assets/'data/rl_prompt_pool_eval.jsonl')
 assert len(train)==1200 and len(evaluation)==200
 assert {r['prompt_id'] for r in train}.isdisjoint(r['prompt_id'] for r in evaluation)
 schedule=make_schedule(train,evaluation,6304)
 assert hashlib.sha256((json.dumps(schedule,indent=2)+'\n').encode()).hexdigest()==job['schedule_sha256']
 (out/'frozen_schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
 for model in ('policy','reward'):
  assert job['models']['identities'][model]=={'model_id':approval[model+'_model'],'revision':approval[model+'_revision']}
 specs=approval['required_runs'];names=[s['name'] for s in specs]
 assert names==['standard','clip_005','central_8','clip_050','kl_000','kl_020']
 frozen=read(p/'metadata/frozen_candidates.json');assert frozen['standard_is_task4_policy'] and frozen['held_out_selection'] is False
 training={};trajectory=[];optimizer=[];budget=[]
 for spec in specs:
  name=spec['name'];s=read(p/f'training/{name}/final/summary.json');training[name]=s;finite(s)
  assert s['status']=='COMPLETE' and s['completed_updates']==spec['updates'] and s['optimization_steps']==2*spec['updates']
  assert s['contract']['run']==spec and s['held_out_used_for_training'] is False
  assert s['final_policy_for_task4']==(name=='standard')
  expected_cfg={**cfg,'updates':spec['updates'],'clip_epsilon':spec['clip_epsilon'],'kl_beta':spec['kl_beta']}
  assert s['contract']['release_config']==expected_cfg
  for k in ['approvals','source','chunk2_files','assets','models','schedule_sha256']:assert s['contract'][k]==job[k],(name,k)
  original_manifest=read(p/f'training/{name}/final/ORIGINAL_FILES_SHA256.json')
  assert {k:v for k,v in original_manifest.items() if k.startswith('policy/')}==frozen['candidates'][name]
  assert digest(p/f'training/{name}/final/summary.json')==original_manifest['summary.json']
  assert len(s['history'])==spec['updates'] and s['nonfinite_failure_events']==0
  for i,h in enumerate(s['history']):
   assert h['completed_updates']==i+1 and h['schedule']==schedule['training'][i]
   assert h['prompt_messages']==prompt_messages(train[h['schedule']['row_index']])
   r=h['rollout'];n=r['valid_generated_tokens'];assert 1<=n<=512 and r['response_length']==n
   assert len(h['responses'])==len(h['terminated_with_eos'])==len(h['truncated'])==1
   assert h['truncated'][0]==(n==512 and not h['terminated_with_eos'][0])
   close(r['effective_reward'],r['raw_reward']-(0 if h['terminated_with_eos'][0] else 1))
   assert [t['ppo_epoch'] for t in h['optimization']]==[0,1]
   for t in h['optimization']:
    close(t['weighted_value_loss'],.5*t['value_mse']);assert t['clip_fraction']==0
    if t['ppo_epoch']==0:close(t['ratio_mean'],1);close(t['ratio_max_abs_change'],0)
    optimizer.append({'condition':name,'rollout_update':i+1,**t})
   for label in ('actor','critic'):
    opt=h[label+'_optimizer'];assert len(opt)==(1 if label=='actor' else 2)
    for g in opt:
     assert g['parameter_dtypes']==['torch.float32']
     assert g['betas']==[.9,.999] and g['eps']==1e-8
     assert all(g['state_dtypes'][k]==['torch.float32'] for k in ('step','exp_avg','exp_avg_sq'))
     expected={'actor':(3e-6,.01),'critic_lora':(1e-4,0.),'critic_head':(3e-4,0.)}[g['group']]
     assert (g['learning_rate'],g['weight_decay'])==expected
   trajectory.append({'condition':name,'rollout_update':i+1,**h['schedule'],**r,
     'terminated_with_eos':h['terminated_with_eos'][0],'truncated':h['truncated'][0],
     'generation_scoring_seconds':h['generation_scoring_seconds'],'optimization_seconds':h['optimization_seconds']})
  assert s['stability']==stability(s['history'])
  assert s['actual_response_tokens']==sum(h['rollout']['valid_generated_tokens'] for h in s['history'])
  assert s['allocated_response_tokens']==512*spec['updates']
  assert s['wall_clock_complete'] and all(x['wall_clock_complete'] for x in s['sessions'])
  close(s['wall_clock_seconds'],sum(x['elapsed_seconds'] for x in s['sessions']))
  for label in ['peak_allocated_gib','peak_reserved_gib']:close(s[label],max(x[label] for x in s['sessions']))
  budget.append({'condition':name,'updates':s['completed_updates'],'optimizer_steps':s['optimization_steps'],
   'allocated_response_tokens':s['allocated_response_tokens'],'actual_response_tokens':s['actual_response_tokens'],
   'max_recorded_clip_fraction':max(x['clip_fraction'] for h in s['history'] for x in h['optimization']),
   'max_recorded_abs_ratio_change':max(x['ratio_max_abs_change'] for h in s['history'] for x in h['optimization']),
   'nonfinite_failure_events':s['nonfinite_failure_events'],**s['stability']})
 # Confirm every point in the previous optimizer CSV and its timing JSON.
 prev=rows(original/'standard_optimizer_steps.csv');current=[{k:v for k,v in r.items() if k!='condition'} for r in optimizer if r['condition']=='standard']
 assert len(prev)==len(current)==40
 for a,b in zip(prev,current):
  for k in a:close(a[k],b[k])
 timing=read(original/'standard_timing_memory.json')
 assert all(training['standard'][k]==v for k,v in timing.items())
 tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,padding_side='left')
 # save_pretrained serializes runtime padding/truncation and migrates the chat
 # template to its own file. Byte equality with the original tokenizer JSON is
 # therefore not expected. Verify the template identity plus every recorded
 # prompt token sequence and decoded response below.
 template_hash=hashlib.sha256(tok.chat_template.encode()).hexdigest()
 for name in names:
  assert read(p/f'training/{name}/final/ORIGINAL_FILES_SHA256.json')['tokenizer/chat_template.jinja']==template_hash
 all_records={};metrics={};truncated_prompts=set();max_sample_error=0.;baseline_prompt_tokens=None;eval_rows=[]
 raw_pairs={(r['condition'],r['prompt_id']):r for r in rows(original/'qualitative_review.csv')}
 environments={f.stem:read(f) for f in (p/'metadata/environment_sessions').glob('*.json')}
 for name in ['midpoint']+names:
  records=[];contract=read(p/f'metadata/evaluation/{name}/contract.json')
  assert contract['job']==job and contract['forward_mode']=='eval' and contract['raw_logprobs']
  if name!='midpoint':assert contract['adapter']==frozen['candidates'][name]
  gen=contract['full_generation_protocol'];assert gen['course_generate_overrides']=={'max_new_tokens':768,'temperature':.7,'top_p':.9,'do_sample':True,'pad_token_id':151643,'eos_token_id':151645}
  assert gen['inherited_generation_config']['top_k']==20 and gen['inherited_generation_config']['repetition_penalty']==1.1
  assert contract['reward_loading']['raw_config_sha256']==approval['reward_config_sha256']
  assert contract['reward_loading']['loaded_position_frequencies_match'] and contract['reward_loading']['frozen']
  for i,entry in enumerate(schedule['evaluation']):
   folder=p/f'evaluation/{name}/prompt_{i:04d}';r=read(folder/'record.json');finite(r)
   assert digest(folder/'record.json')==read(folder/'ORIGINAL_FILES_SHA256.json')['record.json']
   assert r['schedule']==entry and r['condition']==name and r['contract']==contract
   assert r['environment_session_id'] in environments
   assert r['prompt_messages']==prompt_messages(evaluation[i])
   rendered=tok.apply_chat_template(r['prompt_messages'],tokenize=False,add_generation_prompt=True)
   ids=tok(rendered,add_special_tokens=True)['input_ids']
   if len(ids)>256:truncated_prompts.add(entry['prompt_id'])
   expected=tok(rendered,truncation=True,max_length=256)['input_ids']
   assert expected==r['prompt_token_ids']
   n=r['response_length'];assert 1<=n<=768
   assert len(r['response_token_ids'])==len(r['policy_token_logprobs'])==len(r['reference_token_logprobs'])==n
   assert tok.decode(r['response_token_ids'],skip_special_tokens=True)==r['response']
   assert r['terminated_with_eos']==(r['response_token_ids'][-1]==tok.eos_token_id)
   assert r['truncated']==(n==768 and not r['terminated_with_eos'])
   assert tok.eos_token_id not in r['response_token_ids'][:-1]
   a,b=(torch.tensor(r[k],dtype=torch.float32) for k in ['policy_token_logprobs','reference_token_logprobs'])
   for obs,expected in [(r['reference_kl'],float((a-b).mean())),(r['sampled_entropy'],float(-a.mean()))]:
    max_sample_error=max(max_sample_error,abs(obs-expected));close(obs,expected,2e-6)
   close(r['effective_reward'],r['raw_reward']-(0 if r['terminated_with_eos'] else 1))
   if name!='midpoint':
    pair=raw_pairs[(name,entry['prompt_id'])]
    assert r['response']==pair['candidate_response'];close(r['raw_reward'],pair['candidate_reward'])
    assert all_records['midpoint'][i]['response']==pair['midpoint_response'];close(all_records['midpoint'][i]['raw_reward'],pair['midpoint_reward'])
    assert r['prompt_token_ids']==all_records['midpoint'][i]['prompt_token_ids']
   records.append(r)
   eval_rows.append({'condition':name,**entry,**{k:r[k] for k in ['response_length','terminated_with_eos','truncated','raw_reward','effective_reward','reference_kl','sampled_entropy','categorical_entropy','elapsed_generation_scoring_seconds']}})
  all_records[name]=records;metrics[name]=summarize_evaluation(records)
  for k,v in metrics[name].items():
   if isinstance(v,(int,float)):close(v,tables[name][k])
   else:assert v==tables[name][k]
  assert read(p/f'metadata/evaluation/{name}/summary.json')['metrics']==metrics[name]
  print('Verified evaluation:',name,len(records),flush=True)
 cache=torch.load(assets/'cached/ppo_rollout.pt',map_location='cpu',weights_only=False)
 cache_records=[]
 for i,c in enumerate(cache):
  folder=p/f'cache_geometry/row_{i:04d}';r=read(folder/'record.json');finite(r)
  assert digest(folder/'record.json')==read(folder/'ORIGINAL_FILES_SHA256.json')['record.json']
  assert r['row_index']==i and (r['prompt_id'],r['source_index'])==(c['prompt_id'],c['source_index'])
  assert r['contract']['job']==job and r['contract']['candidate']==frozen['candidates']['standard']
  assert r['contract']['cache_sha256']==job['assets']['cached/ppo_rollout.pt']
  e=next(x for x in evaluation if x['prompt_id']==c['prompt_id']);batch=reconstruct_cached(tok,c,e,cfg)
  assert r['response_token_ids']==batch['response_ids'][0].tolist()
  fixed=fixed_cached_advantages(c,cfg)
  for k,v in fixed.items():
   if k!='mask':torch.testing.assert_close(torch.tensor(r[k]),v[0],rtol=1e-6,atol=1e-6)
  cache_records.append(r)
 geometry=geometry_summary(cache_records)
 assert geometry['rows']==32 and geometry['tokens']==8814 and geometry['training_on_cache'] is False
 clipping=rows(original/'clipping_study.csv')
 for r,previous in zip(geometry['results'],clipping):
  for k,v in r.items():
   if k!='valid_tokens':close(v,previous[k],2e-8 if k in ['clipped_surrogate','policy_loss','unclipped_surrogate'] else 1e-10)
  for k,v in training[previous['condition']]['stability'].items():
   if isinstance(v,(int,float)):close(v,previous[k])
   else:assert v==previous[k]
 pressure=rows(original/'kl_pressure_study.csv')
 for r in pressure:
  s=training[r['condition']]
  for k,v in s['stability'].items():
   if isinstance(v,(int,float)):close(v,r[k])
   else:assert v==r[k]
  close(s['actual_response_tokens'],r['actual_tokens']);close(s['allocated_response_tokens'],r['allocated_tokens'])
 (out/'cache_geometry_recomputed.json').write_text(json.dumps(geometry,indent=2)+'\n')
 write_csv(out/'training_trajectories_all_runs.csv',trajectory)
 write_csv(out/'optimizer_steps_all_runs.csv',optimizer)
 write_csv(out/'training_budgets_and_stability.csv',budget)
 write_csv(out/'evaluation_per_prompt_metrics.csv',eval_rows)
 provenance=read(p/'archive_provenance.json');assert len(provenance)==6+1400+32
 standard=next(x for x in provenance if x['archive']=='training/standard/final.zip')
 task4={'condition':'standard','updates':20,'policy_files_sha256':frozen['candidates']['standard'],
   'drive_archive':'ATML-Assignment-2/task2/experiment_v1/training/standard/final.zip','archive_receipt':standard['receipt'],'selection':'Fixed standard endpoint; no held-out selection.'}
 (out/'TASK4_PPO_HANDOFF.json').write_text(json.dumps(task4,indent=2)+'\n')
 result={'technical_audit_passed':True,'raw_export_files_verified':len(manifest),'training_runs':6,'rollout_updates':sum(s['completed_updates'] for s in training.values()),'optimizer_steps':len(optimizer),'evaluation_records':len(eval_rows),'cached_rows':32,'cached_tokens':8814,'training_and_eval_ids_disjoint':True,'frozen_source_and_asset_contracts_match':True,'optimizer_hyperparameters_and_fp32_states_verified':True,'all_recorded_optimizer_diagnostics_finite':True,'nonfinite_failure_events':0,'all_recorded_training_clip_fractions_zero':True,'maximum_sampled_statistic_recomputation_abs_error':max_sample_error,'cached_surrogate_cross_platform_absolute_tolerance':2e-8,'unique_eval_prompts_truncated_at_256':len(truncated_prompts),'task4_standard_adapter_sha256':frozen['candidates']['standard']['policy/adapter_model.safetensors'],'gpu_used_for_audit':False,'model_inference_rerun':False,'qualitative_approval':'See provenance/qualitative_review_approval.json for the explicit user decision','limits':['Saved categorical entropy aggregated and checked finite; full logits are not in the export for independent per-token categorical-entropy recalculation.','Model weights not loaded or inferred again.','Actual generated token counts differ under the approved equal-allowance interpretation.','No active training clipping; between-fork differences are not evidence of clipping intervention effects.']}
 (out/'RECORD_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for n in ['records','output','source','assets','tokenizer','original']:parser.add_argument('--'+n,required=True)
 run(parser.parse_args())
