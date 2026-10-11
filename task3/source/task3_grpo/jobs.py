"""Recoverable standard/fork/probe/evaluation jobs; actual GPU execution only."""
import gc
import json
import statistics
import tempfile
import time
from pathlib import Path
from uuid import uuid4
import torch
from common.data import prompt_messages
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import sampled_kl, sample_entropy
from common.models import reference_mode
from task3_grpo.checkpoint import snapshot, restore, cpu_tree, optimizer_audit
from task3_grpo.runtime import load_policy
from task3_grpo.storage import atomic_json, commit_directory, publish_directory, restore_published, verify_directory, repair_missing_receipt
from task3_grpo.preflight import require, canonical
from task3_grpo.training_core import (generate_group, ordinary, chosen_logprobs, categorical_entropy,
    freeze_rollout, optimize_rollout, length_gradient_records, finite)
from task3_grpo.session_budget import SessionBudgetReached


def contract_paths(out,saved,contract,resume):
    if (out.exists() or saved.exists()) and not resume:
        raise FileExistsError('Existing evidence requires explicit resume: '+str(saved))
    out.mkdir(parents=True,exist_ok=True)
    saved.mkdir(parents=True,exist_ok=True)
    for path in (out/'contract.json',saved/'contract.json'):
        if path.exists():
            require(json.loads(path.read_text())==contract,'Existing run contract differs: '+str(path))
        atomic_json(path,contract)
        require(json.loads(path.read_text())==contract,'Saved contract failed readback')


def recover_units(out,saved,pattern):
    for archive in sorted(saved.glob(pattern+'.zip')):
        repair_missing_receipt(archive)
        destination = out/archive.stem
        if not destination.exists():
            restore_published(archive,destination)
        else:
            verify_directory(destination)
    for folder in sorted(out.glob(pattern)):
        verify_directory(folder)
        publish_directory(folder,saved)  # Repairs interruption between local commit and Drive copy.


def run_training(spec,cfg,train,schedule,folders,base_contract,reward,reward_audit,experiment,local,budget,resume):
    name = spec['name']
    out,saved = Path(local)/'training'/name,Path(experiment)/'training'/name
    contract = canonical(dict(base_contract,run=spec))
    contract_paths(out,saved,contract,resume)
    if not (out/'final').exists() and (saved/'final.zip').exists():
        repair_missing_receipt(saved/'final.zip')
        restore_published(saved/'final.zip',out/'final')
    if (out/'final').exists():
        verify_directory(out/'final')
        summary = json.loads((out/'final/summary.json').read_text())
        require(summary['contract']==contract and summary['completed_updates']==spec['updates'],'Final run contract/count differs')
        publish_directory(out/'final',saved)
        return summary
    budget.check()
    policy,tokenizer,optimizer,loading = load_policy(cfg,folders)
    recover_units(out,saved,'step_[0-9][0-9][0-9][0-9]')
    history = []
    steps = sorted(out.glob('step_[0-9][0-9][0-9][0-9]'))
    for i,folder in enumerate(steps,1):
        record = json.loads((folder/'update.json').read_text())
        require(folder.name==f'step_{i:04d}' and record['completed_updates']==i and record['contract']==contract,'Update history gap/provenance differs')
        history.append(record)
    completed = len(history)
    require(completed<=spec['updates'],'Completed count exceeds run budget')
    if steps:
        state = torch.load(steps[-1]/'checkpoint.pt',map_location='cpu',weights_only=False)
        require(restore(state,policy,optimizer,contract)==completed,'Saved state/update count differs')
        del state
    session_name = 'session_'+uuid4().hex+'.json'
    session = {'status':'RUNNING','start_completed_updates':completed,'elapsed_seconds':0.,
        'wall_clock_complete':False,'model_loading':loading,'reward_loading':reward_audit,
        'peak_allocated_gib':0.,'peak_reserved_gib':0.}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    def clock(status,complete=False):
        torch.cuda.synchronize()
        session.update({'status':status,'completed_updates':completed,'elapsed_seconds':time.perf_counter()-start,
            'wall_clock_complete':complete,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30})
        atomic_json(saved/session_name,session)

    try:
        clock('RUNNING')
        for index in range(completed,spec['updates']):
            budget.check()
            entry = schedule['training'][index]
            row = train[entry['row_index']]
            require((row['prompt_id'],row['source_index'])==(entry['prompt_id'],entry['source_index']),'Training schedule differs')
            messages = prompt_messages(row)
            batch = generate_group(policy,tokenizer,messages,entry,cfg)
            rewards = score_reward_pairs(reward,tokenizer,[messages]*4,batch['responses'],max_length=1024)
            fixed,diagnostics = freeze_rollout(policy,batch,rewards,cfg)
            optimization = optimize_rollout(policy,optimizer,batch,fixed,cfg,spec['loss_type'])
            record = {'completed_updates':index+1,'contract':contract,'schedule':entry,'prompt_messages':messages,
                'execution_id':reward_audit.get('execution_id'),
                'responses':batch['responses'],'terminated_with_eos':batch['terminated_with_eos'],
                'truncated':batch['truncated'],'rollout':diagnostics,'optimization':optimization,
                'optimizer':optimizer_audit(optimizer)}
            with tempfile.TemporaryDirectory(prefix='.uncommitted_',dir=out) as temporary:
                temporary = Path(temporary)
                atomic_json(temporary/'update.json',record)
                torch.save(cpu_tree({'batch':batch,'fixed':fixed}),temporary/'rollout.pt')
                torch.save(snapshot(policy,optimizer,index+1,contract),temporary/'checkpoint.pt')
                destination = out/f'step_{index+1:04d}'
                commit_directory(temporary,destination)
            completed = index+1
            history.append(record)
            publish_directory(destination,saved)
            clock('RUNNING')
            print(f'{name}: {completed}/{spec["updates"]}; reward {diagnostics["raw_reward"]:.4f}; checkpoint verified',flush=True)
            del batch,fixed,rewards
        clock('TRAINING_COMPLETE',True)
        sessions = [json.loads(p.read_text()) for p in sorted(saved.glob('session_*.json'))]
        summary = {'status':'COMPLETE','contract':contract,'completed_updates':completed,'history':history,
            'allocated_response_tokens':completed*4*512,
            'actual_response_tokens':sum(r['rollout']['valid_generated_tokens'] for r in history),
            'loss_bearing_tokens':sum(r['rollout']['loss_bearing_tokens'] for r in history),
            'wall_clock_seconds':sum(s['elapsed_seconds'] for s in sessions),
            'wall_clock_complete':all(s['wall_clock_complete'] for s in sessions),
            'wall_clock_definition':'Active continuation sessions after loading, including rollout/scoring/update/checkpoint persistence; setup/export excluded. Hard disconnect intervals may be incomplete.',
            'peak_allocated_gib':max(s['peak_allocated_gib'] for s in sessions),
            'peak_reserved_gib':max(s['peak_reserved_gib'] for s in sessions),'sessions':sessions,
            'final_policy_for_task4':name=='standard','held_out_used_for_training':False}
        with tempfile.TemporaryDirectory(prefix='.export_',dir=out) as temporary:
            temporary = Path(temporary)
            policy.save_pretrained(temporary/'policy',safe_serialization=True)
            atomic_json(temporary/'summary.json',summary)
            commit_directory(temporary,out/'final')
        publish_directory(out/'final',saved)
        clock('COMPLETE',True)
        return summary
    except BaseException as error:
        try:
            session['error'] = {'type':type(error).__name__,'message':str(error)}
            clock('PAUSED' if isinstance(error,SessionBudgetReached) else 'FAILED',True)
        except BaseException:
            pass
        raise
    finally:
        del policy,tokenizer,optimizer
        gc.collect()
        torch.cuda.empty_cache()


def run_probe(cfg,train,schedule,folders,contract,reward,experiment,local,budget,resume,execution_id=None):
    out,saved = Path(local)/'normalization_probe',Path(experiment)/'normalization_probe'
    contract_paths(out,saved,contract,resume)
    recover_units(out,saved,'probe')
    if (out/'probe').exists():
        record = json.loads((out/'probe/record.json').read_text())
        require(record['contract']==contract,'Probe contract differs')
        return record
    budget.check()
    policy,tokenizer,optimizer,loading = load_policy(cfg,folders)
    try:
        entry = schedule['training'][0]
        messages = prompt_messages(train[entry['row_index']])
        batch = generate_group(policy,tokenizer,messages,entry,cfg)
        rewards = score_reward_pairs(reward,tokenizer,[messages]*4,batch['responses'],max_length=1024)
        fixed,diagnostics = freeze_rollout(policy,batch,rewards,cfg)
        # Detached leaf avoids full parameter graph: this is the declared logp derivative probe.
        probe_logp = fixed['old_logprobs'].detach().clone().requires_grad_(True)
        conditions = {}
        for kind in ('grpo','dr_grpo'):
            loss,detail,gradients = length_gradient_records(probe_logp,fixed,cfg,kind)
            conditions[kind] = {'loss':float(loss.detach()),'gradient_statistics':gradients}
        record = {'contract':contract,'schedule':entry,'rollout':diagnostics,'conditions':conditions,
            'execution_id':execution_id,
            'optimizer_updates':0,'excluded_from_continuation_budget':True,'loading':loading}
        with tempfile.TemporaryDirectory(dir=out) as temporary:
            atomic_json(Path(temporary)/'record.json',record)
            torch.save(cpu_tree({'batch':batch,'fixed':fixed}),Path(temporary)/'rollout.pt')
            commit_directory(temporary,out/'probe')
        publish_directory(out/'probe',saved)
        return record
    finally:
        del policy,tokenizer,optimizer
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_prompt(policy,tokenizer,reward,row,entry,cfg):
    require(not policy.training,'Frozen evaluation requires eval mode')
    require((row['prompt_id'],row['source_index'])==(entry['prompt_id'],entry['source_index']),'Evaluation schedule differs')
    messages = prompt_messages(row)
    set_seed(entry['generation_seed'])
    batch = ordinary(batch_generate(policy,tokenizer,[messages],max_prompt_length=256,max_new_tokens=512,**cfg['generation']))
    mask = batch['response_mask']
    logp,logits = chosen_logprobs(policy,batch)
    entropy = categorical_entropy(logits,mask)
    del logits
    with reference_mode(policy):
        ref,logits = chosen_logprobs(policy,batch)
        del logits
    rewards = score_reward_pairs(reward,tokenizer,[messages],batch['responses'],max_length=1024)
    for name,tensor in [('policy logp',logp),('reference logp',ref),('reward',rewards)]:
        finite(name,tensor)
    n = int(mask.sum())
    reward_text = tokenizer.apply_chat_template(messages+[{'role':'assistant','content':batch['responses'][0]}],tokenize=False,add_generation_prompt=False)
    reward_input_tokens = len(tokenizer(reward_text,add_special_tokens=True)['input_ids'])
    prompt_tokens = len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
    return {'schedule':entry,'prompt_messages':messages,'response':batch['responses'][0],
        'prompt_token_ids':batch['sequences'][0,:batch['prompt_width']].cpu().tolist(),
        'response_token_ids':batch['response_ids'][0,:n].cpu().tolist(),'response_length':n,
        'terminated_with_eos':batch['terminated_with_eos'][0],'truncated':batch['truncated'][0],
        'raw_reward':float(rewards[0]),'reference_kl':float(sampled_kl(logp,ref,mask)),
        'sampled_entropy':float(sample_entropy(logp,mask)),'categorical_entropy':float(entropy),
        'policy_token_logprobs':logp[0,:n].cpu().tolist(),'reference_token_logprobs':ref[0,:n].cpu().tolist(),
        'reward_input_tokens_before_truncation':reward_input_tokens,'reward_input_truncated':reward_input_tokens>1024,
        'prompt_tokens_before_truncation':prompt_tokens,'policy_prompt_truncated':prompt_tokens>256}


def summarize_evaluation(records):
    if not records:
        raise ValueError('Empty evaluation')
    n = sum(r['response_length'] for r in records)
    require(n>0,'No evaluated response tokens')
    result = {'prompts':len(records),'valid_tokens':n,'reference_kl_definition':'signed sampled policy-minus-reference logp over valid response tokens',
        'eos_rate':statistics.mean(float(r['terminated_with_eos']) for r in records),
        'truncation_rate':statistics.mean(float(r['truncated']) for r in records)}
    for key in ('raw_reward','reference_kl','sampled_entropy','categorical_entropy','response_length'):
        values = [r[key] for r in records]
        result[key+'_prompt_mean'] = statistics.mean(values)
        result[key+'_std_population'] = statistics.pstdev(values)
    for key in ('reference_kl','sampled_entropy','categorical_entropy'):
        result[key+'_token_mean'] = sum(r[key]*r['response_length'] for r in records)/n
    return result


def evaluate_condition(name,cfg,rows,schedule,folders,base_contract,reward,experiment,local,budget,resume,execution_id=None):
    training = Path(local)/'training'/name/'final'
    if name!='midpoint':
        verify_directory(training)
        final_summary = json.loads((training/'summary.json').read_text())
        require(final_summary['contract']==canonical(dict(base_contract,run=next(r for r in base_contract['approval']['required_runs'] if r['name']==name))),
                'Evaluation endpoint contract differs')
        endpoint_identity = verify_directory(training)
    else:
        endpoint_identity = base_contract['assets']['verified_files']
    contract = canonical(dict(base_contract,condition=name,endpoint_identity=endpoint_identity))
    out,saved = Path(local)/'evaluation'/name,Path(experiment)/'evaluation'/name
    contract_paths(out,saved,contract,resume)
    recover_units(out,saved,'prompt_[0-9][0-9][0-9][0-9]')
    # Load lazily so completed evaluations need no model reload.
    policy = tokenizer = None
    records = []
    try:
        for entry in schedule['evaluation']:
            folder = out/f'prompt_{entry["row_index"]:04d}'
            if folder.exists():
                verify_directory(folder)
                record = json.loads((folder/'record.json').read_text())
                require(record['contract']==contract and record['schedule']==entry,'Saved evaluation provenance differs')
            else:
                budget.check()
                if policy is None:
                    policy,tokenizer,_,_ = load_policy(cfg,folders,adapter=None if name=='midpoint' else training/'policy',trainable=False)
                record = evaluate_prompt(policy,tokenizer,reward,rows[entry['row_index']],entry,cfg)
                record.update({'condition':name,'contract':contract})
                record['execution_id'] = execution_id
                with tempfile.TemporaryDirectory(dir=out) as temporary:
                    atomic_json(Path(temporary)/'record.json',record)
                    commit_directory(temporary,folder)
            publish_directory(folder,saved)
            records.append(record)
            if len(records)%10==0:
                print(f'Evaluation {name}: {len(records)}/{len(rows)} prompts verified',flush=True)
        require(len(records)==len(rows)==200,'Incomplete fixed evaluation set')
        summary = {'status':'COMPLETE','condition':name,'contract':contract,'metrics':summarize_evaluation(records),'selection_or_tuning':False}
        atomic_json(saved/'summary.json',summary)
        return summary
    finally:
        del policy,tokenizer
        gc.collect()
        torch.cuda.empty_cache()
