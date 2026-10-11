"""Run only approved Task 3 jobs; stop safely at committed-unit boundaries."""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4
import torch
from task3_grpo.runtime import validate_job, load_tokenizer, load_reward
from task3_grpo.jobs import run_probe, run_training, evaluate_condition
from task3_grpo.session_budget import SessionBudget, SessionBudgetReached
from task3_grpo.storage import atomic_json, sha256
from task3_grpo.preflight import require


def audit_persistent_units(experiment):
    receipts = {}
    for p in sorted(experiment.rglob('*.sha256.json')):
        value = json.loads(p.read_text())
        archive = p.parent/value['archive']
        require(archive.is_file() and sha256(archive)==value['sha256'],
                'Persistent archive verification failed: '+str(archive))
        receipts[str(archive.relative_to(experiment))] = value['sha256']
    return receipts


def run(experiment,local,resume=False,session_start_epoch=None,stage='all'):
    experiment = Path(experiment).resolve()
    require(str(experiment).startswith('/content/drive/MyDrive/') and Path('/content/drive/MyDrive').is_dir(),
            'Mounted Colab Drive is required')
    safe_path = experiment/'SAFE_TO_DISCONNECT.json'
    safe_path.unlink(missing_ok=True)
    budget = SessionBudget(120,session_start_epoch)
    cfg,train,rows,schedule,folders,contract,env = validate_job(experiment)
    local = Path(local).resolve()
    require(str(local).startswith('/content/'),'Use a Colab-local working folder')
    local.mkdir(parents=True,exist_ok=True)
    atomic_json(experiment/'last_gpu_environment.json',env)
    execution_id = 'gpu_session_'+uuid4().hex
    atomic_json(experiment/(execution_id+'.json'),{'execution_id':execution_id,
        'environment':env,'start_epoch':time.time(),'source':contract['source']})
    status = 'COMPLETE'
    reward = tokenizer = None
    try:
        budget.check()
        tokenizer = load_tokenizer(folders['policy'])
        reward,reward_audit = load_reward(folders,tokenizer,contract['approval'])
        reward_audit['execution_id'] = execution_id
        atomic_json(experiment/'reward_loading_audit.json',reward_audit)
        if stage in ('all','training','normalization'):
            run_probe(cfg,train,schedule,folders,contract,reward,experiment,local,budget,resume,execution_id)
            for spec in contract['approval']['required_runs']:
                if stage=='normalization' and spec['name']=='standard':
                    continue
                run_training(spec,cfg,train,schedule,folders,contract,reward,reward_audit,
                             experiment,local,budget,resume)
        if stage in ('all','evaluation'):
            from task3_grpo.storage import restore_published, verify_directory, repair_missing_receipt
            for spec in contract['approval']['required_runs']:
                final = local/'training'/spec['name']/'final'
                if not final.exists():
                    repair_missing_receipt(experiment/'training'/spec['name']/'final.zip')
                    restore_published(experiment/'training'/spec['name']/'final.zip',final)
                verify_directory(final)
            for name in ['midpoint']+[s['name'] for s in contract['approval']['required_runs']]:
                evaluate_condition(name,cfg,rows,schedule,folders,contract,reward,experiment,local,budget,resume,execution_id)
            from task3_grpo.report import build_report
            build_report(experiment)
    except SessionBudgetReached:
        status = 'PAUSED_AT_SAVED_BOUNDARY'
    finally:
        del reward,tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    # Other failures propagate: no disconnection permission is produced.
    receipts = audit_persistent_units(experiment)
    os.sync()
    receipt = {'status':status,'stage':stage,'source':contract['source'],
        'all_committed_archives_verified':True,'archives_sha256':receipts,
        'model_jobs_returned':True,'created_epoch':time.time(),
        'requires_parent_process_exit_and_successful_drive_flush':True}
    atomic_json(safe_path,receipt)
    require(json.loads(safe_path.read_text())==receipt,'Disconnect receipt failed readback')
    print(status+': completed units verified on Drive; parent notebook must flush before disconnection.',flush=True)
    return 75 if status.startswith('PAUSED') else 0


def main(default_stage='all'):
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment',required=True)
    ap.add_argument('--local',default='/content/task3_runtime/results')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--session-start-epoch',type=float)
    ap.add_argument('--stage',choices=['all','training','normalization','evaluation'],default=default_stage)
    args = ap.parse_args()
    sys.exit(run(args.experiment,args.local,args.resume,args.session_start_epoch,args.stage))


if __name__=='__main__':
    main()
