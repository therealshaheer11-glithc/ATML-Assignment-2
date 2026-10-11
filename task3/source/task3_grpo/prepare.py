"""CPU-only Colab preparation: pinned assets/models, checks, frozen receipts."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from common.data import load_yaml, repo_path
from task3_grpo.storage import atomic_json, sha256
from task3_grpo.preflight import require, verify_source, verify_config, verify_assets, make_schedule, environment
from task3_grpo.reward_compat import compatible_reward_config
from task3_grpo.analyze_group_size import analyze


def prepare(experiment,resume=False):
    require(not torch.cuda.is_available(),'Use CPU Colab for downloads, tests and preparation')
    experiment = Path(experiment).resolve()
    require(str(experiment).startswith('/content/drive/MyDrive/'),'Preparation must persist to mounted Drive')
    require(Path('/content/drive/MyDrive').is_dir(),'Mount Drive first')
    source,approval = verify_source()
    cfg = load_yaml('configs/grpo.yaml')
    verify_config(cfg)
    env = environment()
    existed = experiment.exists()
    if existed and not resume:
        raise FileExistsError('Experiment root exists; explicit --resume required for the same source contract')
    require(not existed or (experiment/'preparation_owner.json').is_file(),
            'Existing experiment has no ownership receipt; stop for review')
    experiment.mkdir(parents=True,exist_ok=resume)
    start = {'source':source,'approval':approval}
    owner = experiment/'preparation_owner.json'
    if owner.exists():
        require(json.loads(owner.read_text())==start,'Existing experiment belongs to another contract')
    atomic_json(owner,start)
    # Original full course asset bundle and its original installation validator.
    assets = experiment/'course_assets'
    snapshot_download(repo_id='AbDu11aHHH/ATML-PA2-assets',repo_type='dataset',revision=approval['asset_revision'],
        local_dir=str(assets),allow_patterns=['data/**','cached/**','checkpoints/**','manifests/**'])
    for name in ('data','cached','checkpoints','manifests'):
        path = repo_path(name)
        if path.exists():
            require(path.is_symlink() and path.resolve()==(assets/name).resolve(),'Runtime asset path already exists: '+name)
        else:
            path.symlink_to(assets/name,target_is_directory=True)
    # Tiny word-limit file is tracked by the course, not part of the HF bundle.
    word = repo_path('data/word_limit_prompts.jsonl')
    tracked_word = repo_path('release_data/word_limit_prompts.jsonl')
    if not word.exists():
        shutil.copyfile(tracked_word,word)
    elif word.read_bytes()!=tracked_word.read_bytes():
        raise RuntimeError('Fixed word-limit file differs')
    manifest = json.loads(repo_path('manifests/sha256.json').read_text())
    require(sha256(repo_path('manifests/sha256.json'))==approval['asset_manifest_sha256'],'Asset manifest differs')
    # Verify all available bundle entries, except the documented stale self-entry.
    for name,digest in manifest.items():
        if name=='manifests/sha256.json':
            continue
        require(repo_path(name).is_file() and sha256(repo_path(name))==digest,'Course bundle entry differs: '+name)
    subprocess.run([sys.executable,'-m','scripts.validate_assets'],check=True)
    asset_audit,train,evaluation = verify_assets(approval)
    schedule = make_schedule(train,evaluation)
    schedule_path = experiment/'schedule.json'
    if schedule_path.exists():
        require(json.loads(schedule_path.read_text())==schedule,'Frozen schedule differs')
    atomic_json(schedule_path,schedule)
    models = experiment/'models'
    model_manifest = {}
    for kind in ('policy','reward'):
        folder = models/kind
        snapshot_download(repo_id=approval[kind+'_model'],revision=approval[kind+'_revision'],local_dir=str(folder),
            allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','merges.txt','vocab.json'])
        files = {p.relative_to(folder).as_posix():sha256(p) for p in folder.rglob('*')
                 if p.is_file() and '.cache' not in p.relative_to(folder).parts}
        require('config.json' in files and any(n.endswith('.safetensors') for n in files),'Model snapshot incomplete')
        model_manifest[kind] = {'model_id':approval[kind+'_model'],'revision':approval[kind+'_revision'],
                                'directory':kind,'sha256':files}
    require(sha256(models/'reward/config.json')==approval['reward_config_sha256'],'Pinned reward configuration differs')
    _,reward_audit = compatible_reward_config(models/'reward/config.json')
    atomic_json(models/'model_manifest.json',model_manifest)
    tests = experiment/'cpu_tests.json'
    subprocess.run([sys.executable,'scripts/run_tests.py','--output',str(tests)],check=True)
    diagnostic = analyze(repo_path('cached/grpo_k_cache.jsonl'))
    atomic_json(experiment/'group_size.json',diagnostic)
    receipt = {'format':'ATML_TASK3_PREPARATION_V1','source':source,'approval':approval,
        'assets':asset_audit,'environment':env,'schedule_sha256':sha256(schedule_path),
        'model_manifest_sha256':sha256(models/'model_manifest.json'),'reward_config_translation':reward_audit,
        'cpu_tests_sha256':sha256(tests),'group_size_sha256':sha256(experiment/'group_size.json'),
        'full_course_asset_validator_passed':True,'model_weights_loaded':False,'policy_training_performed':False}
    atomic_json(experiment/'PREPARATION.json',receipt)
    print('CPU PREPARATION COMPLETE: pinned files, tests and cache study verified on Drive. No policy training.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment',required=True)
    ap.add_argument('--resume',action='store_true')
    args = ap.parse_args()
    prepare(args.experiment,args.resume)


if __name__=='__main__':
    main()
