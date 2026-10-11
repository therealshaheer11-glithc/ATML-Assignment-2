"""Offline, pinned production CUDA loaders with exact loading and RoPE checks."""
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_rope_utils import _compute_default_rope_parameters
from common.data import load_yaml, repo_path
from common.models import trainable_parameters
from task3_grpo.preflight import require, canonical, verify_source, verify_assets, verify_config, environment, make_schedule
from task3_grpo.reward_compat import compatible_reward_config
from task3_grpo.generation_audit import generation_protocol
from task3_grpo.storage import sha256


def loading_ok(info,label):
    require(not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),
            label+' checkpoint was not loaded exactly: '+str(info))


def validate_job(experiment):
    experiment = Path(experiment).resolve()
    source,approval = verify_source()
    cfg = load_yaml('configs/grpo.yaml')
    verify_config(cfg)
    env = environment(cuda=True)
    receipt = json.loads((experiment/'PREPARATION.json').read_text())
    require(receipt['source']==source and receipt['approval']==approval,'CPU preparation source/approval differs')
    assets = experiment/'course_assets'
    for name in ('data','cached','checkpoints','manifests'):
        p = repo_path(name)
        if not p.exists():
            p.symlink_to(assets/name,target_is_directory=True)
        require(p.is_symlink() and p.resolve()==(assets/name).resolve(),'Runtime assets point elsewhere')
    asset_audit,train,evaluation = verify_assets(approval)
    require(asset_audit==receipt['assets'],'Prepared course files differ')
    schedule = json.loads((experiment/'schedule.json').read_text())
    require(sha256(experiment/'schedule.json')==receipt['schedule_sha256'] and schedule==make_schedule(train,evaluation),'Frozen schedule differs')
    require(sha256(experiment/'cpu_tests.json')==receipt['cpu_tests_sha256'] and
            json.loads((experiment/'cpu_tests.json').read_text())['passed'],'CPU validation receipt differs')
    models = experiment/'models'
    require(sha256(models/'model_manifest.json')==receipt['model_manifest_sha256'],'Model manifest differs')
    manifest = json.loads((models/'model_manifest.json').read_text())
    folders = {}
    for kind in ('policy','reward'):
        entry = manifest[kind]
        require((entry['model_id'],entry['revision'])==(approval[kind+'_model'],approval[kind+'_revision']),'Model identity differs')
        folder = (models/entry['directory']).resolve()
        require(models.resolve() in folder.parents,'Model folder escapes preparation')
        require('config.json' in entry['sha256'] and any(n.endswith('.safetensors') for n in entry['sha256']),'Prepared weights missing')
        for name,digest in entry['sha256'].items():
            path = (folder/name).resolve()
            require(folder in path.parents and path.is_file() and sha256(path)==digest,'Prepared model file differs: '+kind+'/'+name)
        for index in folder.glob('*.safetensors.index.json'):
            require(set(json.loads(index.read_text())['weight_map'].values())<=entry['sha256'].keys(),'Weight index references unverified shards')
        folders[kind] = folder
    # Contracts use JSON normalization from their first creation, avoiding int/string key drift.
    contract_base = canonical({'source':source,'approval':approval,'assets':asset_audit,
        'model_manifest_sha256':receipt['model_manifest_sha256'],'schedule_sha256':receipt['schedule_sha256'],
        'release_config':cfg,'reward_scorer_cap':1024})
    return cfg,train,evaluation,schedule,folders,contract_base,env


def load_tokenizer(folder):
    tokenizer = AutoTokenizer.from_pretrained(str(folder),local_files_only=True,padding_side='left',use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    require(tokenizer.eos_token_id is not None,'EOS token missing')
    return tokenizer


def load_policy(cfg,folders,adapter=None,trainable=True):
    tokenizer = load_tokenizer(folders['policy'])
    base,info = AutoModelForCausalLM.from_pretrained(str(folders['policy']),dtype=torch.float16,
        local_files_only=True,low_cpu_mem_usage=True,output_loading_info=True)
    loading_ok(info,'Policy')
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base,str(adapter or repo_path(cfg['paths']['grpo_midpoint_policy'])),
        is_trainable=trainable,local_files_only=True).cuda().eval()
    if trainable:
        require(all(p.dtype==torch.float32 for p in trainable_parameters(model)),'Trainable policy precision differs; stop instead of casting')
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
    else:
        model.requires_grad_(False)
    optimizer = torch.optim.AdamW(trainable_parameters(model),lr=cfg['learning_rate'],
        betas=(.9,.999),eps=1e-8,weight_decay=.01) if trainable else None
    audit = {'loading':info,'forward_mode':'eval with gradients enabled for updates',
        'attention':model.config._attn_implementation,'generation':generation_protocol(model,tokenizer,512,cfg),
        'gradient_checkpointing_note':'Enabled as released; eval mode disables the training-only checkpoint branch.',
        'trainable_parameters':sum(p.numel() for p in trainable_parameters(model))}
    return model,tokenizer,optimizer,canonical(audit)


def load_reward(folders,tokenizer,approval):
    require(sha256(folders['reward']/'config.json')==approval['reward_config_sha256'],'Reward config identity differs')
    effective,audit = compatible_reward_config(folders['reward']/'config.json')
    require(effective.max_position_embeddings>=1024,'Reward context too small')
    model,info = AutoModelForSequenceClassification.from_pretrained(str(folders['reward']),config=effective,
        local_files_only=True,low_cpu_mem_usage=True,output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),device_map='auto')
    loading_ok(info,'Reward')
    require(getattr(model,'is_loaded_in_8bit',False),'Released 8-bit loading not active')
    require(all(p.device.type=='cuda' for p in model.parameters()),'Reward offload detected; stop for review')
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval().requires_grad_(False)
    expected,_ = _compute_default_rope_parameters(effective,torch.device('cpu'))
    torch.testing.assert_close(model.model.rotary_emb.inv_freq.detach().float().cpu(),expected,rtol=1e-6,atol=1e-7)
    audit.update({'model_weights_loaded':True,'loading':info,'frozen':True,'quantization':'8bit',
                  'loaded_position_frequencies_match':True})
    return model,canonical(audit)
