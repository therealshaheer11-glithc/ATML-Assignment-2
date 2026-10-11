"""Strict Task 3 source/config/asset checks and frozen schedules."""
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import torch
from common.data import repo_path, read_jsonl
from task3_grpo.storage import sha256


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def canonical(value):
    return json.loads(json.dumps(value,allow_nan=False))


def verify_source():
    expected = json.loads(repo_path('docs/SOURCE_SHA256.json').read_text())
    for name,digest in expected.items():
        require(repo_path(name).is_file() and sha256(repo_path(name))==digest,'Source differs: '+name)
    approval = json.loads(repo_path('docs/approval.json').read_text())
    require(approval['approved'] and approval['approved_decisions']==list('ABCDEFGHIJKLM'),'Missing A-M approval')
    return {'source_manifest_sha256':sha256(repo_path('docs/SOURCE_SHA256.json')),
            'course_commit':approval['course_commit']},approval


def verify_config(cfg):
    expected = {'updates':20,'fork_updates':8,'prompts_per_update':1,'policy_epochs':1,'learning_rate':5e-6,
        'clip_epsilon':.2,'kl_beta':.1,'num_generations':4,'group_sizes':[2,4,8],'max_prompt_length':256,
        'max_completion_length':512,'mask_truncated_completions':True,'max_grad_norm':1.,'seed':6304,
        'base_model':'Qwen/Qwen2.5-1.5B-Instruct','reward_model':'yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback',
        'reward_tokenizer':'Qwen/Qwen2.5-1.5B-Instruct','dtype':'float16','quantize_frozen_models':True,
        'generation':{'temperature':.7,'top_p':.9,'do_sample':True}}
    for k,value in expected.items():
        require(cfg.get(k)==value,'Release setting differs: '+k)
    require(cfg['lora']=={'r':8,'alpha':16,'dropout':.05,'target_modules':['q_proj','v_proj']},'LoRA differs')
    require(cfg['paths']['grpo_midpoint_policy']=='checkpoints/grpo_midpoint_policy','Midpoint path differs')
    return expected


def make_schedule(train,evaluation,seed=6304):
    indices = list(range(len(train)))
    random.Random(seed).shuffle(indices)
    return {'base_seed':seed,'algorithm':'random.Random(seed).shuffle(train row indices)',
        'training':[{'update_index':i,'row_index':index,'prompt_id':train[index]['prompt_id'],
            'source_index':train[index]['source_index'],'completion_seeds':[seed+4*i+j for j in range(4)]}
            for i,index in enumerate(indices[:20])],
        'evaluation':[{'row_index':i,'prompt_id':r['prompt_id'],'source_index':r['source_index'],
            'generation_seed':seed+100000+i} for i,r in enumerate(evaluation)],
        'probe':{'training_update_index':0,'definition':'First frozen training group; no optimizer update'},
        'fork_prefix_updates':8}


def verify_assets(approval):
    manifest = repo_path('manifests/sha256.json')
    require(sha256(manifest)==approval['asset_manifest_sha256'],'Asset manifest bytes differ')
    hashes = json.loads(manifest.read_text())
    relevant = {p:h for p,h in hashes.items() if p.startswith('checkpoints/grpo_midpoint_policy/') or
        p in {'data/rl_prompt_pool_train.jsonl','data/rl_prompt_pool_eval.jsonl','cached/grpo_k_cache.jsonl',
              'checkpoints/grpo_midpoint_metadata.json'}}
    require('checkpoints/grpo_midpoint_policy/adapter_model.safetensors' in relevant,'Missing adapter identity')
    for name,digest in relevant.items():
        require(repo_path(name).is_file() and sha256(repo_path(name))==digest,'Asset differs: '+name)
    train,evaluation = read_jsonl('data/rl_prompt_pool_train.jsonl'),read_jsonl('data/rl_prompt_pool_eval.jsonl')
    require((len(train),len(evaluation))==(1200,200),'Prompt pool sizes differ')
    a,b = {r['prompt_id'] for r in train},{r['prompt_id'] for r in evaluation}
    require(len(a)==1200 and len(b)==200 and a.isdisjoint(b),'Duplicate or overlapping prompt IDs')
    from task3_grpo.analyze_group_size import load_k8_cache
    cache = load_k8_cache(repo_path('cached/grpo_k_cache.jsonl'))
    require({r[0]['prompt_id'] for r in cache.values()}<=b,'Cache held-out provenance differs')
    return {'verified_files':relevant,'manifest_self_entry_mismatch_preserved':hashes.get('manifests/sha256.json')!=sha256(manifest)},train,evaluation


def environment(cuda=False,require_trl=True):
    pins = {'transformers':'4.57.1','tokenizers':'0.22.1','peft':'0.17.1'}
    if require_trl:
        pins['trl']='0.27.2'
    for name,version in pins.items():
        require(importlib.metadata.version(name)==version,'Pinned package differs: '+name)
    require(tuple(map(int,torch.__version__.split('+')[0].split('.')[:2]))>=(2,4),'PyTorch below minimum')
    if cuda:
        require(torch.cuda.is_available(),'Actual Colab CUDA GPU required')
    packages = json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json','--disable-pip-version-check'],text=True))
    result = {'python':platform.python_version(),'torch':torch.__version__,'packages':packages,'cuda':torch.version.cuda}
    if cuda:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        result.update({'gpu':props.name,'gpu_total_vram_gib':props.total_memory/2**30})
    return result
