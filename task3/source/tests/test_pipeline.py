import contextlib
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model
from common.data import load_yaml
from common.models import reference_mode, trainable_parameters
from common.generation import _response_mask
from task3_grpo.checkpoint import snapshot, restore, trainable_state
from task3_grpo.training_core import freeze_rollout, optimize_rollout, merge_completions, generate_group
from task3_grpo.preflight import make_schedule, verify_config
from task3_grpo.jobs import run_training, run_probe, summarize_evaluation, contract_paths, evaluate_condition
from task3_grpo.session_budget import UnlimitedTestBudget, SessionBudgetReached
from task3_grpo.analyze_group_size import analyze, load_k8_cache, regroup_equal_generation_budget
from task3_grpo.reward_compat import compatible_reward_config


def tiny_model():
    torch.manual_seed(6304)
    model = get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=1024)),
        LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    return model


def batch():
    response = torch.tensor([[5,2,0],[6,7,2],[8,2,0],[9,10,11]])
    mask = _response_mask(response,2)
    # End padding has its own mask even when pad != EOS.
    sequences = torch.cat([torch.tensor([[1,4]]*4),response],1)
    return {'sequences':sequences,'attention_mask':torch.cat([torch.ones(4,2,dtype=torch.long),mask.long()],1),
        'prompt_width':2,'response_ids':response,'response_mask':mask,'responses':['a','b','c','d'],
        'terminated_with_eos':[True,True,True,False],'truncated':[False,False,False,True],
        'response_lengths':[2,3,2,3]}


def optimizer(model):
    return torch.optim.AdamW(trainable_parameters(model),lr=5e-6,betas=(.9,.999),eps=1e-8,weight_decay=.01)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_yaml('configs/grpo.yaml')
    def test_release_config_exact(self): verify_config(self.cfg)
    def test_real_tiny_generation_is_seeded_and_backpropagates(self):
        class Tokenizer:
            pad_token_id=0
            eos_token_id=2
            def apply_chat_template(self,*a,**k): return 'synthetic'
            def __call__(self,rendered,**k):
                return {'input_ids':torch.tensor([[1,4]]*len(rendered)),
                        'attention_mask':torch.ones(len(rendered),2,dtype=torch.long)}
            def decode(self,ids,**k): return ' '.join(map(str,ids.tolist()))
        model=tiny_model(); cfg=copy.deepcopy(self.cfg); cfg['max_completion_length']=4
        entry={'completion_seeds':[6304,6305,6306,6307]}
        a=generate_group(model,Tokenizer(),[],entry,cfg)
        b=generate_group(model,Tokenizer(),[],entry,cfg)
        torch.testing.assert_close(a['sequences'],b['sequences'],rtol=0,atol=0)
        self.assertFalse(a['sequences'].is_inference())
        fixed,_=freeze_rollout(model,a,torch.arange(4.),cfg)
        result=optimize_rollout(model,optimizer(model),a,fixed,cfg,'grpo')
        self.assertTrue(np.isfinite(result['loss']))
    def test_release_config_change_rejected(self):
        self.cfg['learning_rate']=1e-4
        with self.assertRaises(RuntimeError): verify_config(self.cfg)
    def test_freeze_reference_and_ratio_identity(self):
        model = tiny_model()
        fixed,diag = freeze_rollout(model,batch(),torch.tensor([-1.,0.,1.,2.]),self.cfg)
        self.assertTrue(all(not v.requires_grad for v in fixed.values()))
        self.assertEqual(float(fixed['loss_mask'][-1].sum()),0.)
        self.assertEqual(diag['valid_generated_tokens'],10)
        with reference_mode(model):
            reference_before = model(input_ids=batch()['sequences']).logits.detach().clone()
        opt = optimizer(model)
        result = optimize_rollout(model,opt,batch(),fixed,self.cfg,'grpo')
        self.assertEqual(result['ratio_max_abs_change_before_update'],0.)
        self.assertGreater(result['gradient_norm_preclip'],0.)
        self.assertFalse(model.training)
        with reference_mode(model):
            reference_after = model(input_ids=batch()['sequences']).logits.detach()
        torch.testing.assert_close(reference_before,reference_after,rtol=0,atol=0)
    def test_nonfinite_fails_before_optimizer(self):
        model = tiny_model(); opt=optimizer(model)
        with self.assertRaises(ValueError): freeze_rollout(model,batch(),torch.tensor([0.,1.,float('inf'),3.]),self.cfg)
        self.assertFalse(opt.state)
    def test_training_mode_rejected(self):
        with self.assertRaises(RuntimeError): freeze_rollout(tiny_model().train(),batch(),torch.arange(4.),self.cfg)
    def test_all_masked_zero_gradient_still_one_fixed_adamw_step(self):
        model=tiny_model(); b=batch(); b['truncated']=[True]*4
        fixed,_=freeze_rollout(model,b,torch.arange(4.),self.cfg)
        result=optimize_rollout(model,optimizer(model),b,fixed,self.cfg,'grpo')
        self.assertTrue(result['all_completions_masked'])
        self.assertEqual(result['loss'],0.)
    def test_checkpoint_restores_adamw_rng_and_next_update(self):
        model=tiny_model(); opt=optimizer(model)
        fixed,_=freeze_rollout(model,batch(),torch.arange(4.),self.cfg)
        optimize_rollout(model,opt,batch(),fixed,self.cfg,'grpo')
        state=snapshot(model,opt,1,{'condition':'synthetic'})
        expected_random=(random.random(),np.random.rand(),torch.rand(1))
        next_fixed,_=freeze_rollout(model,batch(),torch.arange(4.),self.cfg)
        optimize_rollout(model,opt,batch(),next_fixed,self.cfg,'grpo')
        expected=trainable_state(model)
        recovered=tiny_model(); recovered_opt=optimizer(recovered)
        self.assertEqual(restore(state,recovered,recovered_opt,{'condition':'synthetic'}),1)
        actual=(random.random(),np.random.rand(),torch.rand(1))
        self.assertEqual(actual[:2],expected_random[:2]); torch.testing.assert_close(actual[2],expected_random[2])
        fixed,_=freeze_rollout(recovered,batch(),torch.arange(4.),self.cfg)
        optimize_rollout(recovered,recovered_opt,batch(),fixed,self.cfg,'grpo')
        for k,v in trainable_state(recovered).items(): torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
    def test_checkpoint_contract_mismatch_rejected(self):
        model=tiny_model(); opt=optimizer(model); state=snapshot(model,opt,0,{'a':1})
        with self.assertRaises(RuntimeError): restore(state,model,opt,{'a':2})
    def test_merge_preserves_eos_padding(self):
        b=batch(); pieces=[]
        for i,n in enumerate(b['response_lengths']):
            piece={k:(v[i:i+1,:2+n] if k in ('sequences','attention_mask') else v[i:i+1,:n]) if isinstance(v,torch.Tensor)
                   else [v[i]] if isinstance(v,list) else v for k,v in b.items()}
            pieces.append(piece)
        merged=merge_completions(pieces,0)
        torch.testing.assert_close(merged['response_mask'],b['response_mask'])
        torch.testing.assert_close(merged['attention_mask'],b['attention_mask'])
    def test_eos_at_cap_is_not_truncated(self):
        torch.testing.assert_close(_response_mask(torch.tensor([[5,6,2],[5,2,2]]),2),torch.tensor([[1.,1,1],[1.,1,0]]))
    def test_schedule_deterministic_and_fork_prefix(self):
        train=[{'prompt_id':str(i),'source_index':i} for i in range(1200)]
        a=make_schedule(train,train[:200]); b=make_schedule(train,train[:200])
        self.assertEqual(a,b); self.assertEqual(len({r['row_index'] for r in a['training']}),20)
        self.assertEqual(a['training'][0]['completion_seeds'],[6304,6305,6306,6307])
    def test_evaluation_uses_generated_lengths(self):
        rows=[dict(response_length=n,raw_reward=r,reference_kl=k,sampled_entropy=2.,categorical_entropy=1.,terminated_with_eos=e,truncated=not e)
              for n,r,k,e in [(2,1.,-1.,True),(4,3.,2.,False)]]
        result=summarize_evaluation(rows)
        self.assertEqual(result['response_length_prompt_mean'],3.)
        self.assertEqual(result['response_length_std_population'],1.)
        self.assertEqual(result['reference_kl_token_mean'],1.)
        self.assertEqual(result['truncation_rate'],.5)
    def test_reward_bridge_changes_only_theta_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json'
            p.write_text(json.dumps({'model_type':'qwen2','architectures':['Qwen2ForSequenceClassification'],
                'hidden_size':16,'num_attention_heads':2,'num_key_value_heads':1,'num_hidden_layers':1,'intermediate_size':32,
                'num_labels':1,'rope_parameters':{'rope_type':'default','rope_theta':1000000.}}))
            before=p.read_bytes(); effective,audit=compatible_reward_config(p)
            self.assertEqual(p.read_bytes(),before); self.assertEqual(set(audit['changed_fields']),{'rope_theta'})
            self.assertEqual(effective.rope_theta,1000000.)
    def test_reward_bridge_conflict_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json'; p.write_text(json.dumps({'model_type':'qwen2','architectures':['Qwen2ForSequenceClassification'],
                'rope_theta':10000.,'rope_parameters':{'rope_type':'default','rope_theta':1000000.}}))
            with self.assertRaises(ValueError): compatible_reward_config(p)


class CacheTests(unittest.TestCase):
    def make_cache(self,p,constant=False):
        rows=[{'prompt_id':f'p{i}','source_index':i,'generation_index':j,'reward':float(i if constant else i+j),
               'clipped_at_max':j==7} for i in range(24) for j in range(8)]
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    def test_conserved_generations_and_frozen_bins(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cache.jsonl'; self.make_cache(p); result=analyze(p)
            self.assertEqual([r['overall']['groups'] for r in result['conditions']],[96,48,24])
            self.assertTrue(all(r['overall']['generations']==192 for r in result['conditions']))
            for k in (2,4,8):
                rows=[i for m in result['membership'] if m['k']==k for i in m['cache_row_indices']]
                self.assertEqual(sorted(rows),list(range(192)))
    def test_duplicate_cache_indices_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cache.jsonl'; self.make_cache(p); rows=p.read_text().splitlines(); rows[-1]=rows[-2]
            p.write_text('\n'.join(rows))
            with self.assertRaises(ValueError): load_k8_cache(p)
    def test_constant_groups_zero_signal(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cache.jsonl'; self.make_cache(p,True); result=analyze(p)
            self.assertTrue(all(r['overall']['relative_signal_variance_population']==0. for r in result['conditions']))


class JobRecoveryTests(unittest.TestCase):
    def test_complete_synthetic_export_audits_800_evaluations(self):
        from task3_grpo.report import build_report
        from task3_grpo.storage import atomic_json,sha256
        cfg=load_yaml('configs/grpo.yaml')
        train=[{'prompt_id':'train','source_index':-1,'prompt':'synthetic'}]
        entry={'prompt_id':'train','source_index':-1,'row_index':0,'completion_seeds':[1,2,3,4]}
        rows=[{'prompt_id':f'e{i}','source_index':i,'prompt':'synthetic eval'} for i in range(200)]
        schedule={'training':[entry], 'evaluation':[{'row_index':i,'prompt_id':r['prompt_id'],
            'source_index':i,'generation_seed':i} for i,r in enumerate(rows)]}
        specs=[{'name':name,'updates':1,'loss_type':kind} for name,kind in
               [('standard','grpo'),('canonical_8','grpo'),('dr_grpo_8','dr_grpo')]]
        contract={'source':{},'approval':{'required_runs':specs},'assets':{'verified_files':{}}}
        def loader(*a,**kw):
            model=tiny_model(); return model,None,optimizer(model),{'synthetic':True}
        def evaluation(policy,tokenizer,reward,row,entry,cfg):
            return {'schedule':entry,'prompt_messages':[{'role':'user','content':'synthetic'}],
                'response':'synthetic','response_length':2,'raw_reward':1.,'reference_kl':0.,
                'sampled_entropy':2.,'categorical_entropy':1.,'terminated_with_eos':True,'truncated':False}
        with tempfile.TemporaryDirectory() as d,contextlib.ExitStack() as stack:
            root=Path(d); drive=root/'drive'; local=root/'local'
            stack.enter_context(patch('task3_grpo.jobs.load_policy',side_effect=loader))
            stack.enter_context(patch('task3_grpo.jobs.generate_group',side_effect=lambda *a:batch()))
            stack.enter_context(patch('task3_grpo.jobs.score_reward_pairs',side_effect=lambda *a,**k:torch.arange(4.)))
            stack.enter_context(patch('task3_grpo.jobs.evaluate_prompt',side_effect=evaluation))
            for name in ('synchronize','reset_peak_memory_stats','empty_cache'):
                stack.enter_context(patch('torch.cuda.'+name))
            for name in ('max_memory_allocated','max_memory_reserved'):
                stack.enter_context(patch('torch.cuda.'+name,return_value=0))
            run_probe(cfg,train,schedule,{},contract,None,drive,local,UnlimitedTestBudget(),False)
            for spec in specs: run_training(spec,cfg,train,schedule,{},contract,None,{},drive,local,UnlimitedTestBudget(),False)
            for name in ['midpoint']+[s['name'] for s in specs]:
                evaluate_condition(name,cfg,rows,schedule,{},contract,None,drive,local,UnlimitedTestBudget(),False)
            atomic_json(drive/'schedule.json',schedule)
            atomic_json(drive/'group_size.json',{'difficulty_definition':{}})
            atomic_json(drive/'PREPARATION.json',{'source':{},'approval':{'required_runs':specs},
                'group_size_sha256':sha256(drive/'group_size.json')})
            result=build_report(drive)
            self.assertEqual(sum(r['prompts'] for r in result['evaluation'].values()),800)
            self.assertTrue((drive/'task3_small_results.zip').is_file())

    def test_interrupted_tiny_job_resumes_exactly_and_final_rerun_is_idempotent(self):
        cfg=load_yaml('configs/grpo.yaml'); train=[{'prompt_id':'p0','source_index':0,'prompt':'test'}]
        entry={'prompt_id':'p0','source_index':0,'row_index':0,'completion_seeds':[1,2,3,4]}
        schedule={'training':[entry,entry]}; spec={'name':'synthetic','updates':2,'loss_type':'grpo'}
        def loader(*a,**kw):
            model=tiny_model(); return model,None,optimizer(model),{'synthetic':True}
        class Boundary:
            calls=0
            def check(self):
                self.calls+=1
                if self.calls==3: raise SessionBudgetReached()
        with tempfile.TemporaryDirectory() as d,contextlib.ExitStack() as stack:
            root=Path(d)
            stack.enter_context(patch('task3_grpo.jobs.load_policy',side_effect=loader))
            stack.enter_context(patch('task3_grpo.jobs.generate_group',side_effect=lambda *a:batch()))
            stack.enter_context(patch('task3_grpo.jobs.score_reward_pairs',side_effect=lambda *a,**k:torch.arange(4.)))
            for name in ('synchronize','reset_peak_memory_stats','empty_cache'):
                stack.enter_context(patch('torch.cuda.'+name))
            for name in ('max_memory_allocated','max_memory_reserved'):
                stack.enter_context(patch('torch.cuda.'+name,return_value=0))
            with self.assertRaises(SessionBudgetReached):
                run_training(spec,cfg,train,schedule,{}, {},None,{},root/'drive',root/'local',Boundary(),False)
            self.assertTrue((root/'drive/training/synthetic/step_0001.sha256.json').exists())
            # New local runtime exercises archive recovery, not just in-place resume.
            result=run_training(spec,cfg,train,schedule,{}, {},None,{},root/'drive',root/'newlocal',UnlimitedTestBudget(),True)
            self.assertEqual(result['completed_updates'],2)
            full=run_training(spec,cfg,train,schedule,{}, {},None,{},root/'full_drive',root/'full_local',UnlimitedTestBudget(),False)
            resumed_state=torch.load(root/'newlocal/training/synthetic/step_0002/checkpoint.pt',weights_only=False)
            full_state=torch.load(root/'full_local/training/synthetic/step_0002/checkpoint.pt',weights_only=False)
            for k,v in resumed_state['policy'].items(): torch.testing.assert_close(v,full_state['policy'][k],rtol=0,atol=0)
            self.assertEqual(result,run_training(spec,cfg,train,schedule,{}, {},None,{},root/'drive',root/'newlocal',UnlimitedTestBudget(),True))
            with self.assertRaises(FileExistsError):
                run_training(spec,cfg,train,schedule,{}, {},None,{},root/'drive',root/'newlocal',UnlimitedTestBudget(),False)
