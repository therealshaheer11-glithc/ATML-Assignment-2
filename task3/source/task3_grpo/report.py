"""Recompute results from verified saved records; no inference or selection of endpoints."""
import argparse
import csv
import hashlib
import io
import json
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from task3_grpo.storage import atomic_json, sha256
from task3_grpo.preflight import require, canonical


def read_unit(archive,name):
    archive = Path(archive)
    receipt = json.loads(archive.with_suffix('.sha256.json').read_text())
    require(receipt['archive']==archive.name and sha256(archive)==receipt['sha256'],'Archive identity differs')
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('FILES_SHA256.json'))
        require(len(z.namelist())==len(manifest)+1 and set(z.namelist())==set(manifest)|{'FILES_SHA256.json'},'Archive file set differs')
        value = z.read(name)
        require(hashlib.sha256(value).hexdigest()==manifest[name],'Record digest differs')
        return json.loads(value)


def gradient_summary(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row['length_bin'],row['advantage_sign'])].append(row['policy_logprob_gradient_l2'])
    return [{'length_bin':length,'advantage_sign':sign,'count':len(groups[(length,sign)]),
        'gradient_l2_mean':statistics.mean(groups[(length,sign)]) if groups[(length,sign)] else None}
        for length in ('1-128','129-256','257-512','masked') for sign in ('positive','negative','zero')]


def qualitative_examples(evaluations,cache):
    # Rules are fixed in source before actual endpoint outcomes exist.
    a,b = evaluations['canonical_8'],evaluations['dr_grpo_8']
    paired = []
    for rule,key in [('largest_absolute_reward_difference','raw_reward'),('largest_absolute_length_difference','response_length')]:
        ranked = sorted(range(200),key=lambda i:(-abs(a[i][key]-b[i][key]),a[i]['schedule']['prompt_id']))[:3]
        for i in ranked:
            paired.append({'selection_rule':rule,'prompt_id':a[i]['schedule']['prompt_id'],
                'prompt_messages':a[i]['prompt_messages'],
                'canonical':{k:a[i][k] for k in ('response','raw_reward','response_length','reference_kl','truncated')},
                'constant_denominator':{k:b[i][k] for k in ('response','raw_reward','response_length','reference_kl','truncated')},
                'review_status':'PENDING_USER_REVIEW','assistant_interpretation':None,
                'caution':'Selected examples are illustrative; reward difference does not establish answer quality.'})
    return {'selection_before_outcomes':True,'paired_normalization':paired,
        'cache_difficulty_mapping':cache['difficulty_definition'],
        'human_quality_labels_created':False,
        'review_instructions':'Read both exact outputs, assess correctness/helpfulness/repetition, and record disagreements with reward or length. Empty interpretations are deliberate pending review.'}


def build_report(experiment):
    from task3_grpo.jobs import summarize_evaluation
    experiment = Path(experiment)
    preparation = json.loads((experiment/'PREPARATION.json').read_text())
    schedule = json.loads((experiment/'schedule.json').read_text())
    out = experiment/'results'
    out.mkdir(exist_ok=True)
    cache_path = experiment/'group_size.json'
    require(sha256(cache_path)==preparation['group_size_sha256'],'Cache diagnostic receipt differs')
    cache = json.loads(cache_path.read_text())
    summaries,trajectories,evaluations = {},{},{}
    required = preparation['approval']['required_runs']
    for spec in required:
        folder = experiment/'training'/spec['name']
        summary = read_unit(folder/'final.zip','summary.json')
        require(summary['completed_updates']==spec['updates'] and summary['status']=='COMPLETE','Continuation incomplete')
        records = [read_unit(folder/f'step_{i:04d}.zip','update.json') for i in range(1,spec['updates']+1)]
        require(summary['history']==records,'Final summary/update history differs')
        require([r['schedule'] for r in records]==schedule['training'][:spec['updates']], 'Training schedules differ')
        require(all(r['contract']==summary['contract'] for r in records),'Update contracts differ')
        require(summary['contract']['source']==preparation['source'] and summary['contract']['run']==spec,'Run source/spec differs')
        summaries[spec['name']] = summary
        trajectories[spec['name']] = records
    names = ['midpoint']+[s['name'] for s in required]
    aggregate = {}
    for name in names:
        folder = experiment/'evaluation'/name
        saved = json.loads((folder/'summary.json').read_text())
        records = [read_unit(folder/f'prompt_{i:04d}.zip','record.json') for i in range(200)]
        require([r['schedule'] for r in records]==schedule['evaluation'],'Evaluation prompt/seed alignment differs')
        require(all(r['contract']==saved['contract'] and r['condition']==name for r in records),'Evaluation contracts differ')
        metrics = summarize_evaluation(records)
        require(metrics==saved['metrics'] and saved['status']=='COMPLETE','Evaluation aggregation differs')
        evaluations[name] = records
        aggregate[name] = metrics
        with (out/(name+'_evaluation.jsonl')).open('w') as f:
            for row in records:
                f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    gradients = {name:gradient_summary([g for r in records for g in r['optimization']['length_gradient_statistics']])
                 for name,records in trajectories.items()}
    probe = read_unit(experiment/'normalization_probe/probe.zip','record.json')
    require(probe['optimizer_updates']==0 and probe['excluded_from_continuation_budget'],'Probe accounting differs')
    require(probe['contract']['source']==preparation['source'],'Probe source differs')
    result = {'status':'COMPLETE_ACTUAL_RUNS','preparation_source':preparation['source'],
        'preparation_receipt':preparation,
        'execution_environments':[json.loads(p.read_text()) for p in sorted(experiment.glob('gpu_session_*.json'))],
        'training':summaries,'evaluation':aggregate,'length_gradient_statistics':gradients,
        'common_probe':probe,'cache_diagnostic':cache,'final_adapter_for_task4':'training/standard/final.zip::policy/',
        'qualitative_review':'PENDING_USER_REVIEW','checkpoint_selection_performed':False}
    atomic_json(out/'results.json',result)
    atomic_json(out/'schedule.json',schedule)
    atomic_json(out/'qualitative_review.json',qualitative_examples(evaluations,cache))
    with (out/'length_gradient_statistics.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['condition','length_bin','advantage_sign','count','gradient_l2_mean'])
        writer.writeheader()
        for name,records in gradients.items():
            for record in records:
                writer.writerow(dict(condition=name,**record))
    with (out/'evaluation.csv').open('w',newline='') as f:
        fields = ['condition','prompts','raw_reward_prompt_mean','reference_kl_token_mean','response_length_prompt_mean',
                  'response_length_std_population','categorical_entropy_token_mean','sampled_entropy_token_mean','truncation_rate']
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for name in names:
            writer.writerow({'condition':name,**{k:aggregate[name][k] for k in fields[1:]}})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,key,label in zip(axes.flat,('raw_reward','reference_kl','response_length_mean','group_reward_std'),
                            ('Rollout raw reward','Signed reference KL','Mean response tokens','Within-group reward std')):
        for name,records in trajectories.items():
            ax.plot(range(1,len(records)+1),[r['rollout'][key] for r in records],label=name)
        ax.set(xlabel='Completed update',ylabel=label)
        ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.savefig(out/'training_trajectories.png',dpi=180)
    plt.close(fig)
    (out/'README.md').write_text('Actual saved Task 3 runs. Results recomputed from hash-verified units.\n\n'
        'The final standard 20-update adapter is in ../training/standard/final.zip, policy/. '
        'Forks have equal allocated caps, not necessarily equal generated tokens. '
        'Wall-clock completeness is reported in each run. Qualitative review remains pending.\n')
    manifest = {p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='FILES_SHA256.json'}
    atomic_json(out/'FILES_SHA256.json',manifest)
    target = experiment/'task3_small_results.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file():
                z.write(p,'results/'+p.name)
    atomic_json(experiment/'task3_small_results.sha256.json',{'archive':target.name,'sha256':sha256(target),'bytes':target.stat().st_size})
    print('All continuation/evaluation counts audited; small result export saved. Qualitative review pending.',flush=True)
    return result


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment',required=True)
    build_report(ap.parse_args().experiment)
