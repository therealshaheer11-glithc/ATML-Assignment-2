"""Approved equal-generation cache diagnostic; no training or rescoring."""
import argparse
import statistics
from collections import defaultdict
from pathlib import Path
import torch
from common.data import read_jsonl
from task3_grpo.grpo import group_relative_advantages
from task3_grpo.storage import atomic_json, sha256


def load_k8_cache(path):
    by_prompt = defaultdict(list)
    for i, row in enumerate(read_jsonl(path)):
        if not torch.isfinite(torch.tensor(row['reward'])):
            raise ValueError('Nonfinite reward')
        by_prompt[str(row['source_index'])].append(dict(row, cache_row_index=i))
    if len(by_prompt) != 24:
        raise ValueError('Pinned cache requires 24 prompts')
    for rows in by_prompt.values():
        rows.sort(key=lambda r: r['generation_index'])
        if [r['generation_index'] for r in rows] != list(range(8)) or len({r['prompt_id'] for r in rows}) != 1:
            raise ValueError('Eight unique indices 0-7 and one prompt per source required')
    return dict(sorted(by_prompt.items()))


def regroup_equal_generation_budget(by_prompt, k):
    if k not in (2,4,8):
        raise ValueError('Only approved K values allowed')
    groups = [rows[i:i+k] for rows in by_prompt.values() for i in range(0,8,k)]
    indices = [r['cache_row_index'] for g in groups for r in g]
    if len(indices) != 192 or len(set(indices)) != 192:
        raise RuntimeError('Equal-generation conservation failed')
    return groups


def difficulty_bins(by_prompt):
    means = {pid: statistics.mean(r['reward'] for r in rows) for pid,rows in by_prompt.items()}
    median = statistics.median(means.values())
    bins = {pid: 'harder_proxy' if m <= median else 'easier_proxy' for pid,m in means.items()}
    if len(set(bins.values())) != 2:
        raise RuntimeError('Approved bin rule gives an empty bin; seek approval before changing')
    return {'definition':'mean of eight rewards; <= median harder, > median easier; ties together',
            'median_threshold':median,'prompt_mean_rewards':means,'prompt_bins':bins}


def summarize_groups(groups):
    stds, advantages, centered = [], [], []
    for rows in groups:
        rewards = torch.tensor([r['reward'] for r in rows],dtype=torch.float32)
        stds.append(float(rewards.std(unbiased=False)))
        advantages.extend(group_relative_advantages(rewards,torch.zeros(len(rows),dtype=torch.long)).tolist())
        centered.extend((rewards-rewards.mean()).tolist())
    fraction = statistics.mean(float(s > 1e-6) for s in stds)
    return {'groups':len(groups),'generations':sum(map(len,groups)),
        'prompts':len({r['source_index'] for g in groups for r in g}),
        'informative_group_rate':fraction,'uninformative_group_fraction':1-fraction,
        'mean_within_group_reward_std':statistics.mean(stds),
        'relative_signal_variance_population':statistics.pvariance(advantages),
        'raw_centered_reward_variance_population':statistics.pvariance(centered),'tolerance':1e-6}


def analyze(path):
    by_prompt = load_k8_cache(path)
    bins = difficulty_bins(by_prompt)
    conditions, membership = [], []
    for k in (2,4,8):
        groups = regroup_equal_generation_budget(by_prompt,k)
        conditions.append({'k':k,'overall':summarize_groups(groups),'difficulty_bins':{
            b:summarize_groups([g for g in groups if bins['prompt_bins'][str(g[0]['source_index'])] == b])
            for b in ('harder_proxy','easier_proxy')}})
        membership.extend({'k':k,'group_index':i,'source_index':g[0]['source_index'],
            'prompt_id':g[0]['prompt_id'],'difficulty_bin':bins['prompt_bins'][str(g[0]['source_index'])],
            'cache_row_indices':[r['cache_row_index'] for r in g],
            'generation_indices':[r['generation_index'] for r in g],
            'truncated_rows':sum(bool(r['clipped_at_max']) for r in g)} for i,g in enumerate(groups))
    return {'cache_sha256':sha256(path),'generations_per_condition':192,'training_performed':False,
        'cache_rescored_or_regenerated':False,'difficulty_definition':bins,'conditions':conditions,
        'membership':membership,'cache_generation_cap':768,'training_generation_cap':512,
        'truncated_rows_included':True,
        'interpretation':'Standardized advantage variance can approximate informative-group rate by construction.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache',default='cached/grpo_k_cache.jsonl')
    ap.add_argument('--output',required=True)
    args = ap.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Analysis exists; retain it')
    result = analyze(args.cache)
    atomic_json(args.output,result)
    print('Cache study complete: 192 conserved generations for each K; no training.')


if __name__ == '__main__':
    main()
