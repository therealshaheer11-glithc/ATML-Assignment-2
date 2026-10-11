"""Rebuild publication tables and a complete diagnostic figure from saved scalars.

No inference, new labels, hyperparameter selection, or changes to original evidence.
CSV/Markdown need only Python's standard library; --plot also needs matplotlib.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'results/actual_run_20261011'


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def build(output, plot=False):
    output = Path(output).resolve()
    if output.is_relative_to(EVIDENCE):
        raise RuntimeError('Keep derived presentation separate from original evidence.')
    from audit_saved_export import audit
    audit(EVIDENCE, output / 'audit')
    result = json.loads((EVIDENCE / 'results.json').read_text())
    training = []
    for name, run in result['training'].items():
        for step in run['history']:
            row = {'condition': name, 'update': step['completed_updates']}
            row.update({'rollout_' + k: v for k, v in step['rollout'].items()})
            row.update({'optimization_' + k: v for k, v in step['optimization'].items()
                        if k != 'length_gradient_statistics'})
            training.append(row)
    write_csv(output / 'training_diagnostics.csv', training)
    cache = []
    for condition in result['cache_diagnostic']['conditions']:
        for name, values in {'overall': condition['overall'], **condition['difficulty_bins']}.items():
            cache.append({'K': condition['k'], 'bin': name, **values})
    write_csv(output / 'group_size_all_bins.csv', cache)
    rows = ['# Complete group-size evidence', '',
            'Difficulty proxy: mean reward across all eight completions per prompt; '
            'at or below the frozen median (2.19873046875) is harder, above it is easier. '
            'Each bin has 12 prompts and 96 generations per K. The overall budget is 192.', '',
            '| K | Bin | Groups | Informative % | Mean group reward std | Relative-signal variance |',
            '|---:|---|---:|---:|---:|---:|']
    for r in cache:
        rows.append(f"| {r['K']} | {r['bin']} | {r['groups']} | {100*r['informative_group_rate']:.2f} | "
                    f"{r['mean_within_group_reward_std']:.6f} | {r['relative_signal_variance_population']:.6f} |")
    rows += ['', 'This is one fixed partition per K, not a repeated-partition uncertainty estimate. '
             'The reward-derived bins are proxies, not independent difficulty labels. '
             'Relative-signal variance partly follows informative-group rate by construction.', '']
    (output / 'GROUP_SIZE.md').write_text('\n'.join(rows))
    scores = json.loads((EVIDENCE / 'review/SCORES.json').read_text())
    selected = json.loads((EVIDENCE / 'qualitative_review.json').read_text())['paired_normalization']
    rows = ['# Exact reviewed examples', '',
            'These five unique prompts occupy six frozen extreme-difference selection slots. '
            'They are illustrative, not a representative quality sample. A is canonical; B uses '
            'the constant denominator. Scores and reasons are in the approved score table. '
            'Prompt and response text below is reproduced unchanged.', '']
    for example in scores['examples']:
        pair = next(p for p in selected if p['prompt_id'] == example['prompt_id'])
        rows += [f"## {example['example_number']}. {example['title']}", '',
                 f"Prompt ID: `{example['prompt_id']}`", '', '### Original request', '']
        for message in pair['prompt_messages']:
            rows += [f"Role: {message['role']}", '', '~~~~~~text', message['content'], '~~~~~~', '']
        for letter, key in [('A', 'canonical'), ('B', 'constant_denominator')]:
            response = pair[key].get('response', pair[key].get('response_text'))
            if not isinstance(response, str):
                raise RuntimeError('Response field missing from original selection.')
            answer = example['answers'][letter]
            rows += [f"### Answer {letter} — {answer['total']}/6", '', '~~~~~~text', response, '~~~~~~', '']
    (output / 'EXACT_REVIEWED_EXAMPLES.md').write_text('\n'.join(rows))
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fields = [
            ('rollout_raw_reward', 'Mean raw reward'),
            ('rollout_reference_kl', 'Signed sampled KL / valid token'),
            ('rollout_group_reward_std', 'Mean within-group reward std'),
            ('rollout_uninformative_group_fraction', 'Uninformative group fraction'),
            ('optimization_policy_term', 'Policy surrogate loss'),
            ('optimization_gradient_norm_preclip', 'Parameter gradient norm (before clipping)'),
            ('rollout_categorical_entropy', 'Categorical entropy / valid token'),
            ('rollout_response_length_mean', 'Mean response length (tokens)'),
        ]
        fig, axes = plt.subplots(4, 2, figsize=(11, 13), layout='constrained')
        for ax, (key, title) in zip(axes.flat, fields):
            for name, color, marker in [('standard', '#2563a6', 'o'), ('canonical_8', '#d97706', 's'),
                                         ('dr_grpo_8', '#15803d', '^')]:
                selected_rows = [r for r in training if r['condition'] == name]
                ax.plot([r['update'] for r in selected_rows], [r[key] for r in selected_rows],
                        label=name, color=color, marker=marker, markersize=3, linewidth=1.2,
                        linestyle='--' if name == 'canonical_8' else '-')
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Completed update')
            ax.grid(alpha=.22)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4), useOffset=False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle('Task 3: saved continuation diagnostics\nCanonical updates 1–8 match standard exactly; all-masked updates are retained.', fontsize=12)
        fig.savefig(output / 'complete_training_diagnostics.png', dpi=160)
        fig.savefig(output / 'complete_training_diagnostics.pdf', metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)
    print('PRESENTATION REBUILT: 36 training rows, 9 group-size rows, 5 exact reviewed examples.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()
    build(args.output, args.plot)
