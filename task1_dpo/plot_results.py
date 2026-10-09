"""Render the two user-approved Task 1 figures from verified saved CSVs.

CPU-only postprocessing. No model loading, experiment changes, smoothing,
confidence intervals, or significance tests. Requires Matplotlib.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'atml-task1-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, INK = '#28649A', '#C46B32', '#233142'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_inputs(repo):
    manifest = json.loads((repo / 'PACKAGE_SHA256.json').read_text())
    sources, tables = {}, {}
    for name in ('summary.csv', 'length_strata.csv', 'word_limits.csv', 'audit.json'):
        relative = 'results/task1/analysis/' + name
        source = repo / relative
        require(sha(source) == manifest[relative], f'Published input changed: {relative}')
        sources[relative] = sha(source)
        if name.endswith('.csv'):
            with source.open(newline='') as stream:
                tables[name] = list(csv.DictReader(stream))
        else:
            require(json.loads(source.read_text())['status'] == 'PASS', 'Saved evidence audit did not pass')
    return sources, tables


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#9AA7B5')
    ax.tick_params(colors=INK, length=3)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E4E9EF', linewidth=.65)


def save(fig, output, name):
    fig.savefig(output / (name + '.png'), dpi=300, facecolor='white')
    fig.savefig(output / (name + '.pdf'), facecolor='white',
                metadata={'Title': name.replace('_', ' '), 'Author': 'ATML PA2 Task 1',
                          'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def render(repo, output):
    sources, tables = load_inputs(repo)
    require(not output.exists(), 'Choose a new figure output folder; existing artifacts are preserved.')
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.labelsize': 9, 'text.color': INK,
                         'axes.labelcolor': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    summary = {r['condition']: r for r in tables['summary.csv']}
    names = ('beta_003', 'beta_010', 'beta_030')
    beta = [summary[n] for n in names]
    require([float(r['beta']) for r in beta] == [.03, .1, .3], 'Unexpected beta protocol')
    require(all(int(r['training_pairs']) == 600 for r in beta), 'Beta budgets differ')
    plotted = {'beta': [], 'strata': [], 'word_compliance': []}
    for name, r in zip(names, beta):
        plotted['beta'].append({'condition': name, 'beta': float(r['beta']), 'training_pairs': 600,
                                **{k: float(r[k]) for k in ('accuracy', 'sampled_kl', 'reward_mean', 'token_mean')}})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3))
    fig.subplots_adjust(left=.10, right=.975, top=.85, bottom=.13, wspace=.36, hspace=.58)
    fig.suptitle('DPO beta study', x=.10, y=.975, ha='left', fontsize=15, weight='bold')
    fig.text(.10, .921, 'Matched 600-pair runs · 300 held-out prompts · seed 6304', fontsize=9, color='#536579')
    specs = [
        ('accuracy', 100, '(a) Preference accuracy', 'Accuracy (%)', (0, 100), lambda v: f'{v:.2f}%'),
        ('sampled_kl', 10000, '(b) Signed sampled KL', 'Log-probability difference (×10⁻⁴)', (-5.6, .5), lambda v: f'{v:.3f}'),
        ('reward_mean', 1, '(c) Reward-model score', 'Mean reward', (0, 1.35), lambda v: f'{v:.3f}'),
        ('token_mean', 1, '(d) Generated response length', 'Mean response tokens', (0, 275), lambda v: f'{v:.2f}'),
    ]
    for ax, (key, factor, title, ylabel, limits, label) in zip(axes.flat, specs):
        values = [float(r[key]) * factor for r in beta]
        style(ax)
        ax.plot(range(3), values, color=BLUE, marker='o', linewidth=1.7, markersize=5)
        ax.set(xlim=(-.3, 2.3), ylim=limits, xticks=range(3), xticklabels=('0.03', '0.10', '0.30'),
               xlabel='Beta (β)', ylabel=ylabel, title=title)
        ax.title.set_horizontalalignment('left'); ax.title.set_position((0, 1))
        for x, value in enumerate(values):
            ax.annotate(label(value), (x, value), xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=9, weight='medium')
        if key == 'sampled_kl':
            ax.axhline(0, color='#697B8C', linewidth=.8, linestyle='--')
        if key == 'token_mean':
            ax.axhline(256, color='#697B8C', linewidth=.8, linestyle='--')
            ax.text(.04, 259, '256-token generation ceiling', transform=ax.get_yaxis_transform(), fontsize=7.5, va='bottom')
    fig.text(.10, .025, 'Points are single-run means; lines only guide the eye. Dispersion is reported in the results table.', fontsize=8)
    save(fig, output, 'beta_comparison')

    conditions = ('standard', 'length_balanced')
    strata = ('preferred_longer', 'length_matched', 'rejected_longer')
    indexed = {(r['condition'], r['stratum']): r for r in tables['length_strata.csv']}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.3), gridspec_kw={'width_ratios': (1.65, 1)})
    fig.subplots_adjust(left=.09, right=.98, top=.68, bottom=.24, wspace=.32)
    fig.suptitle('Length-confounding study', x=.09, y=.975, ha='left', fontsize=15, weight='bold')
    fig.text(.09, .887, 'Both models: 1,500 training pairs · β = 0.10 · seed 6304', fontsize=9, color='#536579')
    for index, (name, color, display) in enumerate(zip(conditions, (BLUE, ORANGE), ('Standard DPO', 'Length-balanced DPO'))):
        source = [indexed[(name, s)] for s in strata]
        require(all(int(r['pairs']) == 82 for r in source), 'Unexpected stratum size')
        values = [100*float(r['preference_accuracy']) for r in source]
        positions = [x + (-.19 if index == 0 else .19) for x in range(3)]
        bars = axes[0].bar(positions, values, width=.34, label=display, color=color,
                           hatch='///' if index else None, edgecolor='white', linewidth=.6)
        for bar, r in zip(bars, source):
            count = int(r['correct_pairs'])
            require(abs(count/82 - float(r['preference_accuracy'])) < 1e-12, 'Stratum numerator mismatch')
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2.5, f'{count}/82', ha='center', fontsize=8)
            plotted['strata'].append({'condition': name, 'stratum': r['stratum'], 'correct': count,
                                      'pairs': 82, 'accuracy': count/82})
        word = [r for r in tables['word_limits.csv'] if r['condition'] == name]
        require(len(word) == 10 and len({r['prompt_id'] for r in word}) == 10, 'Word pool membership')
        wins = sum(int(r['words']) <= int(r['limit']) for r in word)
        require(all((int(r['words']) <= int(r['limit'])) == (r['complies'] == 'True') for r in word), 'Word count rule mismatch')
        axes[1].bar(index, 10*wins, width=.55, color=color, hatch='///' if index else None,
                    edgecolor='white', linewidth=.6)
        axes[1].text(index, 10*wins+3, f'{wins}/10', ha='center', fontsize=10)
        plotted['word_compliance'].append({'condition': name, 'compliant': wins, 'prompts': 10, 'fraction': wins/10})
    axes[0].set(xticks=range(3), xticklabels=('Preferred\nlonger', 'Length\nmatched', 'Rejected\nlonger'),
                ylabel='Preference accuracy (%)', title='(a) Held-out length strata')
    axes[1].set(xticks=(0,1), xticklabels=('Standard', 'Balanced'), xlim=(-.6,1.6),
                ylabel='Compliant answers (%)', title='(b) Word-limit compliance')
    for ax in axes:
        style(ax); ax.set_ylim(0, 100)
        ax.title.set_horizontalalignment('left'); ax.title.set_position((0, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.075,.848), frameon=False, ncol=2, fontsize=9)
    fig.text(.09, .093, 'Bars show exact observed fractions; no significance or confidence-interval claim.', fontsize=8)
    fig.text(.09, .044, 'Strata: 82 pairs each. Compliance: the same 10 fixed prompts and the released word-count rule.', fontsize=8)
    save(fig, output, 'length_comparison')
    (output/'plotted_values.json').write_text(json.dumps(plotted, indent=2)+'\n')
    provenance = {'status': 'GENERATED_FROM_VERIFIED_SAVED_RESULTS', 'matplotlib': matplotlib.__version__,
                  'script_sha256': sha(__file__), 'source_sha256': sources,
                  'approach': 'Two user-approved figures; exact means/fractions, no smoothing or inferential error bars.',
                  'new_training': False, 'new_generation': False, 'GPU_used': False,
                  'output_sha256': {f.name: sha(f) for f in sorted(output.iterdir()) if f.is_file()}}
    (output/'figure_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('TASK1_FIGURES_READY:', output)
    print('Saved two figures as PDF and PNG, exact plotted values and source hashes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--repo', type=Path, default=root)
    parser.add_argument('--output', type=Path, default=root/'figures/task1')
    args = parser.parse_args()
    render(args.repo.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
