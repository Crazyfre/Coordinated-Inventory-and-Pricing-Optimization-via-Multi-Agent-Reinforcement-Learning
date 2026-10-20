"""Standalone research figure from frozen confirmation summary; no simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(output):
    out = Path(output); summary = json.loads((out/'summary.json').read_text())
    endpoints = summary['analysis']['endpoints']; seeds = [211, 222, 233, 244, 255]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), constrained_layout=True)
    for ax, metric, factor, xlabel in zip(axes, ['reward_day', 'fill'], [1, 100],
                                         ['Reward difference per day', 'Fill difference (percentage points)']):
        e = endpoints['full7_5_'+metric]
        means = np.array(e['seed_means'])*factor
        ci = np.array(e['conditional_seed_ci95'])*factor
        ax.errorbar(means, np.arange(5), xerr=np.array([means-ci[:, 0], ci[:, 1]-means]),
                    fmt='o', color='#2c6c91', capsize=3, label='Seed estimate; conditional 95% interval')
        value = e['mean']*factor; lower, upper = np.array(e['ci9875'])*factor
        ax.errorbar([value], [6], xerr=[[value-lower], [upper-value]], fmt='D', color='#bb5a27',
                    capsize=5, label='Aggregate; crossed 98.75% interval')
        ax.axvline(0, color='#555555', linewidth=1)
        if metric == 'fill':
            ax.axvline(-.5, color='#aa3333', linestyle='--', linewidth=1, label='Service safeguard (-0.5 pp)')
        ax.set_yticks([0, 1, 2, 3, 4, 6], [str(s) for s in seeds]+['Aggregate'])
        ax.invert_yaxis(); ax.set_xlabel(xlabel); ax.grid(axis='x', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.16))
    axes[0].set_ylabel('Training seed')
    fig.suptitle('Parameter-space policy-gradient RL versus full7.5 starting policy', fontsize=12)
    fig.savefig(out/'confirmation_effects.png', dpi=180)
    fig.savefig(out/'confirmation_effects.pdf')
    plt.close(fig)
    print('Saved confirmation_effects.png and confirmation_effects.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); plot(p.parse_args().output)
