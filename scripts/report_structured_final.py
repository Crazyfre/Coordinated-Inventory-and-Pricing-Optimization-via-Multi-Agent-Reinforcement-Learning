"""Publish the verified final matrix without selecting seeds or retraining."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.train_compact_residual import digest


def report(output):
    out=Path(output);load=lambda name:json.loads((out/name).read_text())
    result=load('summary.json');verified=load('verification.json');manifest=load('frozen_manifest.json')
    assert verified['status']=='passed' and result['status']=='complete'
    assert verified['primary_gate']==result['primary_gate']
    assert digest(out/'frozen_manifest.json')==result['frozen_manifest_sha256']
    for name,h in result['trace_hashes'].items():assert digest(out/name)==h
    primary=result['analysis']['S2'];seeds=manifest['seeds']
    outcome='PASS' if result['primary_gate'] else 'FAIL'
    lines=['# Final structured policy-gradient study', '',f'Primary held-out S2 gate: **{outcome}**.', '',
        'This is an evaluation within the frozen calibrated simulator, not validation of real-world causal demand. '
        'The method is Gaussian parameter-space score-function policy search, not PPO/MAPPO, a neural representation, or a novel algorithm. '
        'The simulator, delivered-unit billing and reward are unchanged.', '',
        'Ten final training seeds are included without best-seed selection. Five main checkpoints came from the earlier confirmation; '
        'five main, ten pooled-cover and ten independent-noise policies were trained under the final protocol. '
        'Each policy used 64 updates and 1,024 training episodes. The test matrix uses 50 common scenario keys per condition, '
        '30 learned policies and four baselines, for 17,000 scored episodes and 40 exact replays.', '',
        '## Primary S2 result', '',
        '| Endpoint | Main minus full7.5 | 97.5% crossed interval |', '|---|---:|---:|']
    for metric in ['reward_day','fill']:
        e=primary['primary'][metric]
        lines.append(f"| {metric} | {e['mean']:+.6f} | [{e['interval'][0]:+.6f}, {e['interval'][1]:+.6f}] |")
    lines+=['',f"All ten seed means improve reward: **{primary['every_seed_reward_positive']}**. "
        'The predeclared gate requires positive reward lower bound, fill lower bound at least -0.005, and every seed positive. '
        'This is distinct from reaching the absolute 0.95 service target.', '',
        '| Seed | Main reward/day | Full7.5 reward/day | Difference | Main fill |', '|---|---:|---:|---:|---:|']
    reference=primary['means']['full7_5']
    for seed in seeds:
        m=primary['means'][f'main_{seed}']
        lines.append(f"| {seed} | {m['reward_day']:.6f} | {reference['reward_day']:.6f} | {m['reward_day']-reference['reward_day']:+.6f} | {m['fill']:.6f} |")
    lines+=['','## Matched component comparisons','','| Contrast | Difference | 98.75% crossed interval |','|---|---:|---:|']
    for name,e in primary['secondary'].items():
        lines.append(f"| main minus {name} | {e['mean']:+.6f} | [{e['interval'][0]:+.6f}, {e['interval'][1]:+.6f}] |")
    lines+=['','Pooled cover changes parameter sharing and capacity. Independent-noise pairs change exogenous-noise coupling. '
        'A component benefit requires its own interval evidence; passing the main gate does not establish either benefit.','',
        '## Scenario robustness (exploratory)','','| Condition | Reward gain/day | 95% interval | Fill gain | 95% interval |','|---|---:|---:|---:|---:|']
    names=list(result['analysis'])
    for name in names:
        a=result['analysis'][name]['exploratory'];r=a['full7_5_reward_day'];f=a['full7_5_fill']
        lines.append(f"| {name} | {r['mean']:+.6f} | [{r['interval'][0]:+.6f}, {r['interval'][1]:+.6f}] | {f['mean']:+.6f} | [{f['interval'][0]:+.6f}, {f['interval'][1]:+.6f}] |")
    lines+=['','S2, S6 and S4_e15 use duplicate factory settings and are not independent confirmations. '
        'Other baselines and conditions are descriptive/exploratory; no universal dominance claim follows.','',
        '## Service and operational tradeoffs','','| Main seed | Episodes with fill >=0.95 | Wilson 95% interval | Return p05 |','|---|---:|---:|---:|']
    for seed in seeds:
        d=primary['distributions'][f'main_{seed}'];s=d['success']
        lines.append(f"| {seed} | {s['successes']}/50 | [{s['ci95'][0]:.6f}, {s['ci95'][1]:.6f}] | {d['episodic_return']['p05']:.6f} |")
    rows=load('S2_metrics.json')['rows'];main=[r for r in rows if r['label'].startswith('main_')];baseline=[r for r in rows if r['label']=='full7_5']
    lines+=['','| Operational mean per episode | Main (all seeds) | Full7.5 | Difference |','|---|---:|---:|---:|']
    for metric in ['lost_sales','mean_inventory','stockout_sku_day_fraction','clipped_units','delivered_units','revenue','ordering_cost','holding_cost','stockout_cost','price_change_cost']:
        a=np.mean([r[metric] for r in main]);b=np.mean([r[metric] for r in baseline])
        lines.append(f'| {metric} | {a:.6f} | {b:.6f} | {a-b:+.6f} |')
    lines+=['','These operational means are descriptive and do not imply every metric improves.','',
        '## Verification, compute and remaining limitations','',
        f"Evaluation: {result['counters']['episodes']} episodes, {result['counters']['days']} simulator days, {result['seconds']:.3f} seconds; zero training updates. "
        f"Archive verifier checked {verified['checked_episodes']} scored episodes and {verified['exact_replays']} exact replays without new simulator days. "
        f"Heuristic decisions ambiguous due to archived float32 rounding: {verified['heuristic_rounding_ambiguous_sku_decisions']}.", '',
        'Training recovery retained 585 complete episodes; none were repeated. Original first-seed total runtime was lost at interruption, '
        'so compliance with its original runtime cap is not claimed. The recovered cohort contains 25,600 canonical training episodes, '
        'of which 25,015 were newly simulated after recovery. The earlier five main fits add 5,120 episodes.', '',
        'Calibration predictive acceptance remains failed (9/10 SKUs within the factor-two band). '
        'The separate matched joint pricing/ordering pilot was negative; this ordering result does not demonstrate coordination. '
        'MAPPO remains a historical comparison with differing budgets/calibration/action scope, not a matched final superiority claim. '
        'The repeated use of the validation calendar during development is retained as a limitation. '
        'No retraining or checkpoint selection is permitted after this final result.', '',
        '![Final primary and seed effects](final_primary.png)', '', '![Exploratory scenario effects](final_robustness.png)', '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric,title in zip(axes,['reward_day','fill'],['Reward gain per day','Fill-rate gain']):
        e=primary['primary'][metric];ax.bar([str(s) for s in seeds],e['seed_means'],color='#357c9b')
        ax.axhline(0,color='black',lw=.8);ax.axhline(e['mean'],color='#b24b35',ls='--',label='Across-seed mean')
        ax.axhspan(*e['interval'],alpha=.12,color='#b24b35',label='97.5% crossed interval')
        ax.set_title(title+' vs full7.5');ax.set_xlabel('Training seed');ax.tick_params(axis='x',rotation=45)
    axes[0].legend(fontsize=8);fig.suptitle('Held-out S2: all ten final policies');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'final_primary.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,metric,title in zip(axes,['reward_day','fill'],['Reward gain per day','Fill-rate gain']):
        values=[result['analysis'][n]['exploratory'][f'full7_5_{metric}'] for n in names]
        means=np.array([e['mean'] for e in values]);bounds=np.array([e['interval'] for e in values])
        ax.errorbar(means,np.arange(len(names)),xerr=np.array([means-bounds[:,0],bounds[:,1]-means]),fmt='o',color='#357c9b')
        ax.set_yticks(np.arange(len(names)),names);ax.axvline(0,color='black',lw=.8);ax.set_title(title+' vs full7.5')
    fig.suptitle('Exploratory 95% crossed intervals; duplicate settings retained');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'final_robustness.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);report(p.parse_args().output)
