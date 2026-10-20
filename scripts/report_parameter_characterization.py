"""Render the verified review supplement without fitting or simulating."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(output):
    out=Path(output);load=lambda name:json.loads((out/name).read_text())
    summary=load('summary.json');assert load('verification.json')['status']=='passed'
    archived=load('archived_metrics.json')['analysis'];seeds=[211,222,233,244,255]
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    for seed in seeds:
        data=summary['progress'][str(seed)]['evaluations']
        ax.plot([r['steps']/28 for r in data],[r['mean_evaluation_return'] for r in data],marker='o',label=f'Seed {seed}')
    ax.axhline(1,color='#555',linestyle='--',label='Descriptive threshold: +1/day')
    ax.set(xlabel='Training episodes per seed',ylabel='Evaluation reward/day gain versus own exact start',
           title='Archived mean policies on 32 fresh training scenarios')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.savefig(out/'learning_progress.png',dpi=180);fig.savefig(out/'learning_progress.pdf');plt.close(fig)
    names=list(summary['analysis']);fig,axes=plt.subplots(1,2,figsize=(11,6),layout='constrained')
    for ax,metric,factor,label in zip(axes,['reward_day','fill'],[1,100],['Reward/day difference','Fill difference (percentage points)']):
        effects=[summary['analysis'][n]['effects_vs_full7_5'][metric] for n in names]
        m=np.array([e['mean'] for e in effects])*factor;ci=np.array([e['ci95'] for e in effects])*factor
        ax.errorbar(m,np.arange(len(names)),xerr=[m-ci[:,0],ci[:,1]-m],fmt='o',capsize=3)
        ax.axvline(0,color='#555');ax.set_yticks(np.arange(len(names)),names);ax.invert_yaxis()
        ax.set_xlabel(label);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Frozen policies versus full7.5: exploratory 95% intervals')
    fig.savefig(out/'scenario_robustness.png',dpi=180);fig.savefig(out/'scenario_robustness.pdf');plt.close(fig)
    shortfall=archived['reference_shortfall'];fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    ax.plot(range(1,len(shortfall['keys'])+1),shortfall['cumulative']);ax.axhline(0,color='#555')
    ax.set(xlabel='Assessment scenarios in fixed key order',ylabel='Cumulative signed reference shortfall',
           title='Full7.5 minus learned return: not optimal or online regret')
    ax.grid(alpha=.2);fig.savefig(out/'reference_shortfall.png',dpi=180);plt.close(fig)
    lines=['# Frozen ordering RL: review metrics and robustness','',
        'This supplement characterizes the already verified five-seed method. No weights were fitted, selected or changed. '
        'The original confirmation remains the primary reward/service result; these new comparisons are descriptive development evidence. '
        'The held-out test remains unopened.','',
        '## Return distribution and task success','',
        'The following statistics use the original 128 paired confirmation scenarios. Return is the complete 28-day '
        'canonical economic reward, not revenue. Each seed is reported separately; the five policies on shared scenarios '
        'are not treated as 640 independent episodes. Success means fill >=95%, unchanged from the project target.','',
        '|Controller|Mean return|Median|SD|5th percentile|Successful episodes /128|Wilson 95% interval|',
        '|---|---:|---:|---:|---:|---:|---|']
    for label in ['full7_5']+[f'learned_{s}' for s in seeds]+['projected','constant','ss']:
        c=archived['controllers'][label];r=c['episodic_return'];success=c['success']
        lines.append(f"|{label}|{r['mean']:.3f}|{r['median']:.3f}|{r['std']:.3f}|{r['p05']:.3f}|{success['successes']}|"
                     f"[{100*success['ci95'][0]:.2f}%,{100*success['ci95'][1]:.2f}%]|")
    lines+=['','## Operational metrics','',
        'Costs, revenue, lost sales and clips below are per 28-day episode; inventory is mean end-of-day on-hand units. '
        'Receiving clips are capacity handling, not an infeasible on-hand state.','',
        '|Controller|Revenue|Procurement|Holding|Stockout cost|Lost units|Clipped units|Mean inventory|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in ['full7_5']+[f'learned_{s}' for s in seeds]:
        m=archived['controllers'][label]['means']
        lines.append('|'+label+'|'+'|'.join(f'{m[k]:.3f}' for k in ['revenue','ordering_cost','holding_cost','stockout_cost','lost_sales','clipped_units','mean_inventory'])+'|')
    lines+=['','The JSON additionally records delivered units, SKU-day stockout frequency, price-change costs and signed paired '
        'reference shortfall. Its cumulative curve follows arbitrary fixed scenario-key order and is not online-learning regret.','',
        '## Learning progress and interaction efficiency','',
        'All archived means at updates 0,16,32,48,64 were assessed on 32 new random training windows. No checkpoint was '
        'selected from this curve. The descriptive attainment rule is gain >=1 reward/day for three consecutive scheduled '
        'assessments. It does not establish asymptotic convergence or an optimized stopping time.','',
        '|Seed|Final gain/day|Threshold status|Onset training steps|Confirmed training steps|',
        '|---|---:|---|---:|---:|']
    for seed in seeds:
        p=summary['progress'][str(seed)];t=p['threshold']
        lines.append(f"|{seed}|{p['evaluations'][-1]['mean_evaluation_return']:.3f}|{t['status']}|{t['onset_step']}|{t['confirmed_step']}|")
    lines+=['','Each seed originally trained for 1024 episodes (28672 daily transitions). The learning-curve assessment adds '
        '672 evaluation episodes in total, with identical update0 executed once per scenario and explicitly shared between seeds.','',
        '## Scenario robustness','',
        'These 32-scenario comparisons reuse the validation calendar with fresh stochastic keys. Intervals resample '
        'five training seeds and 32 scenario columns. They are exploratory 95% intervals per condition; there is no '
        'familywise superiority claim or tuning from this matrix.','',
        '|Condition|Reward/day gain vs full7.5|95% interval|Fill gain (pp)|95% interval (pp)|',
        '|---|---:|---|---:|---|']
    for name in names:
        e=summary['analysis'][name]['effects_vs_full7_5'];r,f=e['reward_day'],e['fill']
        lines.append(f"|{name}|{r['mean']:+.3f}|[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]|{100*f['mean']:+.3f}|"
                     f"[{100*f['ci95'][0]:.3f},{100*f['ci95'][1]:.3f}]|")
    lines+=['','S1 has low variability and frozen calendar; S3 increases variance; S4 changes elasticity; S5 increases '
        'correlated shocks; S7 increases mean demand; S8 halves capacity to250. S2, S6 and S4 at elasticity -1.5 '
        'are identical under the current factory and common keys. They are retained for coverage and explicitly '
        'counted as duplicated settings, not three independent robustness successes. These are frozen-policy stress tests, '
        'not retrained component ablations. Full controller means and paired S2-minus-stress degradation are in summary.json.','',
        '## Train/validation calendar gap','',
        'The same frozen final policies were assessed on random training windows and the validation calendar, using '
        'independent key cohorts. The gap combines calendar/distribution change and sample variation, so it cannot isolate '
        'overfitting. Training seed rows are resampled jointly and each cohort scenario axis independently.','',
        '|Metric|Train minus validation|Exploratory 95% interval|','|---|---:|---|']
    for metric,g in summary['train_validation_gap'].items():
        lines.append(f"|{metric}|{g['train_minus_validation']:.6f}|[{g['ci95'][0]:.6f},{g['ci95'][1]:.6f}]|")
    compute=load('compute.json')
    lines+=['','## Compute and verification','',
        f"Platform: {compute['platform']}. Each actor benchmark uses 20 warm-up and 1000 timed calls over 100 archived "
        'observations. It includes Python and input conversion, excludes simulator/loading/HTTP, and does not measure energy or peak memory.',
        '', '|Seed|Median latency (ms)|95th-percentile latency (ms)|Learned parameters|Parameter bytes|',
        '|---|---:|---:|---:|---:|']
    for seed,m in compute['results'].items():
        lines.append(f"|{seed}|{m['median_ms']:.6f}|{m['p95_ms']:.6f}|{m['parameters']}|{m['parameter_bytes']}|")
    lines+=['',f"New characterization: {summary['counters']['episodes']} episodes/{summary['counters']['days']} simulated days, "
        f"{summary['seconds']:.3f}s, zero training updates. Includes 21 exact restored replays. Archive verification reconstructs "
        'primary policy actions, economic/operational accounting, pairing, scenario coverage, progress thresholds and uncertainty. '
        'It adds zero simulator days. Three new metric tests passed; no full-suite result is implied by those focused tests.','',
        '## Interpretation and remaining work','',
        'Reward improvement is conditional on this simulator, engineered policy family and fixed-price setting. The calibration '
        'has documented predictive bias and controlled elasticity assumptions. Existing failure reports are retained. '
        'Neither these stress tests nor the fixed-price improvement establish joint pricing/ordering coordination or a new '
        'algorithmic contribution. The separately declared joint-role pilot and final-scope decision remain separate work.','',
        'Figures: [learning progress](learning_progress.png), [scenario robustness](scenario_robustness.png), '
        '[reference shortfall](reference_shortfall.png). Complete data: summary.json, archived_metrics.json, '
        'compute.json, per-condition metrics and verification.json.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote verified characterization report and figures.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);report(p.parse_args().output)
