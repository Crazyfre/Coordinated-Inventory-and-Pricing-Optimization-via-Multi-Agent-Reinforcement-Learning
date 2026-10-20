"""Independent support-count, trace, paired-metric and replay audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from src.eval.review_metrics import return_summary


def verify(directory):
    out=Path(directory);d=json.loads((out/'summary.json').read_text());p=d['protocol']
    assert not d['held_out_test_used'] and d['rl_updates']==0
    digest=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    assert digest('scripts/diagnose_residual_support.py')==p['source_sha256']
    for k,path in p['paths'].items():assert digest(path)==p['hashes'][k]
    traces=[json.loads(line) for line in (out/'traces.jsonl').read_text().splitlines()]
    assert len(traces)==4480
    groups={}
    for r in traces:
        groups.setdefault((r['label'],r['seed']),[]).append(r)
        c,t,projection=np.array(r['center']),np.array(r['target']),np.array(r['projected'])
        expected=np.maximum(np.minimum(t,np.minimum(c+2,40)),np.maximum(c-2,0))
        np.testing.assert_array_equal(projection,expected)
        np.testing.assert_array_equal(r['outside'],t!=expected)
        np.testing.assert_array_equal(r['unreachable_units'],5*np.abs(t-expected))
        np.testing.assert_array_equal(r['actions'],c if r['label']=='center' else t if r['label']=='target' else expected)
    assert len(groups)==160
    indexed={(r['label'],r['seed']):r for r in d['episode_rows']}
    for key,rows in groups.items():
        label,seed=key
        assert [r['day'] for r in rows]==list(range(1,29))
        assert abs(sum(r['reward'] for r in rows)/28-indexed[key]['reward_day'])<1e-8
        assert abs(sum(sum(r['sales']) for r in rows)/sum(sum(r['demand']) for r in rows)-indexed[key]['fill'])<1e-10
        assert [r['demand'] for r in rows]==[r['demand'] for r in groups['center',seed]]
    for seed in p['seeds']:
        assert groups['projected',seed]==[dict(r,label='projected') for r in groups['replay',seed]]
    for label in p['controllers']:
        rs=[r for r in traces if r['label']==label];outside=np.array([r['outside'] for r in rs])
        units=np.array([r['unreachable_units'] for r in rs]);reported=d['coverage'][label]
        assert reported['outside_count']==int(outside.sum()) and reported['sku_decisions']==11200
        assert reported['outside_fraction']==float(outside.mean())
        assert reported['days_any_outside']==int(outside.any(axis=1).sum())
        assert reported['max_unreachable_units']==int(units.max())
        assert reported['mean_unreachable_units_per_decision']==float(units.mean())
        np.testing.assert_array_equal(reported['per_sku_outside_fraction'],outside.mean(axis=0))
    for key,r in d['comparisons'].items():
        a,b=key.split('_minus_')
        reward=return_summary([indexed[a,s]['reward_day']-indexed[b,s]['reward_day'] for s in p['seeds']])
        fill=return_summary([indexed[a,s]['fill']-indexed[b,s]['fill'] for s in p['seeds']])
        np.testing.assert_allclose(reward['ci95'],r['reward_ci95'],atol=1e-10)
        np.testing.assert_allclose(fill['ci95'],r['fill_ci95'],atol=1e-10)
        assert d['gates'][key]==(reward['ci95'][0]>0 and fill['ci95'][0]>=-.005)
    audit=dict(status='passed',scored_episodes=120,replay_episodes=40,trace_days=4480,
               projection_and_support_counts_verified=True,paired_intervals_and_gates_verified=True,
               hashes_verified=True,projection_replay_verified=True)
    (out/'independent_verification.json').write_text(json.dumps(audit,indent=2))
    lines=['# Residual support: measured results','', '| Controller | Reward/day | Fill | Target outside support on own states |','|---|---:|---:|---:|']
    for label,r in d['means'].items():lines.append(f"| {label} | {r['reward_day']:.6f} | {r['fill']:.6f} | {100*d['coverage'][label]['outside_fraction']:.3f}% |")
    lines+=['','| Difference | Reward/day [paired95% CI] | Fill [paired95% CI] | Gate |','|---|---:|---:|---|']
    for key,r in d['comparisons'].items():lines.append(f"| {key} | {r['reward_delta']:.6f} [{r['reward_ci95'][0]:.6f}, {r['reward_ci95'][1]:.6f}] | {r['fill_delta']:.6f} [{r['fill_ci95'][0]:.6f}, {r['fill_ci95'][1]:.6f}] | {d['gates'][key]} |")
    lines+=['','Projection is one feasible supported policy, not a support-constrained optimum. No RL training.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    labels=p['controllers']
    for i,label in enumerate(labels):
        stat=return_summary([indexed[label,s]['reward_day'] for s in p['seeds']])
        ax[0].errorbar(stat['mean'],i,xerr=[[stat['mean']-stat['ci95'][0]],[stat['ci95'][1]-stat['mean']]],fmt='o')
    ax[0].set_yticks(range(3),labels);ax[0].set_xlabel('Reward/day (episode95% CI)')
    x=np.arange(10)
    for i,label in enumerate(labels):ax[1].bar(x+(i-1)*.25,np.array(d['coverage'][label]['per_sku_outside_fraction'])*100,.25,label=label)
    ax[1].set_xticks(x,[str(i+1) for i in range(10)]);ax[1].set(xlabel='SKU index (series order in summary.json)',ylabel='Target action outside residual support (%)')
    ax[1].legend();fig.suptitle('Frozen target and nearest-support projection: fresh paired validation')
    fig.savefig(out/'results.png',dpi=150);plt.close(fig)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');verify(p.parse_args().directory)
