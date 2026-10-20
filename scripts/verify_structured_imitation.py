"""Audit the reconstructed dataset and scored traces; generate evidence tables."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.algos.imitation_ordering import teacher_indices
from src.eval.review_metrics import return_summary


def verify(directory):
    out=Path(directory);summary=json.loads((out/'summary.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    assert summary['protocol']==protocol and summary['rl_updates']==0 and not summary['held_out_test_used']
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert digest(protocol['calibration'])==protocol['calibration_sha256']
    for name,expected in protocol['source_sha256'].items():
        assert digest(Path(protocol['training_source'])/name)==expected
    assert digest('scripts/diagnose_structured_imitation.py')==protocol['runner_sha256']
    assert digest('src/algos/structured_imitation.py')==protocol['actor_sha256']
    assert digest(out/'reconstructed_data.npz')==summary['data_sha256']
    assert digest(out/'structured_policy.json')==summary['checkpoint_sha256']
    data=np.load(out/'reconstructed_data.npz')
    policy=json.loads((out/'structured_policy.json').read_text())
    n,ph=policy['n'],policy['pipe_horizon'];coef=np.asarray(policy['coefficients'])
    agreements={}
    for partition,expected in [('train',8960),('holdout',448)]:
        x,y=data[partition+'_obs'],data[partition+'_labels']
        assert len(x)==expected and y.shape==(expected,10)
        np.testing.assert_array_equal(teacher_indices(torch.tensor(x),n,ph),y)
        # Independently reconstruct the fitted score, without calling policy.predict.
        pos=torch.tensor(x[:,:n])+torch.tensor(x[:,n:n*(1+ph)]).reshape(-1,n,ph).sum(-1)
        score=coef[0]*(x[:,-n:].astype(float)/5)+coef[1]*(pos.numpy().astype(float)/5)+coef[2]
        actions=np.ceil(np.maximum(score,0)).clip(0,40).astype(int)
        agreements[partition]=float(np.mean(actions==y))
        assert abs(agreements[partition]-summary['agreement']['structured_'+partition])<1e-12
    assert all(15<=s and s+27<=1885 for s in summary['training_windows'])
    traces=[json.loads(line) for line in (out/'validation_traces.jsonl').read_text().splitlines()]
    groups={}
    for r in traces:groups.setdefault((r['label'],r['seed']),[]).append(r)
    assert len(groups)==140 and len(traces)==3920
    for rows in groups.values():assert [r['day'] for r in rows]==list(range(1,29))
    indexed={(r['label'],r['seed']):r for r in summary['episode_rows']}
    assert len(indexed)==120
    for (label,seed),r in indexed.items():
        daily=groups[label,seed]
        assert abs(sum(t['reward'] for t in daily)/28-r['reward_day'])<1e-8
        assert abs(sum(sum(t['sales']) for t in daily)/sum(sum(t['demand']) for t in daily)-r['fill'])<1e-10
        assert abs(np.mean([np.mean(np.equal(t['actions'],t['teacher'])) for t in daily])-r['teacher_agreement'])<1e-10
        assert [t['demand'] for t in daily]==[t['demand'] for t in groups['teacher',seed]]
    for seed in protocol['validation_seeds']:
        assert groups['structured',seed]==[dict(r,label='structured') for r in groups['replay',seed]]
    for ref,reported in summary['comparisons'].items():
        for metric,key in [('reward_day','reward_ci95'),('fill','fill_ci95')]:
            stats=return_summary([indexed['structured',s][metric]-indexed[ref,s][metric] for s in protocol['validation_seeds']])
            np.testing.assert_allclose(stats['ci95'],reported[key],atol=1e-10)
    expected_gate=(agreements['holdout']>=.99 and summary['means']['structured']['teacher_agreement']>=.99
        and summary['comparisons']['teacher']['reward_ci95'][0]>=-1
        and summary['comparisons']['teacher']['fill_ci95'][0]>=-.005
        and summary['comparisons']['initial']['reward_ci95'][0]>0
        and summary['comparisons']['initial']['fill_ci95'][0]>=-.005)
    assert expected_gate==summary['gate']['passed']
    verified=dict(status='passed',source_and_artifact_hashes=True,train_states=8960,holdout_states=448,
        saved_labels_match_teacher=True,independent_prediction_agreement=agreements,
        scored_episodes=120,replay_episodes=20,validation_trace_days=3920,
        paired_intervals_and_gate_verified=True,training_windows_valid=True)
    (out/'independent_verification.json').write_text(json.dumps(verified,indent=2))
    lines=['# Structured imitation: measured results','',
           '| Controller | Reward/day | Fill | Own-state teacher agreement |','|---|---:|---:|---:|']
    for label,r in summary['means'].items():lines.append(f"| {label} | {r['reward_day']:.6f} | {r['fill']:.6f} | {100*r['teacher_agreement']:.3f}% |")
    lines+=['','| Structured minus comparator | Reward/day [paired95% CI] | Fill [paired95% CI] |','|---|---:|---:|']
    for ref,r in summary['comparisons'].items():lines.append(f"| {ref} | {r['reward_delta']:.6f} [{r['reward_ci95'][0]:.6f}, {r['reward_ci95'][1]:.6f}] | {r['fill_delta']:.6f} [{r['fill_ci95'][0]:.6f}, {r['fill_ci95'][1]:.6f}] |")
    lines+=['',f"Fitted coefficients: {coef.tolist()}. Margin: {summary['fit']['margin']:.9g}.",
             f"Train agreement: {agreements['train']:.6f}; training-holdout: {agreements['holdout']:.6f}; gate: {expected_gate}.",
             'One deterministic supervised fit; no reward optimization or coordination claim.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for i,(label,r) in enumerate(summary['means'].items()):
        stats=return_summary([indexed[label,s]['reward_day'] for s in protocol['validation_seeds']])
        ax[0].errorbar(stats['mean'],i,xerr=[[stats['mean']-stats['ci95'][0]],[stats['ci95'][1]-stats['mean']]],fmt='o')
    ax[0].set_yticks(range(6),list(summary['means']));ax[0].set_xlabel('Reward/day (95% episode CI)')
    ax[0].axvline(0,color='gray',linewidth=.8)
    categories=['Training labels','Training holdout','Validation own states']
    a=summary['agreement'];indices=np.arange(3)
    ax[1].bar(indices-.18,[a['raw_train'],a['raw_holdout'],summary['means']['raw_actor']['teacher_agreement']],.36,label='Archived raw actor')
    ax[1].bar(indices+.18,[a['structured_train'],a['structured_holdout'],summary['means']['structured']['teacher_agreement']],.36,label='Structured imitation')
    ax[1].axhline(.99,color='gray',linestyle='--',label='Gate .99');ax[1].set_ylim(0,1.08)
    ax[1].set_xticks(indices,categories,rotation=15);ax[1].set_ylabel('SKU action agreement with teacher');ax[1].legend(fontsize=8)
    fig.suptitle('Representation diagnostic: saved training labels, fixed1.20x prices')
    fig.savefig(out/'results.png',dpi=150);plt.close(fig)
    print(json.dumps(verified,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');verify(p.parse_args().directory)
