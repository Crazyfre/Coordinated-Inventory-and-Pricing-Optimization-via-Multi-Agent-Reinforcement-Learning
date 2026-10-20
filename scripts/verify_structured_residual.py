"""Independently count and recompute the bounded residual experiment evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from src.eval.review_metrics import return_summary


def verify(directory):
    out=Path(directory);d=json.loads((out/'summary.json').read_text());p=d['protocol']
    assert p==json.loads((out/'protocol.json').read_text()) and not d['held_out_test_used']
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for name in ['center','calibration']:assert digest(p[name])==p[name+'_sha256']
    assert digest('scripts/diagnose_structured_residual.py')==p['runner_sha256']
    assert digest('src/algos/structured_residual.py')==p['actor_sha256']
    assert d['completed_updates'] in [0,64]
    assert d['training_transitions']==28*d['completed_updates']
    assert all(15<=s and s+27<=1885 for s in d['training_windows'])
    traces=[json.loads(line) for line in (out/'traces.jsonl').read_text().splitlines()]
    groups={}
    for r in traces:groups.setdefault((r['label'],r['seed']),[]).append(r)
    assert len(groups)==d['scored_episodes']+d['replay_episodes']
    assert len(traces)==len(groups)*28
    rows={(r['label'],r['seed']):r for r in d['episode_rows']}
    for key,ts in groups.items():
        label,seed=key;r=rows[key]
        assert [t['day'] for t in ts]==list(range(1,29))
        assert abs(sum(t['reward'] for t in ts)/28-r['reward_day'])<1e-8
        assert abs(sum(sum(t['sales']) for t in ts)/sum(sum(t['demand']) for t in ts)-r['fill'])<1e-10
        assert abs(np.mean([np.mean(np.equal(t['actions'],t['teacher'])) for t in ts])-r['teacher_agreement'])<1e-10
        assert [t['demand'] for t in ts]==[t['demand'] for t in groups['teacher',seed]]
    for mode in ['greedy','sampled']:
        original=('trained_' if d['completed_updates'] else 'initial_')+mode
        for seed in p['validation_seeds']:
            assert groups[original,seed]==[dict(t,label=original) for t in groups['replay_'+mode,seed]]
    def delta(label,ref,metric):
        return return_summary([rows[label,s][metric]-rows[ref,s][metric] for s in p['validation_seeds']])
    initial_pass=[]
    for mode in ['greedy','sampled']:
        label='initial_'+mode
        reward,fill=delta(label,'teacher','reward_day'),delta(label,'teacher','fill')
        gate=d['means'][label]['teacher_agreement']>=.99 and reward['ci95'][0]>=-1 and fill['ci95'][0]>=-.005
        assert gate==d['initial_gates'][mode]['passed'];initial_pass.append(gate)
    assert bool(d['completed_updates'])==all(initial_pass)
    conditions=[]
    for mode,comparisons in d['comparisons'].items():
        for ref,r in comparisons.items():
            reward,fill=delta('trained_'+mode,ref,'reward_day'),delta('trained_'+mode,ref,'fill')
            np.testing.assert_allclose(reward['ci95'],r['reward_ci95'],atol=1e-10)
            np.testing.assert_allclose(fill['ci95'],r['fill_ci95'],atol=1e-10)
            conditions.append(reward['ci95'][0]>0 and fill['ci95'][0]>=-.005)
    assert d['learning_gate']==bool(conditions and all(conditions))
    audit=dict(status='passed',scored_episodes=d['scored_episodes'],replay_episodes=d['replay_episodes'],
        training_episodes=d['completed_updates'],evaluation_trace_days=len(traces),
        total_simulated_days=len(traces)+d['training_transitions'],hashes_verified=True,
        paired_metrics_and_gates_verified=True,checkpoint_replay_verified=True,
        checkpoint_hashes={f.name:digest(f) for f in out.glob('*.pt')})
    (out/'independent_verification.json').write_text(json.dumps(audit,indent=2))
    lines=['# Structured residual PPO measurements','',
           '| Controller | Reward/day | Fill | Teacher agreement |','|---|---:|---:|---:|']
    for label,r in d['means'].items():
        if not label.startswith('replay'):
            lines.append(f"| {label} | {r['reward_day']:.6f} | {r['fill']:.6f} | {100*r['teacher_agreement']:.3f}% |")
    lines+=['',f"Updates: {d['completed_updates']}. Learning gate: {d['learning_gate']}. Actor parameter L2 change: {d['actor_parameter_l2_change']:.6f}.",
            '', '| Execution / reference | Reward/day difference [paired95% CI] | Fill difference [paired95% CI] |','|---|---:|---:|']
    for mode,comparisons in d['comparisons'].items():
        for ref,r in comparisons.items():
            lines.append(f"| {mode} / {ref} | {r['reward_delta']:.6f} [{r['reward_ci95'][0]:.6f}, {r['reward_ci95'][1]:.6f}] | {r['fill_delta']:.6f} [{r['fill_ci95'][0]:.6f}, {r['fill_ci95'][1]:.6f}] |")
    lines+=['','Single training seed; confidence intervals quantify episode uncertainty, not training-seed stability.',
             'Fixed prices and restricted residual support; not a vanilla MAPPO coordination comparison.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');verify(p.parse_args().directory)
