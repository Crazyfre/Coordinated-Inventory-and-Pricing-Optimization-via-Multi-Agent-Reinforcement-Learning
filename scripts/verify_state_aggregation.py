"""Independent artifact accounting and reporting for the completed bounded run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from src.eval.review_metrics import return_summary


def verify(directory):
    out=Path(directory)
    data=json.loads((out/'summary.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    assert protocol==data['protocol'] and not data['held_out_test_used']
    assert set(protocol['holdout_seeds']).isdisjoint(protocol['collection_seeds'])
    assert all(15<=start and start+27<=1885 for start in data['training_windows'])
    assert hashlib.sha256(Path(protocol['calibration']).read_bytes()).hexdigest()==protocol['calibration_sha256']
    assert hashlib.sha256(Path('scripts/diagnose_state_aggregation.py').read_bytes()).hexdigest()==(out/'source_sha256.txt').read_text()
    traces=[json.loads(line) for line in (out/'traces.jsonl').read_text().splitlines()]
    groups={}
    for r in traces:
        groups.setdefault((r['label'],r['seed']),[]).append(r)
    for rows in groups.values():
        assert [r['day'] for r in rows]==list(range(1,29))
        assert all(len(r['actions'])==10 and all(0<=a<=40 for a in r['actions']) for r in rows)
    counts=Counter(label for label,seed in groups)
    assert counts['heldout_train_teacher']==16 and counts['warmup_teacher']==64
    for arm in protocol['arms']:
        for round_index in range(1,5):
            assert counts[f'{arm}_collection_{round_index}']==64
        assert counts[arm]==counts[arm+'_replay']==20
    assert len(groups)==772 and len(traces)==21616
    for row in data['episode_rows']:
        daily=groups[row['label'],row['seed']]
        assert abs(sum(r['reward'] for r in daily)/28-row['reward_day'])<1e-8
        assert abs(sum(sum(r['sales']) for r in daily)/sum(sum(r['demand']) for r in daily)-row['fill'])<1e-10
        assert abs(np.mean([np.equal(r['actions'],r['teacher']).mean() for r in daily])-row['teacher_agreement'])<1e-10
    by_label={label:{r['seed']:r for r in data['episode_rows'] if r['label']==label} for label in data['means']}
    for arm in protocol['arms']:
        for reference, reported in data['comparisons'][arm].items():
            stats=return_summary([r['reward_day']-by_label[reference][seed]['reward_day'] for seed,r in by_label[arm].items()])
            np.testing.assert_allclose(stats['ci95'],reported['reward_ci95'],atol=1e-10)
        history=[r for r in data['history'] if r['arm']==arm]
        assert [r['labeled_states'] for r in history]==[3584,5376,7168,8960]
        assert sum(r['optimizer_steps'] for r in history)==1960
    checks=dict(status='passed',trace_days=len(traces),episodes=len(groups),
                counts=dict(counts),source_hash_verified=True,calibration_hash_verified=True,
                summary_trace_metrics_verified=True,paired_reward_intervals_verified=True,
                chronological_training_windows_verified=True,
                checkpoint_sha256={arm:hashlib.sha256((out/(arm+'.pt')).read_bytes()).hexdigest()
                                   for arm in protocol['arms']})
    (out/'independent_verification.json').write_text(json.dumps(checks,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for arm in protocol['arms']:
        history=[r for r in data['history'] if r['arm']==arm]
        axes[0].plot([0,1,2,3,4],[data['warmup']['heldout_training_agreement']]+[
            r['heldout_training_agreement'] for r in history],marker='o',label=arm)
    axes[0].axhline(.99,color='gray',linestyle='--',label='Required agreement')
    axes[0].set(xlabel='Aggregation round (0 = shared warm-up)',ylabel='Held-out training action agreement',ylim=(0,1.03))
    axes[0].legend(fontsize=8)
    labels=list(data['means'])
    for i,label in enumerate(labels):
        stats=return_summary([r['reward_day'] for r in by_label[label].values()])
        axes[1].errorbar(stats['mean'],i,xerr=[[stats['mean']-stats['ci95'][0]],[stats['ci95'][1]-stats['mean']]],fmt='o')
    axes[1].set_yticks(range(len(labels)),labels)
    axes[1].set_xlabel('Validation reward/day (episode 95% CI)')
    axes[1].axvline(0,color='gray',linewidth=.8)
    seeds=sorted(by_label['learner_states'])
    diffs=[by_label['learner_states'][s]['reward_day']-by_label['teacher_only'][s]['reward_day'] for s in seeds]
    axes[2].bar(range(1,21),diffs)
    axes[2].axhline(0,color='gray',linewidth=.8)
    axes[2].set(xlabel='Paired validation path',ylabel='Learner-state minus teacher-only reward/day')
    fig.suptitle('Bounded supervised imitation: one initialization seed, fixed 1.20x price')
    fig.savefig(out/'results.png',dpi=150);fig.savefig(out/'results.svg');plt.close(fig)
    lines=['# Learner-state aggregation: completed measurements','',
           '| Controller | Reward/day | Fill | Own-state teacher agreement |','|---|---:|---:|---:|']
    for label,means in data['means'].items():
        lines.append(f"| {label} | {means['reward_day']:.6f} | {means['fill']:.6f} | {100*means['teacher_agreement']:.2f}% |")
    lines+=['','| Arm | Teacher-holdout agreement | Equivalence gate |','|---|---:|---|']
    for arm in protocol['arms']:
        last=[r for r in data['history'] if r['arm']==arm][-1]
        lines.append(f"| {arm} | {100*last['heldout_training_agreement']:.2f}% | {data['gates'][arm]['passed']} |")
    lines+=['','| Learner-state comparator | Reward/day delta [95% paired CI] | Fill delta [95% paired CI] |','|---|---:|---:|']
    for ref,c in data['comparisons']['learner_states'].items():
        lines.append(f"| {ref} | {c['reward_delta']:.6f} [{c['reward_ci95'][0]:.6f}, {c['reward_ci95'][1]:.6f}] | {c['fill_delta']:.6f} [{c['fill_ci95'][0]:.6f}, {c['fill_ci95'][1]:.6f}] |")
    lines+=['',f"Campaign wall time: {data['seconds']:.2f}s. 772 episodes / 21,616 daily traces independently counted.",
             'Supervised learning only. No final-test scoring. One training seed; intervals measure episode uncertainty.',
             'Final checkpoints only; no validation checkpoint selection. See protocol.json for fixed budgets and gates.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    verify(parser.parse_args().directory)
