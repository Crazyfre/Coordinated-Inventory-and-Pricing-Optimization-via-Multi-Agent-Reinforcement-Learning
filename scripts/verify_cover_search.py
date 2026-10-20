"""Independent trace accounting, frozen-choice verification and cover-search plots."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from src.eval.review_metrics import return_summary


def verify(directory):
    out=Path(directory);d=json.loads((out/'summary.json').read_text());p=d['protocol']
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert not d['held_out_test_used'] and d['rl_updates']==0
    assert digest(p['calibration'])==p['calibration_sha256']
    assert digest('scripts/diagnose_cover_search.py')==p['runner_sha256']
    assert digest(out/'selection.json')==d['selection_sha256']
    assert json.loads((out/'selection.json').read_text())==d['selection']
    assert d['selection']['validation_scored'] is False
    all_groups={}
    for partition in ['training','validation']:
        groups={}
        for line in (out/(partition+'_traces.jsonl')).read_text().splitlines():
            row=json.loads(line);groups.setdefault((row['label'],row['seed']),[]).append(row)
        expected=576 if partition=='training' else 200
        assert len(groups)==expected
        for key,ts in groups.items():
            assert [r['day'] for r in ts]==list(range(1,29))
            assert all(len(r['actions'])==10 and all(0<=a<=40 for a in r['actions']) for r in ts)
        for r in d[partition+'_episode_rows']:
            ts=groups[r['label'],r['seed']]
            assert abs(sum(t['reward'] for t in ts)/28-r['reward_day'])<1e-8
            assert abs(sum(sum(t['sales']) for t in ts)/sum(sum(t['demand']) for t in ts)-r['fill'])<1e-10
        reference='cover_6' if partition=='training' else 'teacher'
        for (label,seed),ts in groups.items():
            assert [r['demand'] for r in ts]==[r['demand'] for r in groups[reference,seed]]
        all_groups[partition]=groups
    assert len(d['training_windows'])==576 and all(15<=s and s+27<=1885 for s in d['training_windows'])
    assert all(d['training_windows'][i*64:(i+1)*64]==d['training_windows'][:64] for i in range(9))
    candidates=[]
    for cover in p['covers']:
        rows=[r for r in d['training_episode_rows'] if r['label']==f'cover_{cover:g}']
        candidates.append(dict(cover=cover,reward_day=float(np.mean([r['reward_day'] for r in rows])),
                               fill=float(np.mean([r['fill'] for r in rows]))))
    assert candidates==d['selection']['training_candidates']
    teacher_fill=next(r['fill'] for r in candidates if r['cover']==6)
    eligible=[r for r in candidates if r['fill']>=teacher_fill-.005]
    selected=sorted(eligible,key=lambda r:(-r['reward_day'],abs(r['cover']-6),r['cover']))[0]
    assert selected==d['selection']['selected']
    indexed={(r['label'],r['seed']):r for r in d['validation_episode_rows']}
    for ref,c in d['comparisons'].items():
        reward=return_summary([indexed['selected',s]['reward_day']-indexed[ref,s]['reward_day'] for s in p['validation_seeds']])
        fill=return_summary([indexed['selected',s]['fill']-indexed[ref,s]['fill'] for s in p['validation_seeds']])
        np.testing.assert_allclose(reward['ci95'],c['reward_ci95'],atol=1e-10)
        np.testing.assert_allclose(fill['ci95'],c['fill_ci95'],atol=1e-10)
        assert d['gates'][ref]==(reward['ci95'][0]>0 and fill['ci95'][0]>=-.005)
    for s in p['validation_seeds']:
        assert all_groups['validation']['selected',s]==[dict(r,label='selected') for r in all_groups['validation']['replay',s]]
    assert d['all_comparators_gate']==all(d['gates'].values())
    audit=dict(status='passed',training_episodes=576,validation_episodes=160,replay_episodes=40,
               total_trace_days=21728,hashes_verified=True,selection_recomputed=True,paired_metrics_verified=True,
               chronological_windows_verified=True,selection_replay_verified=True)
    (out/'independent_verification.json').write_text(json.dumps(audit,indent=2))
    lines=['# Bounded cover search: measured results','', '| Training cover | Reward/day | Fill | Eligible |','|---|---:|---:|---|']
    for r in candidates:lines.append(f"| {r['cover']} | {r['reward_day']:.6f} | {r['fill']:.6f} | {r in eligible} |")
    lines+=['',f"Selected on training only: cover {selected['cover']}.",'',
            '| Validation controller | Reward/day | Fill |','|---|---:|---:|']
    for label,r in d['means'].items():lines.append(f"| {label} | {r['reward_day']:.6f} | {r['fill']:.6f} |")
    lines+=['','| Selected minus reference | Reward/day [paired95% CI] | Fill [paired95% CI] | Gate |','|---|---:|---:|---|']
    for ref,r in d['comparisons'].items():lines.append(f"| {ref} | {r['reward_delta']:.6f} [{r['reward_ci95'][0]:.6f}, {r['reward_ci95'][1]:.6f}] | {r['fill_delta']:.6f} [{r['fill_ci95'][0]:.6f}, {r['fill_ci95'][1]:.6f}] | {d['gates'][ref]} |")
    lines+=['','Training selected one rule; validation did not select or retune it. No optimality or RL claim.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    covers=[r['cover'] for r in candidates]
    ax[0].plot(covers,[r['reward_day'] for r in candidates],'o-');ax[0].axvline(selected['cover'],color='gray',linestyle='--')
    ax[0].set(xlabel='Training cover (days)',ylabel='Training mean reward/day')
    ax[1].plot(covers,[r['fill'] for r in candidates],'o-');ax[1].axhline(teacher_fill-.005,color='gray',linestyle='--')
    ax[1].set(xlabel='Training cover (days)',ylabel='Training mean fill')
    for i,(label,r) in enumerate(d['means'].items()):
        stats=return_summary([indexed[label,s]['reward_day'] for s in p['validation_seeds']])
        ax[2].errorbar(stats['mean'],i,xerr=[[stats['mean']-stats['ci95'][0]],[stats['ci95'][1]-stats['mean']]],fmt='o')
    ax[2].set_yticks(range(4),list(d['means']));ax[2].set_xlabel('Validation reward/day (episode95% CI)')
    fig.suptitle('Bounded uniform cover search: training selection, frozen validation comparison')
    fig.savefig(out/'results.png',dpi=150);plt.close(fig)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');verify(p.parse_args().directory)
