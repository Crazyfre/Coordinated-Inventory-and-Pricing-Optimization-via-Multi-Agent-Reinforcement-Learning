"""Freeze all eligible policies before the one-shot held-out simulator matrix."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scripts.train_compact_residual import setup,INPUTS,digest,write_json
from scripts.discover_parameter_rl import episode
from scripts.characterize_parameter_rl import CONDITIONS
from src.eval.parameter_review import episode_metrics
from src.eval.structured_final import SEEDS,KEYS,analyze
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env


def freeze(training):
    root=Path(training);load=lambda p:json.loads(p.read_text())
    manifest,summary,audit=load(root/'manifest.json'),load(root/'summary.json'),load(root/'verification.json')
    if summary['status']!='complete' or audit['status']!='passed':raise ValueError('verified complete training required')
    if audit.get('recovery_accounting_verified') is not True:raise ValueError('verified interruption recovery required')
    for p,h in manifest['hashes'].items():
        if digest(p)!=h:raise ValueError(f'frozen training source changed: {p}')
    records={};policies={};runmap={(r['condition'],r['seed']):r for r in summary['runs']}
    for condition in ['main','pooled','independent']:
        for seed in SEEDS:
            if condition=='main' and str(seed) in manifest['existing_main']:
                old=manifest['existing_main'][str(seed)];path=Path(old['path']);expected=old['sha256']
            else:
                path=root/f'{condition}_{seed}'/'final.json';expected=runmap[condition,seed]['checkpoint_sha256']
            if digest(path)!=expected:raise ValueError('frozen checkpoint changed')
            state=load(path)
            if state['steps']!=64:raise ValueError('final64-update mean required')
            p=np.asarray(state['mean'],float)
            if condition=='pooled':
                if p.shape!=(1,):raise ValueError('pooled scalar required')
                p=np.repeat(p,10)
            if p.shape!=(10,) or not np.isfinite(p).all() or np.any((p<0)|(p>16)):raise ValueError('invalid cover policy')
            label=f'{condition}_{seed}';policies[label]=p
            records[label]=dict(path=str(path),sha256=expected,parameters=p.tolist(),training_seed=seed,condition=condition)
    sources=['STRUCTURED_FINAL_STUDY_PROTOCOL.md','STRUCTURED_TRAINING_RECOVERY.md',
        'Docs/3. Structured RL study adoption.md',__file__,'scripts/verify_structured_final.py',
        'src/eval/structured_final.py','src/eval/parameter_review.py','src/eval/review_metrics.py',
        'scripts/discover_parameter_rl.py','scripts/characterize_parameter_rl.py',
        'src/algos/parameter_policy_gradient.py','src/algos/structured_imitation.py',
        'src/experiments/phase4.py','src/experiments/research.py','src/env/joint_env.py',
        'src/env/replenishment_env.py','src/env/pipeline.py','src/config/locked_config.py']+list(INPUTS.values())
    sources += ['scripts/train_compact_residual.py','scripts/train_clean_research.py',
        'src/algos/compact_residual.py','src/algos/imitation_ordering.py',
        'src/demand/identifiable_calibration.py','scripts/verify_parameter_confirmation.py']
    sources += [str(p) for p in Path('src').rglob('*.py')]
    sources += ['src/config/skus.json','scripts/report_structured_final.py',
        'data/processed/aligned_skus.csv','data/processed/aligned_skus.key.json']
    old_roots={str(Path(r['path']).parent) for r in manifest['existing_main'].values()}
    for old_root in old_roots:
        old=Path(old_root);proof=load(old/'verification.json')
        if proof['status']!='passed' or proof['reconstructed_updates']!=320 or proof['checked_training_episodes']!=5120:
            raise ValueError('existing main cohort verification required')
        sources += [str(old/name) for name in ['verification.json','frozen_checkpoints.json','manifest.json']]
    hashes=dict(manifest['hashes']);hashes.update({p:digest(p) for p in sources})
    result=dict(status='frozen_before_test',policies=records,hashes=hashes,training=str(root),
        training_summary_sha256=digest(root/'summary.json'),training_verification_sha256=digest(root/'verification.json'),
        seeds=SEEDS,keys=KEYS,conditions=CONDITIONS,split='test',start=1914,days=28,
        primary_condition='S2',primary_reference='full7_5',max_seconds=12000,
        baseline_labels=['full7_5','projected','constant','ss'],price_multiplier=1.2,pricing_scope='shared',
        claim='Conditional calibrated-simulator final evaluation; no causal demand or coordination claim')
    return result,policies


def run(training,output):
    manifest,policies=freeze(training)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'frozen_manifest.json',manifest)  # Must precede creation/reset of any test environment.
    started=time.perf_counter();deadline=started+12000;counters=dict(episodes=0,days=0);stage='setup'
    torch.set_num_threads(1)
    try:
        _,reference=setup(211) # training-only constructor for projected reference; no rollout.
        calibration=load_identifiable_training_calibration(INPUTS['calibration'])
        policies.update(full7_5=np.full(10,7.5),projected='projected',constant='constant',ss='ss')
        analyses={}
        for name,scenario,elasticity in CONDITIONS:
            stage=name;env=make_research_env(scenario,split='test',start=1914,days=28,params=calibration,
                pricing_scope='shared',elasticity=elasticity)
            rows=[];targets={}
            with (out/f'{name}_traces.jsonl').open('w') as handle:
                for key in KEYS:
                    noise=None
                    for label,p in policies.items():
                        _,trace=episode(env,p,key,label,handle,counters,deadline,reference)
                        assert all(r['start']==1914 and r['info']['total_on_hand']<=(250 if scenario=='S8' else 500)+1e-8 for r in trace)
                        demand=[r['info']['demand'] for r in trace]
                        if noise is None:noise=demand
                        assert noise==demand
                        rows.append(episode_metrics(trace))
                        if key==KEYS[0] and label in ['main_211','pooled_211','independent_211','full7_5']:targets[label]=trace
                    write_json(out/'progress.json',dict(condition=name,completed_scenarios=key-KEYS[0]+1,counters=counters))
            result=analyze(rows);analyses[name]=result
            write_json(out/f'{name}_metrics.json',dict(rows=rows,analysis=result))
            with (out/f'{name}_replays.jsonl').open('w') as handle:
                for label,expected in targets.items():
                    # Restore saved numerical parameters from the immutable freeze, not an updated learner.
                    p=np.array(manifest['policies'][label]['parameters']) if label!='full7_5' else np.full(10,7.5)
                    _,actual=episode(env,p,KEYS[0],label,handle,counters,deadline,reference)
                    assert actual==expected
            print(f'{name}:50 paired scenarios and4 exact replays complete',flush=True)
        assert counters==dict(episodes=17040,days=477120) and time.perf_counter()<=deadline
        for p,h in manifest['hashes'].items():assert digest(p)==h
        for r in manifest['policies'].values():assert digest(r['path'])==r['sha256']
        write_json(out/'summary.json',dict(status='complete',analysis=analyses,
            primary_gate=analyses['S2']['gate'],counters=counters,seconds=time.perf_counter()-started,
            exact_replays=40,test_scoring=True,training_updates=0,
            frozen_manifest_sha256=digest(out/'frozen_manifest.json'),
            trace_hashes={p.name:digest(p) for p in out.glob('*.jsonl')}))
    except Exception as exc:
        write_json(out/'failed_status.json',dict(stage=stage,error=str(exc),counters=counters,
            seconds=time.perf_counter()-started,test_scoring=True))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.training,a.output)
