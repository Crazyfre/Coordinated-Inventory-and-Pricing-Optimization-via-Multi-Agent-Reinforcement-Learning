"""Frozen review supplement: archived metrics, progress and scenario robustness."""
import argparse
import json
import platform
from pathlib import Path
import time
import numpy as np
import torch
from scripts.train_compact_residual import setup, INPUTS, digest, write_json
from scripts.discover_parameter_rl import episode
from scripts.verify_parameter_confirmation import episodes
from src.eval.parameter_review import episode_metrics, summarize, progress_threshold, crossed, SEEDS
from src.algos.parameter_policy_gradient import initial_mean
from src.algos.learned_cover_policy import LearnedCoverPolicy
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env

CONDITIONS = [('S1','S1',None),('S2','S2',None),('S3','S3',None),
    ('S4_e05','S4',-.5),('S4_e15','S4',-1.5),('S4_e30','S4',-3.),
    ('S5','S5',None),('S6','S6',None),('S7','S7',None),('S8','S8',None)]


def run(output, study):
    out, study = Path(output), Path(study)
    out.mkdir(parents=True, exist_ok=False)
    verified = json.loads((study/'verification.json').read_text())
    assert verified['status'] == 'passed' and verified['confirmed_gate']
    manifest = json.loads((study/'manifest.json').read_text())
    for path, expected in manifest['hashes'].items():
        assert digest(path) == expected
    inputs = list(manifest['hashes']) + [str(study/n) for n in ['summary.json','verification.json',
        'assessment_traces.jsonl','frozen_checkpoints.json']]
    inputs += [str(study/f'sku_seed{s}_{kind}.json') for s in SEEDS for kind in ['final','history']]
    inputs += ['PARAMETER_RL_CHARACTERIZATION_PROTOCOL.md', __file__,
               'scripts/verify_parameter_characterization.py','src/eval/parameter_review.py',
               'src/algos/learned_cover_policy.py']
    hashes = {p:digest(p) for p in inputs}
    write_json(out/'manifest.json', dict(hashes=hashes, study=str(study), seeds=SEEDS,
        conditions=CONDITIONS, train_keys=[1600000,1600031], validation_keys=[1610000,1610031],
        updates=[0,16,32,48,64], max_seconds=2400, test_scoring=False))
    torch.set_num_threads(1)
    # Archive-only metrics are outside the simulation runtime cap.
    archived, observations = [], []
    for _, trace in episodes(study/'assessment_traces.jsonl'):
        archived.append(episode_metrics(trace))
        if len(observations) < 100:
            observations += [(r['obs'],r['day']) for r in trace[:min(28,100-len(observations))]]
    write_json(out/'archived_metrics.json',dict(rows=archived,analysis=summarize(archived),new_simulated_days=0))
    calibration = json.loads(Path(INPUTS['calibration']).read_text())
    timings = {}
    for seed in SEEDS:
        controller = LearnedCoverPolicy.load(study/'policies'/f'seed{seed}.json',
            series_order=calibration['series_order'], calibration_path=INPUTS['calibration'])
        for obs, day in observations[:20]:
            controller.act(obs, day)
        samples=[]
        for i in range(1000):
            obs, day = observations[i % len(observations)]
            start=time.perf_counter_ns(); controller.act(obs,day)
            samples.append((time.perf_counter_ns()-start)/1e6)
        timings[str(seed)] = dict(median_ms=float(np.median(samples)),p95_ms=float(np.quantile(samples,.95)),
            parameters=10,parameter_bytes=int(controller.parameters.nbytes))
    write_json(out/'compute.json',dict(platform=platform.platform(),processor=platform.processor(),
        warmup_per_seed=20,timed_per_seed=1000,observations=len(observations),results=timings,
        scope='Loaded deterministic actor including Python/list conversion; excludes simulator/loading/HTTP; bytes are not peak RAM'))
    finals = {f'learned_{s}': np.array(json.loads((study/f'sku_seed{s}_final.json').read_text())['mean']) for s in SEEDS}
    progress_policies = {'full7_5':initial_mean('sku')}
    for s in SEEDS:
        history=json.loads((study/f'sku_seed{s}_history.json').read_text())
        for u in [16,32,48,64]:
            progress_policies[f'u{u}_{s}']=np.array(history[u-1]['state']['mean'])
    started=time.perf_counter(); deadline=started+2400; counters=dict(episodes=0,days=0)
    rows_by_condition={}; replay_targets={}; stage='progress'
    try:
        train_env, reference=setup(SEEDS[0])
        train_rows=[]
        with (out/'progress_traces.jsonl').open('w') as handle:
            for key in range(1600000,1600032):
                demand=None
                for label, parameters in progress_policies.items():
                    _, trace=episode(train_env,parameters,key,label,handle,counters,deadline,reference)
                    noise=[r['info']['demand'] for r in trace]
                    if demand is None: demand=noise
                    assert demand == noise
                    train_rows.append(episode_metrics(trace))
                    if key==1600000 and label=='u64_211': replay_targets['progress']=(label,parameters,key,trace)
                write_json(out/'progress.json',dict(stage=stage,keys=key-1600000+1,counters=counters))
        progress_analysis=progress_threshold(train_rows)
        write_json(out/'learning_progress.json',dict(rows=train_rows,analysis=progress_analysis))
        params=load_identifiable_training_calibration(INPUTS['calibration'])
        policies=dict(finals,full7_5=initial_mean('sku'),projected='projected',constant='constant',ss='ss')
        for name, scenario, elasticity in CONDITIONS:
            stage=name
            env=make_research_env(scenario,split='validation',start=1886,days=28,params=params,
                                  pricing_scope='shared',elasticity=elasticity)
            rows=[]
            with (out/f'{name}_traces.jsonl').open('w') as handle:
                for key in range(1610000,1610032):
                    demand=None
                    for label, parameters in policies.items():
                        _, trace=episode(env,parameters,key,label,handle,counters,deadline,reference)
                        assert all(r['info']['total_on_hand'] <= (250 if scenario=='S8' else 500)+1e-8 for r in trace)
                        noise=[r['info']['demand'] for r in trace]
                        if demand is None: demand=noise
                        assert demand == noise
                        rows.append(episode_metrics(trace))
                        if key==1610000 and label in ['learned_211','full7_5']:
                            replay_targets[name+'_'+label]=(label,parameters,key,trace)
                    write_json(out/'progress.json',dict(stage=stage,keys=key-1610000+1,counters=counters))
            rows_by_condition[name]=rows
            write_json(out/f'{name}_metrics.json',dict(rows=rows,analysis=summarize(rows)))
            print(f'{name}: completed 32 scenarios; {counters["episodes"]} episodes total',flush=True)
            with (out/f'{name}_replays.jsonl').open('w') as handle:
                for label in ['learned_211','full7_5']:
                    label,parameters,key,expected=replay_targets[name+'_'+label]
                    _, actual=episode(env,parameters,key,label,handle,counters,deadline,reference)
                    assert actual == expected
        with (out/'progress_replay.jsonl').open('w') as handle:
            label,parameters,key,expected=replay_targets['progress']
            _,actual=episode(train_env,parameters,key,label,handle,counters,deadline,reference)
            assert actual == expected
        analyses={name:summarize(rows) for name,rows in rows_by_condition.items()}
        train_index={(r['label'],r['scenario']):r for r in train_rows}
        val_index={(r['label'],r['scenario']):r for r in rows_by_condition['S2']}
        # Independent key cohorts: bootstrap seed rows jointly, scenario columns independently.
        rng=np.random.default_rng(20261021); gaps={}
        for metric in ['reward_day','fill']:
            a=np.array([[train_index[f'u64_{s}',k][metric] for k in range(1600000,1600032)] for s in SEEDS])
            b=np.array([[val_index[f'learned_{s}',k][metric] for k in range(1610000,1610032)] for s in SEEDS])
            draws=[]
            for _ in range(10000):
                r=rng.integers(0,5,5); c=rng.integers(0,32,32); d=rng.integers(0,32,32)
                draws.append(a[np.ix_(r,c)].mean()-b[np.ix_(r,d)].mean())
            gaps[metric]=dict(train_minus_validation=float(a.mean()-b.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())
        degradation={}
        for name,rows in rows_by_condition.items():
            idx={(r['label'],r['scenario']):r for r in rows}
            degradation[name]={m:crossed([[val_index[f'learned_{s}',k][m]-idx[f'learned_{s}',k][m]
                 for k in range(1610000,1610032)] for s in SEEDS]) for m in ['reward_day','fill']}
        assert counters==dict(episodes=3573,days=100044)
        assert time.perf_counter() <= deadline
        for path,expected in hashes.items(): assert digest(path)==expected
        write_json(out/'summary.json',dict(status='complete',analysis=analyses,progress=progress_analysis,
            train_validation_gap=gaps,S2_minus_stress=degradation,counters=counters,seconds=time.perf_counter()-started,
            exact_replays=21,test_scoring=False,training_updates=0,
            trace_hashes={p.name:digest(p) for p in out.glob('*.jsonl')}))
        print('Frozen characterization complete; no training or test scoring.',flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json',dict(stage=stage,error=str(exc),counters=counters,seconds=time.perf_counter()-started))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--study',required=True)
    a=p.parse_args();run(a.output,a.study)
