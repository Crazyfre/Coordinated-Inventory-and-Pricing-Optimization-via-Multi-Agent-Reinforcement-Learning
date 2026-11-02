"""Prospective three-seed MAPPO/control comparison; no adaptive decisions."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.structured_mappo import StructuredMAPPO
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env
from src.experiments.execution_journal import digest,atomic_json
from scripts.recovery_source_inventory import runtime_identity
from scripts.train_compact_residual import serial

OUT=Path('outputs/direct_mappo/comparison_v1')
CALIBRATION='outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json'
ORDERING='outputs/rl_discovery/parameter_confirmation_v1/sku_seed211_final.json'
SEEDS=(1801,1802,1803)
ARMS=('mappo','ippo','separate')


def compose(pricing_model,ordering_model,env,covers,seed):
    model=StructuredMAPPO(env,covers,seed,critic_mode='local')
    model.pricing.load_state_dict(pricing_model.pricing.state_dict())
    model.ordering.load_state_dict(ordering_model.ordering.state_dict())
    return model # Evaluation only; its critics/optimizers have not been trained.


def analyze(rows):
    expected={('baseline',0,k) for k in range(3180000,3180032)}
    expected|={(a,s,k) for a in ARMS for s in SEEDS for k in range(3180000,3180032)}
    index={(r['arm'],r['seed'],r['key']):r for r in rows}
    if set(index)!=expected or len(rows)!=len(expected):raise ValueError('incomplete evaluation grid')
    rng=np.random.default_rng(20261218)
    seed_draws=rng.integers(3,size=(10000,3));key_draws=rng.integers(32,size=(10000,32))
    effects={}
    for reference in ('baseline','ippo','separate'):
        for metric in ('reward_day','fill'):
            delta=np.array([[index['mappo',s,k][metric]-index[reference,0 if reference=='baseline' else s,k][metric]
                for k in range(3180000,3180032)] for s in SEEDS])
            draws=delta[seed_draws[:,:,None],key_draws[:,None,:]].mean(axis=(1,2))
            effects[f'{reference}:{metric}']=dict(mean=float(delta.mean()),
                seed_means=delta.mean(1).tolist(),ci991667=np.quantile(draws,[.05/12,1-.05/12]).tolist())
    gates={ref:bool(effects[f'{ref}:reward_day']['ci991667'][0]>0 and
                   effects[f'{ref}:fill']['ci991667'][0]>=-.005) for ref in ('baseline','ippo','separate')}
    gates['baseline']=bool(gates['baseline'] and effects['baseline:reward_day']['mean']>=.5)
    return dict(effects=effects,gates=gates,means={arm:{m:float(np.mean([r[m] for r in rows if r['arm']==arm]))
        for m in ('reward_day','fill')} for arm in ('baseline',)+ARMS},
        limitation='Only three training seeds; exploratory crossed-bootstrap intervals, reused training calendar.')


def main():
    began=time.perf_counter();deadline=began+1740
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    paths={p.as_posix() for folder in ('src','scripts') for p in Path(folder).rglob('*.py')}
    paths|={CALIBRATION,ORDERING,'DIRECT_MAPPO_COMPARISON_PROTOCOL.md','src/config/skus.json',
        'data/processed/aligned_skus.csv','data/processed/aligned_skus.key.json',
        'outputs/run_direct_mappo_comparison.py'}
    hashes={p:digest(p) for p in sorted(paths)}
    atomic_json(OUT/'manifest.json',dict(hashes=hashes,runtime=runtime_identity(),seeds=list(SEEDS),
        expected_days=267008,training_episodes=9216,evaluation_episodes=320,
        internal_seconds=1740,external_seconds=1800,protocol='DIRECT_MAPPO_COMPARISON_PROTOCOL.md'))
    days=0;records=[];models={};checkpoints={}
    def collect(env,model,key,**kwargs):
        nonlocal days
        if days+28>267008:raise RuntimeError('comparison day cap')
        env.sim.start_day_index=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
        episode=model.collect(env,key,deadline=deadline,**kwargs);days+=28
        return episode
    def save(model,name,**extra):
        path=OUT/(name+'.pt');torch.save(dict(**model.checkpoint(),**extra),path)
        checkpoints[path.name]=digest(path)
    try:
        env=make_research_env('S2',split='train',start=15,days=28,pricing_scope='per_sku',
            params=load_identifiable_training_calibration(CALIBRATION))
        covers=np.asarray(json.loads(Path(ORDERING).read_text())['mean'])
        baseline=StructuredMAPPO(env,covers,1801);save(baseline,'baseline')
        for j,seed in enumerate(SEEDS):
            base=3070000+20000*j
            for label,mode,updates,offset,frozen,rng_seed in (
                ('mappo','central',128,0,(),seed+10000),
                ('ippo','local',128,0,(),seed+10000),
                ('separate_pricing','local',64,0,('ordering',),seed+10000),
                ('separate_ordering','local',64,512,('pricing',),seed+20000)):
                model=StructuredMAPPO(env,covers,seed,critic_mode=mode)
                save(model,f'{seed}_{label}_initial');torch.manual_seed(rng_seed)
                for update in range(updates):
                    batch=[collect(env,model,base+offset+8*update+i,frozen=frozen) for i in range(8)]
                    if time.perf_counter()>deadline:raise TimeoutError('update deadline')
                    metrics=model.update(batch)
                    records.append(dict(seed=seed,arm=label,update=update+1,
                        first_key=base+offset+8*update,last_key=base+offset+8*update+7,
                        **metrics,training_reward_day=float(np.mean([sum(e['rewards'])/28 for e in batch]))))
                    if (update+1)%16==0:
                        atomic_json(OUT/'progress.json',dict(status='training',seed=seed,arm=label,
                            update=update+1,days=days,elapsed_seconds=time.perf_counter()-began))
                models[seed,label]=model;save(model,f'{seed}_{label}_final')
            models[seed,'separate']=compose(models[seed,'separate_pricing'],models[seed,'separate_ordering'],env,covers,seed)
            save(models[seed,'separate'],f'{seed}_separate_final',evaluation_only=True)
        if days!=258048:raise AssertionError('training accounting mismatch')
        atomic_json(OUT/'training_metrics.json',records)
        atomic_json(OUT/'checkpoint_freeze.json',dict(hashes=checkpoints,training_days=days))
        rows=[]
        with (OUT/'evaluation.jsonl').open('x',encoding='utf-8') as handle:
            policies=[('baseline',0,baseline)]+[(a,s,models[s,a]) for s in SEEDS for a in ARMS]
            for arm,seed,model in policies:
                for key in range(3180000,3180032):
                    episode=collect(env,model,key,greedy=True,training=False)
                    demand=sum(np.asarray(r['info']['demand']).sum() for r in episode['trace'])
                    sales=sum(np.asarray(r['info']['sales']).sum() for r in episode['trace'])
                    row=dict(arm=arm,seed=seed,key=key,reward_day=sum(episode['rewards'])/28,
                        fill=float(sales/demand) if demand else 1.)
                    handle.write(json.dumps(serial(dict(**row,trace=episode['trace'])),allow_nan=False)+'\n')
                    handle.flush();rows.append(row)
                atomic_json(OUT/'progress.json',dict(status='evaluating',arm=arm,seed=seed,days=days))
        if days!=267008:raise AssertionError('total day accounting mismatch')
        if any(digest(p)!=h for p,h in hashes.items()):raise RuntimeError('source/input changed')
        if any(digest(OUT/p)!=h for p,h in checkpoints.items()):raise RuntimeError('checkpoint changed after freeze')
        if time.perf_counter()>deadline:raise TimeoutError('finalization deadline')
        atomic_json(OUT/'report.json',dict(status='complete',days=days,rows=rows,
            elapsed_seconds=time.perf_counter()-began,analysis=analyze(rows),
            independently_verified=False,full_goal_achieved=False))
    except BaseException as exc:
        atomic_json(OUT/'failure.json',dict(status='failed',error=f'{type(exc).__name__}: {exc}',
            completed_days=days,failed_episode_steps='unknown if interrupted',elapsed_seconds=time.perf_counter()-began))
        raise


if __name__=='__main__':main()
