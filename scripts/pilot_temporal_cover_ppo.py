"""Matched episode-coherent versus independent daily latent-plan PPO."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.temporal_cover_ppo import TemporalCoverPPO,evaluation_plan
from src.algos.parameter_policy_gradient import order_action
from src.env.interface import JointAction
from src.eval.parameter_review import episode_metrics
from src.experiments.research import make_research_env
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from scripts.train_clean_research import TrainingWindows
from scripts.train_compact_residual import INPUTS,digest,serial,write_json

STUDY=Path('outputs/rl_discovery/parameter_confirmation_v1')


def collect(env,plan,key,label,handle,counters,deadline):
    env.reset(seed=key);trace=[]
    for day in range(1,29):
        if time.perf_counter()>deadline:raise TimeoutError('temporal PPO time cap')
        if env.pricing_due:env.select_price(np.full(10,4))
        obs=env.replenishment_observation().copy();latent=plan[0 if len(plan)==1 else day-1]
        action=order_action(obs,day,latent)
        _,reward,done,info=env.step(JointAction(None,action));assert done==(day==28)
        row=serial(dict(label=label,scenario=key,day=day,start=int(env.sim.start_day_index),obs=obs,
            latent=latent,actions=action,reward=reward,info=info,demand_seed=int(env.sim.scenario_seed),
            supplier_seed=int(env.pipeline.scenario_seed)))
        handle.write(json.dumps(row,allow_nan=False)+'\n');trace.append(row);counters['days']+=1
    counters['episodes']+=1;handle.flush();result=episode_metrics(trace)
    result['discounted_day']=sum(.99**i*r['reward'] for i,r in enumerate(trace))/28
    return result,trace


def analyze(rows):
    index={(r['label'],r['scenario']):r for r in rows};assert len(index)==len(rows)==448
    keys=range(2350000,2350064);draws=np.random.default_rng(20261030).integers(64,size=(10000,64))
    pairs=[(f'{arm}_{mode}',ref) for arm in ['episode','daily'] for mode in ['greedy','sampled']
        for ref in ['baseline',f'{arm}_initial']]+[('episode_'+m,'daily_'+m) for m in ['greedy','sampled']]
    effects={}
    for label,ref in pairs:
        for metric in ['reward_day','fill']:
            d=np.array([index[label,k][metric]-index[ref,k][metric] for k in keys])
            effects[f'{label}-{ref}:{metric}']=dict(mean=float(d.mean()),ci95=np.quantile(d[draws].mean(1),[.025,.975]).tolist())
    means={label:{m:float(np.mean([index[label,k][m] for k in keys])) for m in ['reward_day','fill']}
           for label in sorted({r['label'] for r in rows})}
    eligible=[f'{arm}_{mode}' for arm in ['episode','daily'] for mode in ['greedy','sampled']
        if effects[f'{arm}_{mode}-baseline:reward_day']['mean']>=.5
        and effects[f'{arm}_{mode}-baseline:fill']['mean']>=-.005
        and effects[f'{arm}_{mode}-{arm}_initial:reward_day']['mean']>0]
    return dict(means=means,effects=effects,promising=eligible)


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    assert json.loads((STUDY/'verification.json').read_text())['confirmed_gate']
    covers=np.array(json.loads((STUDY/'sku_seed211_final.json').read_text())['mean'])
    paths={str(p) for folder in ['src','scripts'] for p in Path(folder).rglob('*.py')}
    paths|={INPUTS['calibration'],str(STUDY/'sku_seed211_final.json'),'TEMPORAL_EXPLORATION_PROTOCOL.md'}
    hashes={p:digest(p) for p in sorted(paths)}
    write_json(out/'manifest.json',dict(hashes=hashes,covers=covers,planned_episodes=2500,maximum_seconds=1800))
    torch.set_num_threads(1);started=time.perf_counter();counters=dict(episodes=0,days=0)
    try:
        env=TrainingWindows(make_research_env('S2',split='train',start=15,days=28,pricing_scope='per_sku',
            params=load_identifiable_training_calibration(INPUTS['calibration'])),28)
        finals={}
        with (out/'training.jsonl').open('w') as handle:
            for arm in ['episode','daily']:
                model=TemporalCoverPPO(covers,arm);torch.save(model.state(),out/f'{arm}_initial.pt')
                for u in range(64):
                    epsilon,plans=model.plans();returns=[]
                    for i,plan in enumerate(plans):
                        result,_=collect(env,plan,2340000+u*8+i//2,f'{arm}_u{u}_p{i}',handle,counters,started+1800)
                        returns.append(result['discounted_day'])
                    returns=np.array(returns).reshape(8,2)
                    record=dict(epsilon=epsilon.tolist(),plans=plans.tolist(),returns=returns.tolist(),
                        metrics=model.update(plans,returns),state=model.state())
                    torch.save(record,out/f'{arm}_u{u:03d}.pt')
                    write_json(out/'progress.json',dict(arm=arm,updates=u+1,counters=counters))
                    if (u+1)%8==0:print(f'{arm}: {u+1}/64 updates',flush=True)
                finals[arm]=model.mean.detach().numpy().copy()
        frozen={p:digest(out/p) for p in ['episode_u063.pt','daily_u063.pt']}
        write_json(out/'assessment_freeze.json',frozen);rows=[];targets={}
        with (out/'assessment.jsonl').open('w') as handle:
            for key in range(2350000,2350064):
                policies={'baseline':evaluation_plan(covers,'episode',key,True)}
                for arm in ['episode','daily']:
                    policies[f'{arm}_initial']=evaluation_plan(covers,arm,key)
                    for mode in ['greedy','sampled']:policies[f'{arm}_{mode}']=evaluation_plan(finals[arm],arm,key,mode=='greedy')
                for label,plan in policies.items():
                    result,trace=collect(env,plan,key,label,handle,counters,started+1800);rows.append(result)
                    if key==2350000 and label.endswith(('greedy','sampled')):targets[label]=(plan,trace)
        with (out/'replays.jsonl').open('w') as handle:
            for label,(plan,expected) in targets.items():
                _,trace=collect(env,plan,2350000,label,handle,counters,started+1800);assert trace==expected
        assert counters==dict(episodes=2500,days=70000)
        assert all(digest(p)==h for p,h in hashes.items()) and all(digest(out/p)==h for p,h in frozen.items())
        elapsed=time.perf_counter()-started;assert elapsed<=1800;result=analyze(rows)
        write_json(out/'summary.json',dict(status='complete',analysis=result,rows=rows,counters=counters,seconds=elapsed,
            test_scoring=False,archive_hashes={p.name:digest(p) for p in out.glob('*.pt')},
            trace_hashes={p.name:digest(p) for p in out.glob('*.jsonl')}))
        print(json.dumps(result),flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json',dict(error=str(exc),counters=counters));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
