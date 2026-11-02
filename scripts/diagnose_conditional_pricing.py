"""Predeclared split-future weekly pricing diagnostic; no policy fitting."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.parameter_policy_gradient import order_action
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.env.interface import JointAction
from src.experiments.research import make_research_env
from scripts.diagnose_counterfactual_rollouts import clone_environment
from scripts.audit_repeated_futures import snapshot, resample_future
from scripts.train_compact_residual import INPUTS, digest, serial, write_json

STUDY=Path('outputs/rl_discovery/parameter_confirmation_v1')


def select_split(values):
    values=np.asarray(values,float)
    if values.shape!=(4,5) or not np.isfinite(values).all():
        raise ValueError('four finite paired futures by five prices required')
    fit=values[:2].mean(0)
    chosen=max([4,0,1,2,3],key=lambda a:fit[a])
    return dict(chosen=chosen,selection_gain=float(fit[chosen]-fit[4]),
        transferred_gain=float(np.mean(values[2:,chosen]-values[2:,4])),
        action_mean_gains=(values-values[:,4,None]).mean(0).tolist(),
        action_gain_std=(values-values[:,4,None]).std(0,ddof=1).tolist(),
        hindsight_gain=float(np.max(values.mean(0))-values[:,4].mean()))


def analyze(rows):
    if [(r['episode'],r['day']) for r in rows]!=[(e,d) for e in range(16) for d in [1,8,15,22]]:
        raise ValueError('complete declared window/weekly-state grid required')
    x=np.array([r['transferred_gain'] for r in rows]).reshape(16,4).mean(1)
    draws=np.random.default_rng(20261028).integers(16,size=(10000,16))
    bounds=np.quantile(x[draws].mean(1),[.025,.975])
    discounts=sum(r['chosen']!=4 for r in rows)
    return dict(mean_transferred_discounted_gain=float(x.mean()),ci95=bounds.tolist(),
        selected_discount_states=int(discounts),promising_signal=bool(bounds[0]>0 and discounts>=8),
        mean_hindsight_gain=float(np.mean([r['hindsight_gain'] for r in rows])),
        interpretation='Split-future simulator-planning signal; not deployable policy gain or RL success.')


def advance(env,covers,first_day,sku,alternative,counters,deadline):
    if time.perf_counter()>deadline: raise TimeoutError('conditional pricing time cap')
    day=int(env.day); pobs=None; pa=None
    if env.pricing_due:
        pobs=env.pricing_observation().copy(); pa=np.full(10,4,dtype=int)
        if day==first_day: pa[sku]=alternative
        env.select_price(pa)
    obs=env.replenishment_observation().copy(); action=order_action(obs,day,covers)
    _,reward,done,info=env.step(JointAction(None,action)); counters['days']+=1
    return serial(dict(day=day,start=int(env.sim.start_day_index),obs=obs,actions=action,
        pricing_obs=pobs,price_actions=pa,reward=reward,info=info,
        demand_seed=int(env.sim.scenario_seed),supplier_seed=int(env.pipeline.scenario_seed))),done


def suffix(env,covers,sku,alternative,counters,deadline):
    first=int(env.day); rows=[]
    while True:
        row,done=advance(env,covers,first,sku,alternative,counters,deadline); rows.append(row)
        if done:return rows


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    assert json.loads((STUDY/'verification.json').read_text())['confirmed_gate']
    covers=np.array(json.loads((STUDY/'sku_seed211_final.json').read_text())['mean'])
    sources={str(p) for folder in ['src','scripts'] for p in Path(folder).rglob('*.py')}
    sources|={INPUTS['calibration'],str(STUDY/'sku_seed211_final.json'),'CONDITIONAL_PRICING_SIGNAL_PROTOCOL.md'}
    hashes={p:digest(p) for p in sorted(sources)}
    write_json(out/'manifest.json',dict(hashes=hashes,covers=covers,planned_days=25536,maximum_seconds=900))
    torch.set_num_threads(1);started=time.perf_counter();counters=dict(days=0);statistics=[]
    try:
        params=load_identifiable_training_calibration(INPUTS['calibration'])
        env=make_research_env('S2',split='train',start=15,params=params,pricing_scope='per_sku')
        with (out/'states.jsonl').open('w') as states,(out/'originals.jsonl').open('w') as originals:
            for ep,key in enumerate(range(2310000,2310016)):
                start=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
                env.sim.start_day_index=start;env.reset(seed=key);original=[];probes={}
                while not env._finished:
                    day=int(env.day)
                    if env.pricing_due:
                        sku=(ep*4+(day-1)//7)%10;before=snapshot(env)
                        original_keys=[int(env.sim.scenario_seed),int(env.pipeline.scenario_seed)]
                        probe=suffix(clone_environment(env),covers,sku,4,counters,started+900)
                        probes[day]=probe;draws=[];replica_replay=None;values=[]
                        for rep in range(4):
                            keys=[int(np.random.SeedSequence([2320000,ep,day,rep,s]).generate_state(1,dtype=np.uint64)[0]) for s in [0,1]]
                            branches=[];gains=[]
                            for a in range(5):
                                branch=resample_future(env,keys);initial=snapshot(branch)
                                trace=suffix(branch,covers,sku,a,counters,started+900)
                                value=float(sum(.99**i*r['reward'] for i,r in enumerate(trace)))
                                branches.append(dict(action=a,initial=initial,trace=trace,value=value));gains.append(value)
                            if rep==0:
                                replica_replay=suffix(resample_future(env,keys),covers,sku,4,counters,started+900)
                                assert replica_replay==branches[4]['trace']
                            draws.append(dict(keys=keys,branches=branches));values.append(gains)
                        assert snapshot(env)==before
                        assert [int(env.sim.scenario_seed),int(env.pipeline.scenario_seed)]==original_keys
                        result=dict(episode=ep,key=key,day=day,sku=sku,**select_split(values));statistics.append(result)
                        record=dict(episode=ep,key=key,day=day,sku=sku,state_before=before,state_after=snapshot(env),
                            original_keys=original_keys,probe=probe,draws=draws,replica0_replay=replica_replay,result=result)
                        states.write(json.dumps(record,allow_nan=False)+'\n');states.flush()
                    row,_=advance(env,covers,day,0,4,counters,started+900);original.append(row)
                for day,probe in probes.items():assert probe==original[day-1:]
                env.reset(seed=key);replay=suffix(env,covers,0,4,counters,started+900)
                assert replay==original
                originals.write(json.dumps(dict(episode=ep,key=key,start=start,trace=original,replay=replay),allow_nan=False)+'\n')
                originals.flush();print(f'conditional pricing: {ep+1}/16 windows',flush=True)
        assert counters['days']==25536 and all(digest(p)==h for p,h in hashes.items())
        elapsed=time.perf_counter()-started;assert elapsed<=900
        result=analyze(statistics)
        write_json(out/'summary.json',dict(status='complete',statistics=statistics,analysis=result,days=counters['days'],
            resampled_suffixes=1280,original_episodes=16,full_replays=16,suffix_replays=64,original_probes=64,
            seconds=elapsed,test_scoring=False,trace_hashes={p.name:digest(p) for p in out.glob('*.jsonl')}))
        print(json.dumps(result),flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json',dict(error=str(exc),counters=counters));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
