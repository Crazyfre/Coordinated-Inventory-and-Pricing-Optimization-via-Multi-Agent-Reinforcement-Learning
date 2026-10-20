"""Predeclared matched-budget joint versus independently trained role pilot."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.joint_parameter_policy import JointParameterGradient,initial_parameters,price_action,ordering_action
from src.env.interface import JointAction
from src.experiments.research import make_research_env
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.eval.parameter_review import episode_metrics
from scripts.train_clean_research import TrainingWindows
from scripts.train_compact_residual import INPUTS,digest,write_json,serial


def collect(env,p,key,label,handle,counters,deadline):
    env.reset(seed=key);trace=[]
    for day in range(1,29):
        if time.perf_counter()>deadline:raise TimeoutError('pilot time cap reached')
        pricing_obs=None;price_indices=None
        if env.pricing_due:
            pricing_obs=env.pricing_observation().copy()
            price_indices=price_action(pricing_obs,day,p);env.select_price(price_indices)
        obs=env.replenishment_observation().copy();actions=ordering_action(obs,day,p)
        _,reward,done,info=env.step(JointAction(None,actions))
        counters['days']+=1
        assert done==(day==28) and info['total_on_hand']<=500+1e-8
        row=serial(dict(label=label,scenario=key,start=int(env.sim.start_day_index),day=day,
            pricing_obs=pricing_obs,price_actions=price_indices,obs=obs,actions=actions,reward=reward,info=info,
            demand_seed=int(env.sim.scenario_seed),supplier_seed=int(env.pipeline.scenario_seed)))
        handle.write(json.dumps(row,allow_nan=False)+'\n');trace.append(row)
    handle.flush();counters['episodes']+=1
    result=episode_metrics(trace)
    result['discounted_day']=sum(.99**(r['day']-1)*r['reward'] for r in trace)/28
    result['price_counts']=np.bincount(np.concatenate([r['price_actions'] for r in trace if r['price_actions'] is not None]),minlength=5).tolist()
    return result,trace


def analyze(rows):
    index={(r['label'],r['scenario']):r for r in rows};keys=list(range(1710000,1710064))
    assert len(index)==len(rows)==384
    effects={};eligible=True
    rng=np.random.default_rng(20261022);draws=rng.integers(0,64,size=(10000,64))
    for ref in ['start','independent']:
        for metric in ['reward_day','fill']:
            x=np.array([index['joint',k][metric]-index[ref,k][metric] for k in keys])
            effects[f'{ref}_{metric}']=dict(mean=float(x.mean()),ci9875=np.quantile(x[draws].mean(1),[.00625,.99375]).tolist())
            eligible &= bool(x.mean()>=.5 if metric=='reward_day' else x.mean()>=-.005)
    means={l:{m:float(np.mean([index[l,k][m] for k in keys])) for m in ['reward_day','fill']}
           for l in ['joint','independent','ordering','pricing','start','full7_5']}
    counts={l:np.sum([index[l,k]['price_counts'] for k in keys],axis=0).tolist() for l in means}
    return dict(effects=effects,means=means,price_counts=counts,eligible_for_new_confirmation=bool(eligible),
                interpretation='Single-seed development selection, not coordination confirmation')


def run(output,study):
    out,study=Path(output),Path(study);out.mkdir(parents=True,exist_ok=False)
    assert json.loads((study/'verification.json').read_text())['confirmed_gate']
    covers=np.array(json.loads((study/'sku_seed211_final.json').read_text())['mean'])
    sources=['JOINT_PARAMETER_PILOT_PROTOCOL.md','src/algos/joint_parameter_policy.py',__file__,
        'scripts/verify_joint_parameter_pilot.py','src/eval/parameter_review.py',
        'src/algos/parameter_policy_gradient.py','src/experiments/research.py','src/experiments/phase4.py',
        'src/env/joint_env.py','src/env/replenishment_env.py','src/env/pipeline.py','scripts/train_clean_research.py',
        INPUTS['calibration'],str(study/'sku_seed211_final.json'),str(study/'verification.json')]
    hashes={p:digest(p) for p in sources}
    write_json(out/'manifest.json',dict(hashes=hashes,initial_covers=covers.tolist(),seed=501,
        updates=dict(joint=24,ordering=12,pricing=12),pairs=8,max_seconds=900,test_scoring=False))
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+900;counters=dict(episodes=0,days=0)
    stage='training'
    try:
        calibration=load_identifiable_training_calibration(INPUTS['calibration'])
        env=TrainingWindows(make_research_env('S2',split='train',start=15,days=28,params=calibration,pricing_scope='per_sku'),28)
        finals={}
        with (out/'training_traces.jsonl').open('w') as handle:
            for arm,updates,first in [('joint',24,1700000),('ordering',12,1701000),('pricing',12,1702000)]:
                learner=JointParameterGradient(covers,arm,501);history=[]
                for u in range(updates):
                    e=learner.perturbations();before=learner.mean.copy();records=[];diff=[]
                    for pair in range(8):
                        key=first+u*8+pair;pair_rows={};noise=[]
                        for sign,direction in [('plus',1),('minus',-1)]:
                            result,trace=collect(env,before+direction*learner.sigma*e[pair],key,
                                f'{arm}_u{u}_p{pair}_{sign}',handle,counters,deadline)
                            pair_rows[sign]=result;noise.append([(r['demand_seed'],r['supplier_seed']) for r in trace])
                        assert noise[0]==noise[1]
                        diff.append(pair_rows['plus']['discounted_day']-pair_rows['minus']['discounted_day'])
                        records.append(dict(key=key,**pair_rows))
                    gradient=learner.update(e,diff)
                    history.append(dict(update=u+1,mean_before=before,epsilon=e,differences=diff,gradient=gradient,
                        records=records,state=learner.state()))
                    write_json(out/f'{arm}_history.json',history)
                    if (u+1)%4==0:print(f'{arm}: {u+1}/{updates} updates',flush=True)
                finals[arm]=learner.mean.copy();write_json(out/f'{arm}_final.json',learner.state())
        stage='assessment';frozen={a:digest(out/f'{a}_final.json') for a in finals}
        write_json(out/'frozen_checkpoints.json',frozen)
        combined=finals['ordering'].copy();combined[10:]=finals['pricing'][10:]
        policies=dict(finals,independent=combined,start=initial_parameters(covers),full7_5=initial_parameters(np.full(10,7.5)))
        rows=[];targets={}
        with (out/'assessment_traces.jsonl').open('w') as handle:
            for key in range(1710000,1710064):
                noise=None
                for label,p in policies.items():
                    result,trace=collect(env,p,key,label,handle,counters,deadline)
                    current=[(r['demand_seed'],r['supplier_seed']) for r in trace]
                    if noise is None:noise=current
                    assert noise==current
                    rows.append(result)
                    if key==1710000 and label in ['joint','start']:targets[label]=trace
                write_json(out/'progress.json',dict(stage=stage,scenarios=key-1710000+1,counters=counters))
        with (out/'replays.jsonl').open('w') as handle:
            for label,expected in targets.items():
                _,actual=collect(env,policies[label],1710000,label,handle,counters,deadline)
                assert actual==expected
        assert counters==dict(episodes=1154,days=32312)
        for path,h in hashes.items():assert digest(path)==h
        for arm,h in frozen.items():assert digest(out/f'{arm}_final.json')==h
        assert time.perf_counter()<=deadline
        result=analyze(rows)
        write_json(out/'summary.json',dict(status='complete',analysis=result,rows=rows,counters=counters,
            seconds=time.perf_counter()-started,exact_replays=2,test_scoring=False,
            trace_hashes={p.name:digest(p) for p in out.glob('*.jsonl')}))
        print(json.dumps(result),flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json',dict(stage=stage,error=str(exc),counters=counters,seconds=time.perf_counter()-started))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--study',required=True)
    a=p.parse_args();run(a.output,a.study)
