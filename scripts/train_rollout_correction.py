"""One predeclared reward-regression policy improvement pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scripts.diagnose_counterfactual_rollouts import clone_environment, decision, advance, suffix, check_branches
from scripts.diagnose_cover_search import CoverPolicy
from scripts.diagnose_teacher_ordering import compare
from src.algos.rollout_correction import RolloutCorrection, features, fit_ridge
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env
from src.experiments.phase4 import heuristic


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();deadline=started+600
    paths=dict(center='outputs/structured_imitation/seed11_v1/structured_policy.json',
               selection='outputs/cover_search/bounded_v1/selection.json',
               calibration='outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json',
               protocol='ROLLOUT_LEARNING_PROTOCOL.md',runner=__file__,model='src/algos/rollout_correction.py')
    # Include imported simulation and branch code in provenance.
    for p in sorted(Path('src').rglob('*.py')):
        paths[str(p)]=str(p)
    for p in ['scripts/diagnose_counterfactual_rollouts.py','scripts/diagnose_cover_search.py',
              'scripts/diagnose_residual_support.py','scripts/diagnose_teacher_ordering.py']:
        paths[p]=p
    hashes={k:digest(p) for k,p in paths.items()}
    (out/'manifest.json').write_text(json.dumps(dict(paths=paths,hashes=hashes),indent=2))
    torch.set_num_threads(1)
    params=load_identifiable_training_calibration(paths['calibration'])
    env=make_research_env('S2',split='train',start=15,params=params,pricing_scope='shared')
    center=InventoryPositionLinearPolicy.load(paths['center'],env.n,env.obs_pipe_horizon)
    target=CoverPolicy(env.n,env.obs_pipe_horizon,json.loads(Path(paths['selection']).read_text())['selected']['cover'])
    initial=RolloutCorrection(center,target,env.n,env.obs_pipe_horizon)
    records=[];originals=[];xs=[];ys=[]
    for ep,seed in enumerate(range(930000,930064)):
        start=int(np.random.default_rng(np.random.SeedSequence([seed,314159])).integers(15,1859))
        env.sim.start_day_index=start;env.reset(seed=seed);original=[];episode=[]
        while not env._finished:
            c,action=decision(env,center,target);day=env.day
            obs=env.replenishment_observation()
            np.testing.assert_array_equal(initial.action(obs,day),action)
            if day in (1,8,15,22):
                sku=(ep*4+(day-1)//7)%env.n
                branches=[]
                for a in range(max(0,int(c[sku])-2),min(40,int(c[sku])+2)+1):
                    first=action.copy();first[sku]=a
                    trace=suffix(clone_environment(env),center,target,first,deadline)
                    value=float(np.dot(.99**np.arange(len(trace)),[r['reward'] for r in trace]))
                    branches.append(dict(action=a,trace=trace,value=value))
                ref=next(b for b in branches if b['action']==int(action[sku]))
                check_branches(branches,ref['trace'])
                for b in branches:
                    x=features(obs,env.n,env.obs_pipe_horizon,sku,b['action']-int(action[sku]),day)
                    xs.append(x);ys.append(b['value']-ref['value'])
                episode.append(dict(seed=seed,start=start,day=day,sku=sku,obs=obs.tolist(),
                                    center=int(c[sku]),reference_action=int(action[sku]),branches=branches))
            row,_=advance(env,action);original.append(row)
        for rec in episode:
            ref=next(b for b in rec['branches'] if b['action']==rec['reference_action'])
            assert ref['trace']==original[rec['day']-1:]
        records.extend(episode);originals.append(dict(seed=seed,start=start,trace=original))
        if (ep+1)%8==0:print(f'Collected {ep+1}/64 training scenarios',flush=True)
    x,y=np.asarray(xs),np.asarray(ys)
    weights,scale=fit_ridge(x,y)
    np.savez_compressed(out/'fit_data.npz',x=x,y=y)
    checkpoint=dict(schema='reward_rollout_ridge_v1',weights=weights.tolist(),scale=scale.tolist(),
                    n=env.n,horizon=env.obs_pipe_horizon,threshold=1.,gamma=.99,ridge=10.,
                    center_path=paths['center'],center_hash=hashes['center'],cover=target.cover,
                    initial_weights=np.zeros(12).tolist(),policy='deterministic single-SKU correction')
    (out/'checkpoint.json').write_text(json.dumps(checkpoint,indent=2))
    checkpoint_hash=digest(out/'checkpoint.json')
    (out/'training_traces.json').write_text(json.dumps(dict(records=records,originals=originals)))
    print('Reward model frozen; beginning validation',flush=True)
    val=make_research_env('S2',split='validation',params=params,pricing_scope='shared')
    learned=RolloutCorrection(center,target,env.n,env.obs_pipe_horizon,weights,scale)
    restored=json.loads((out/'checkpoint.json').read_text())
    replay=RolloutCorrection(center,target,env.n,env.obs_pipe_horizon,restored['weights'],restored['scale'])
    rows={};traces={};changed=0
    for label,actor in [('learned',learned),('initial',initial),('teacher',CoverPolicy(env.n,env.obs_pipe_horizon,6)),
                        ('cover7.5',target),('constant','constant'),('ss','ss'),('replay',replay)]:
        rows[label]=[];traces[label]=[]
        for seed in range(940000,940040):
            val.reset(seed=seed);trace=[]
            while not val._finished:
                if time.perf_counter()>deadline:raise TimeoutError('600-second campaign budget')
                _,reference=decision(val,center,target)
                obs=val.replenishment_observation()
                if isinstance(actor,RolloutCorrection):action=actor.action(obs,val.day)
                elif isinstance(actor,str):action=heuristic(val,kind=actor)
                else:action=actor.act(obs)[0]
                if label=='learned':changed+=int(np.any(action!=reference))
                row,_=advance(val,action);row['seed']=seed;trace.append(row)
            rows[label].append(dict(seed=seed,reward_day=sum(r['reward'] for r in trace)/28,
                                    fill=sum(sum(r['sales']) for r in trace)/sum(sum(r['demand']) for r in trace)))
            traces[label].extend(trace)
        print(f'Validated {label}',flush=True)
    assert traces['replay']==traces['learned']
    assert all([r['demand'] for r in tr]==[r['demand'] for r in traces['initial']] for tr in traces.values())
    comparisons={label:compare(rows['learned'],rows[label]) for label in ['initial','teacher','cover7.5','constant','ss']}
    gates={k:bool(v['reward_ci95'][0]>0 and v['fill_ci95'][0]>=-.005) for k,v in comparisons.items()}
    assert digest(out/'checkpoint.json')==checkpoint_hash
    assert {k:digest(p) for k,p in paths.items()}==hashes
    (out/'validation_traces.json').write_text(json.dumps(traces))
    summary=dict(means={k:{m:float(np.mean([r[m] for r in rs])) for m in ['reward_day','fill']} for k,rs in rows.items()},
                 comparisons=comparisons,gates=gates,all_comparators_gate=all(gates.values()),episode_rows=rows,
                 changed_days=changed,evaluated_days=1120,fit_samples=len(y),training_states=len(records),
                 training_branch_days=sum(len(b['trace']) for r in records for b in r['branches']),
                 original_days=1792,validation_days=6720,replay_days=1120,
                 training_rmse=float(np.sqrt(np.mean((x/scale@weights-y)**2))),
                 checkpoint_sha256=checkpoint_hash,seconds=time.perf_counter()-started,
                 ppo_updates=0,reward_regression_fits=1,held_out_test_used=False,
                 interpretation='Single fixed model-based approximate policy-improvement pilot; not PPO or coordination evidence.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:summary[k] for k in ['means','comparisons','gates','changed_days','seconds']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
