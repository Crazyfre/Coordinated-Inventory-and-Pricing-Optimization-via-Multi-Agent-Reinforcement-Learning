"""Fixed target representability and nearest-support controller diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scripts.diagnose_cover_search import CoverPolicy
from scripts.diagnose_teacher_ordering import compare
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.algos.structured_residual import StructuredResidualActor
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env
from src.env.interface import JointAction


def project_target(center,target):
    center,target=np.asarray(center),np.asarray(target)
    if center.shape!=target.shape or not np.isin(center,np.arange(41)).all() or not np.isin(target,np.arange(41)).all():
        raise ValueError('matching integer grid indices0..40 required')
    return np.clip(target,np.maximum(center-2,0),np.minimum(center+2,40)).astype(np.int64)


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();deadline=start+300
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    paths=dict(center='outputs/structured_imitation/seed11_v1/structured_policy.json',
               selection='outputs/cover_search/bounded_v1/selection.json',
               calibration='outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json')
    cover=json.loads(Path(paths['selection']).read_text())['selected']['cover']
    protocol=dict(variant='frozen_cover_residual_support_audit_v1',paths=paths,
        hashes={k:digest(v) for k,v in paths.items()},source_sha256=digest(__file__),
        cover=cover,seeds=list(range(80000,80040)),scenario='S2',validation_days=[1886,1913],days=28,
        fixed_price=1.2,capacity=500,billing='accepted_delivered_units',offset_units=[-10,-5,0,5,10],
        controllers=['center','target','projected'],projection='nearest physical target action within frozen-center support',
        gate='paired reward lower95%>0 and fill lower95%>=-.005; no selection or retuning',
        max_seconds=300,held_out_test_used=False,rl_updates=0,
        interpretation='action-support diagnostic; projected target is feasible but not optimal within support')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    try:
        torch.set_num_threads(1)
        params=load_identifiable_training_calibration(paths['calibration'])
        env=make_research_env('S2',split='validation',params=params,pricing_scope='shared')
        center=InventoryPositionLinearPolicy.load(paths['center'],env.n,env.obs_pipe_horizon)
        target=CoverPolicy(env.n,env.obs_pipe_horizon,cover)
        obs,_=env.reset(seed=0)
        support_actor=StructuredResidualActor(len(obs['replenishment']),env.n,env.obs_pipe_horizon,center.coefficients)
        records=[];episodes=[];label_traces={}
        for label in protocol['controllers']+['replay']:
            label_traces[label]=[]
            for seed in protocol['seeds']:
                env.reset(seed=seed);done=False;reward_sum=sales=demand=0.;day=0
                while not done:
                    if time.perf_counter()>deadline:raise TimeoutError('support audit deadline')
                    if env.pricing_due:env.select_price(4)
                    obs=env.replenishment_observation()
                    c=center.predict(obs);t=target.act(obs)[0];p=project_target(c,t)
                    # Confirm center and physical support agree with actual PPO actor.
                    with torch.no_grad():
                        x=torch.tensor(obs).unsqueeze(0)
                        np.testing.assert_array_equal(support_actor.centers(x)[0],c)
                        ds=support_actor(x)
                        for i,dist in enumerate(ds):
                            np.testing.assert_array_equal(torch.where(dist.probs[0]>0)[0],np.arange(max(0,c[i]-2),min(40,c[i]+2)+1))
                    action=c if label=='center' else t if label=='target' else p
                    _,reward,done,info=env.step(JointAction(None,action))
                    day+=1;reward_sum+=reward;sales+=info['sales'].sum();demand+=info['demand'].sum()
                    row=dict(label=label,seed=seed,day=day,center=c.tolist(),target=t.tolist(),projected=p.tolist(),
                        outside=(t!=p).tolist(),unreachable_units=(5*np.abs(t-p)).tolist(),
                        actions=action.tolist(),reward=float(reward),demand=info['demand'].tolist(),sales=info['sales'].tolist())
                    records.append(row);label_traces[label].append(row)
                episodes.append(dict(label=label,seed=seed,reward_day=float(reward_sum/28),fill=float(sales/demand)))
            (out/'traces.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            print('Completed '+label,flush=True)
        assert label_traces['projected']==[dict(r,label='projected') for r in label_traces['replay']]
        assert all([r['demand'] for r in ts]==[r['demand'] for r in label_traces['center']] for ts in label_traces.values())
        by_label={l:[r for r in episodes if r['label']==l] for l in protocol['controllers']}
        comparisons={f'{a}_minus_{b}':compare(by_label[a],by_label[b]) for a,b in
                     [('target','center'),('projected','center'),('target','projected')]}
        coverage={}
        for label in protocol['controllers']:
            rs=label_traces[label];outside=np.asarray([r['outside'] for r in rs]);units=np.asarray([r['unreachable_units'] for r in rs])
            coverage[label]=dict(sku_decisions=int(outside.size),outside_count=int(outside.sum()),
                outside_fraction=float(outside.mean()),days_any_outside=int(outside.any(axis=1).sum()),
                per_sku_outside_fraction=outside.mean(axis=0).tolist(),
                mean_unreachable_units_per_decision=float(units.mean()),max_unreachable_units=int(units.max()))
        summary=dict(protocol=protocol,episode_rows=episodes,comparisons=comparisons,coverage=coverage,
            sku_ids=env.sim.series_ids,
            means={l:{k:float(np.mean([r[k] for r in rows])) for k in ['reward_day','fill']} for l,rows in by_label.items()},
            gates={key:bool(r['reward_ci95'][0]>0 and r['fill_ci95'][0]>=-.005) for key,r in comparisons.items()},
            scored_episodes=120,replay_episodes=40,trace_days=len(records),seconds=time.perf_counter()-start,
            actual_actor_support_verified=True,paired_demands_verified=True,projection_replay_verified=True,
            held_out_test_used=False,rl_updates=0)
        assert all(digest(v)==protocol['hashes'][k] for k,v in paths.items())
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        print(json.dumps({k:summary[k] for k in ['means','comparisons','coverage','seconds']},indent=2))
    except Exception as exc:
        (out/'failed_status.json').write_text(json.dumps(dict(status='failed_partial',error=str(exc))))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);run(p.parse_args().output)
