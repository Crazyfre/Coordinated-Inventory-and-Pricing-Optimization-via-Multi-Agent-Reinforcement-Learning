"""One powered prediction comparison with strictly conditional validation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.stats import t
from scripts.final_rollout_collection import collect
from scripts.compare_averaged_targets import matrices,assess
from scripts.train_rollout_correction import digest
from scripts.diagnose_counterfactual_rollouts import decision,advance
from scripts.diagnose_cover_search import CoverPolicy
from scripts.diagnose_teacher_ordering import compare
from src.algos.rollout_correction import RolloutCorrection,fit_ridge
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env
from src.experiments.phase4 import heuristic


def prediction_gates(result):
    comparisons={}
    a=np.asarray(result['models']['averaged']['episode_mse'])
    for label in ['single','zero']:
        delta=a-np.asarray(result['models'][label]['episode_mse'])
        half=float(t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta)))
        comparisons[label]=dict(delta=float(delta.mean()),ci95=[float(delta.mean()-half),float(delta.mean()+half)],passed=bool(delta.mean()+half<0))
    return dict(comparisons=comparisons,passed=all(r['passed'] for r in comparisons.values()))


def validate(out,params,center,target,old_model,checkpoint,deadline):
    env=make_research_env('S2',split='validation',params=params)
    def restored():
        ck=json.loads(checkpoint.read_text())
        return RolloutCorrection(center,target,env.n,env.obs_pipe_horizon,ck['weights'],ck['scale'])
    actors=[('averaged',restored()),('initial',RolloutCorrection(center,target,env.n,env.obs_pipe_horizon)),
            ('old_ridge',old_model),('teacher',CoverPolicy(env.n,env.obs_pipe_horizon,6)),
            ('cover7.5',target),('constant','constant'),('ss','ss'),('replay',restored())]
    traces={};rows={}
    for label,actor in actors:
        traces[label]=[];rows[label]=[]
        for seed in range(984000,984080):
            env.reset(seed=seed);episode=[]
            while not env._finished:
                if time.perf_counter()>deadline:raise TimeoutError('3600-second campaign limit')
                decision(env,center,target);obs=env.replenishment_observation()
                action=actor.action(obs,env.day) if isinstance(actor,RolloutCorrection) else heuristic(env,kind=actor) if isinstance(actor,str) else actor.act(obs)[0]
                row,_=advance(env,action);row['seed']=seed;episode.append(row)
            rows[label].append(dict(seed=seed,reward_day=sum(r['reward'] for r in episode)/28,
                fill=sum(sum(r['sales']) for r in episode)/sum(sum(r['demand']) for r in episode)))
            traces[label].extend(episode)
        (out/('validation_'+label+'.json')).write_text(json.dumps(traces[label]));print('Validated '+label,flush=True)
    assert traces['averaged']==traces['replay']
    assert all([r['demand'] for r in tr]==[r['demand'] for r in traces['initial']] for tr in traces.values())
    comparisons={label:compare(rows['averaged'],rows[label]) for label in ['initial','old_ridge','teacher','cover7.5','constant','ss']}
    gates={k:bool(v['reward_ci95'][0]>0 and v['fill_ci95'][0]>=-.005) for k,v in comparisons.items()}
    return dict(status='completed',rows=rows,comparisons=comparisons,gates=gates,passed=all(gates.values()),episodes=560,replay_episodes=80)


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();deadline=started+3600
    old=json.loads(Path('outputs/rollout_learning/ridge_v1/manifest.json').read_text())
    for k,p in old['paths'].items():assert digest(p)==old['hashes'][k]
    paths=dict(old['paths']);paths.update(runner=__file__,protocol='FINAL_ROLLOUT_PROTOCOL.md',
        budget='outputs/final_rollout_study/budget_plan.json',planner='scripts/plan_final_rollout_study.py',
        collector='scripts/final_rollout_collection.py',comparison='scripts/compare_averaged_targets.py',
        resampling='scripts/audit_repeated_futures.py',rollout='scripts/audit_rollout_predictions.py',
        checkpoint='outputs/rollout_learning/ridge_v1/checkpoint.json')
    hashes={k:digest(p) for k,p in paths.items()};(out/'manifest.json').write_text(json.dumps(dict(paths=paths,hashes=hashes),indent=2))
    ck=json.loads(Path(paths['checkpoint']).read_text());torch.set_num_threads(1)
    params=load_identifiable_training_calibration(paths['calibration']);env=make_research_env('S2',split='train',start=15,params=params)
    center=InventoryPositionLinearPolicy.load(paths['center'],env.n,env.obs_pipe_horizon);target=CoverPolicy(env.n,env.obs_pipe_horizon,ck['cover'])
    old_model=RolloutCorrection(center,target,env.n,env.obs_pipe_horizon,ck['weights'],ck['scale'])
    file_hashes={};counts={};models={};checkpoint_hashes={}
    for phase,seed0,reps,namespace in [('fit',980000,8,982100),('assessment',981000,32,983100)]:
        chunks=[];metadata=[];days=branches=0
        for ep in range(64):
            data=collect(env,center,target,old_model,seed0+ep,1,reps,namespace,deadline,phase,episode_offset=ep)
            path=out/f'{phase}_{ep:03d}.json';path.write_text(json.dumps(data));file_hashes[path.name]=digest(path)
            rs=data['records'];chunks.append(matrices(rs,env.n,env.obs_pipe_horizon));metadata.extend([dict(actions=r['actions']) for r in rs])
            branches+=sum(len(d['branches']) for r in rs for d in r['draws']);days+=sum(len(b['trace']) for r in rs for d in r['draws'] for b in d['branches'])
            (out/'progress.json').write_text(json.dumps(dict(phase=phase,completed=ep+1,total=64,seconds=time.perf_counter()-started)))
            if (ep+1)%4==0:print(f'{phase}: {ep+1}/64 complete',flush=True)
        x,single,averaged,noise,mask=[np.concatenate([c[i] for c in chunks]) for i in range(5)]
        counts[phase]=dict(states=128,branches=branches,branch_days=days)
        if phase=='fit':
            np.savez_compressed(out/'fit_data.npz',x=x,single=single,averaged=averaged)
            for label,y in [('single',single),('averaged',averaged)]:
                w,scale=fit_ridge(x,y);models[label]=dict(schema='final_reward_ridge_v1',weights=w.tolist(),scale=scale.tolist(),n=env.n,horizon=env.obs_pipe_horizon,
                    reference_checkpoint_sha256=hashes['checkpoint'],future_count=1 if label=='single' else 8,threshold=1.,cover=ck['cover'])
                path=out/(label+'_checkpoint.json');path.write_text(json.dumps(models[label],indent=2));checkpoint_hashes[label]=digest(path)
            (out/'frozen_fit.json').write_text(json.dumps(dict(checkpoint_hashes=checkpoint_hashes,assessment_started=False),indent=2))
            print('Both checkpoints frozen before independent assessment',flush=True)
        else:
            predictions={label:x/np.asarray(m['scale'])@np.asarray(m['weights']) for label,m in models.items()};predictions['zero']=np.zeros(len(averaged))
            np.savez_compressed(out/'assessment_data.npz',x=x,y=averaged,noise=noise,mask=mask,**predictions)
            result=assess(metadata,predictions,averaged,noise,mask);gates=prediction_gates(result)
            (out/'prediction_result.json').write_text(json.dumps(dict(result=result,gates=gates),indent=2))
    validation=validate(out,params,center,target,old_model,out/'averaged_checkpoint.json',deadline) if gates['passed'] else dict(status='not_run_prediction_gate_failed',episodes=0,replay_episodes=0)
    assert {k:digest(p) for k,p in paths.items()}==hashes
    assert {label:digest(out/(label+'_checkpoint.json')) for label in models}==checkpoint_hashes
    (out/'trace_hashes.json').write_text(json.dumps(file_hashes,indent=2))
    summary=dict(prediction=result,prediction_gates=gates,validation=validation,counts=counts,checkpoint_hashes=checkpoint_hashes,
                 original_days=3584,probe_days=4480,replay_days=4480,seconds=time.perf_counter()-started,fits=2,held_out_test_used=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(dict(prediction_gates=gates,validation_status=validation['status'],counts=counts,seconds=summary['seconds']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
