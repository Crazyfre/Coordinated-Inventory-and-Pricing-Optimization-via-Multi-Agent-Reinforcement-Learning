"""Replay existing train-only labels and fit one bounded structured imitator."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.structured_imitation import position_features,fit_intervals,InventoryPositionLinearPolicy
from src.algos.imitation_ordering import teacher_indices,ImitationAssistedMAPPO
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.env.interface import JointAction
from src.experiments.research import make_research_env
from scripts.train_clean_research import TrainingWindows
from scripts.diagnose_teacher_ordering import measure,compare
from scripts.diagnose_state_aggregation import acceptance,agreement


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(env, groups, labels, deadline):
    obs,targets=[],[]
    episodes=0
    for label in labels:
        for (group_label,seed),rows in sorted(groups.items()):
            if group_label!=label:
                continue
            assert [r['day'] for r in rows]==list(range(1,29))
            env.reset(seed=seed);episodes+=1
            for row in rows:
                if time.perf_counter()>deadline:
                    raise TimeoutError('structured representation deadline')
                if env.pricing_due:
                    env.select_price(4)
                observation=env.replenishment_observation()
                target=teacher_indices(torch.as_tensor(observation),env.n,env.obs_pipe_horizon).numpy()
                np.testing.assert_array_equal(target,row['teacher'])
                obs.append(observation.copy());targets.append(target.copy())
                _,reward,done,info=env.step(JointAction(None,np.array(row['actions'],dtype=np.int64)))
                for key in ['demand','sales','price']:
                    np.testing.assert_array_equal(info[key],row[key])
                assert reward==row['reward'] and done==(row['day']==28)
    return np.asarray(obs),np.asarray(targets),episodes


def run(output, source='outputs/state_aggregation/seed11_v1'):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();deadline=started+600
    source=Path(source)
    old=json.loads((source/'protocol.json').read_text())
    calibration=old['calibration']
    protocol=dict(variant=InventoryPositionLinearPolicy.variant,training_source=str(source),
        source_sha256={name:digest(source/name) for name in ['protocol.json','traces.jsonl','learner_states.pt']},
        runner_sha256=digest(__file__),actor_sha256=digest('src/algos/structured_imitation.py'),
        calibration=calibration,calibration_sha256=digest(calibration),seed=11,
        scenario='S2',days=28,pricing_scope='shared_fixed_fixture',price_multiplier=1.2,
        capacity=500,billing='accepted_delivered_units',max_seconds=600,solver_max_seconds=60,
        validation_seeds=list(range(50000,50020)),validation_days=[1886,1913],
        held_out_test_used=False,new_teacher_collection_episodes=0,rl_updates=0,
        fitting='three shared coefficients; max-margin clipped-ceil action bins; training labels only',
        coefficient_bounds=[[-20,20],[-20,20],[-2,2]],margin_bounds=[0,.49],
        gate=dict(training_holdout_agreement=.99,own_state_agreement=.99,
                  teacher_reward_ci_lower_min=-1,teacher_fill_ci_lower_min=-.005,
                  initial_reward_ci_lower_strict_min=0,initial_fill_ci_lower_min=-.005))
    if protocol['calibration_sha256']!=old['calibration_sha256']:
        raise ValueError('source calibration mismatch')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    try:
        torch.set_num_threads(1);torch.manual_seed(11)
        params=load_identifiable_training_calibration(calibration)
        train=TrainingWindows(make_research_env('S2',split='train',start=15,params=params,pricing_scope='shared'),28)
        validation=make_research_env('S2',split='validation',params=params,pricing_scope='shared')
        allowed=['warmup_teacher']+[f'learner_states_collection_{i}' for i in range(1,5)]+['heldout_train_teacher']
        groups={}
        for line in (source/'traces.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['label'] in allowed:
                groups.setdefault((row['label'],row['seed']),[]).append(row)
        x,y,train_episodes=reconstruct(train,groups,allowed[:-1],deadline)
        hx,hy,holdout_episodes=reconstruct(train,groups,allowed[-1:],deadline)
        assert (train_episodes,holdout_episodes,len(x),len(hx))==(320,16,8960,448)
        assert all(15<=start and start+27<=1885 for start in train.windows)
        np.savez_compressed(out/'reconstructed_data.npz',train_obs=x,train_labels=y,holdout_obs=hx,holdout_labels=hy)
        print('Replayed 336 training-window episodes exactly; fitting only 8960 training states.',flush=True)
        fitted=fit_intervals(position_features(x,train.n,train.obs_pipe_horizon),y)
        policy=InventoryPositionLinearPolicy(train.n,train.obs_pipe_horizon,fitted['coefficients'])
        initial=InventoryPositionLinearPolicy(train.n,train.obs_pipe_horizon,[0,0,0])
        policy.save(out/'structured_policy.json')
        raw=ImitationAssistedMAPPO(validation,normalize_inputs=True,target_kl=.02)
        raw.load(source/'learner_states.pt')
        accuracy=dict(structured_train=float(np.mean(policy.predict(x)==y)),
                      structured_holdout=float(np.mean(policy.predict(hx)==hy)),
                      raw_train=agreement(raw.replenishment,torch.tensor(x),torch.tensor(y)),
                      raw_holdout=agreement(raw.replenishment,torch.tensor(hx),torch.tensor(hy)))
        print(json.dumps(dict(fit=fitted,agreement=accuracy)),flush=True)
        rows_by_label,traces_by_label={},{}
        for label,actor in [('initial',initial),('structured',policy),('raw_actor',raw.replenishment),
                            ('teacher',None),('constant','constant'),('ss','ss')]:
            rows,traces,_,_=measure(validation,actor,protocol['validation_seeds'],label,
                deadline=deadline,trace_path=out/'validation_traces.jsonl')
            rows_by_label[label],traces_by_label[label]=rows,traces
        restored=InventoryPositionLinearPolicy.load(out/'structured_policy.json',train.n,train.obs_pipe_horizon)
        _,replay,_,_=measure(validation,restored,protocol['validation_seeds'],'replay',
                             deadline=deadline,trace_path=out/'validation_traces.jsonl')
        assert [dict(r,label='structured') for r in replay]==traces_by_label['structured']
        reference=[r['demand'] for r in traces_by_label['teacher']]
        assert all([r['demand'] for r in traces]==reference for traces in traces_by_label.values())
        gate=acceptance(accuracy['structured_holdout'],rows_by_label['structured'],rows_by_label['teacher'],rows_by_label['initial'])
        summary=dict(protocol=protocol,fit=fitted,agreement=accuracy,gate=gate,
            comparisons={ref:compare(rows_by_label['structured'],rows_by_label[ref]) for ref in rows_by_label if ref!='structured'},
            means={label:{k:float(np.mean([r[k] for r in rows])) for k in ['reward_day','fill','teacher_agreement']}
                   for label,rows in rows_by_label.items()},episode_rows=sum(rows_by_label.values(),[]),
            reconstruction_episodes=336,reconstruction_days=9408,validation_episodes=120,replay_episodes=20,
            total_simulated_days=13328,training_windows=train.windows,
            reconstruction_verified=True,checkpoint_replay_verified=True,paired_demand_verified=True,
            data_sha256=digest(out/'reconstructed_data.npz'),checkpoint_sha256=digest(out/'structured_policy.json'),
            seconds=time.perf_counter()-started,held_out_test_used=False,rl_updates=0)
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        print(json.dumps({k:summary[k] for k in ['fit','agreement','gate','means','seconds']},indent=2),flush=True)
        return summary
    except Exception as exc:
        (out/'failed_status.json').write_text(json.dumps(dict(status='failed_partial',error=str(exc),held_out_test_used=False)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.output)
