"""Pre-gated structured-center residual PPO, one seed and 64 updates maximum."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.structured_residual import StructuredResidualMAPPO
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env
from scripts.train_clean_research import TrainingWindows
from scripts.diagnose_teacher_ordering import measure,compare


def equivalence(rows,teacher):
    difference=compare(rows,teacher)
    return dict(passed=bool(np.mean([r['teacher_agreement'] for r in rows])>=.99
                and difference['reward_ci95'][0]>=-1 and difference['fill_ci95'][0]>=-.005),
                comparison=difference)


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();deadline=start+570
    center=Path('outputs/structured_imitation/seed11_v1/structured_policy.json')
    calibration=Path('outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json')
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    protocol=dict(variant=StructuredResidualMAPPO.variant,seed=11,updates=64,
        training_seeds=list(range(610000,610064)),validation_seeds=list(range(60000,60020)),
        center=str(center),center_sha256=digest(center),calibration=str(calibration),calibration_sha256=digest(calibration),
        runner_sha256=digest(__file__),actor_sha256=digest('src/algos/structured_residual.py'),
        fixed_price=1.2,pricing_scope='shared_fixed_fixture',scenario='S2',days=28,capacity=500,
        billing='accepted_delivered_units',offset_units=[-10,-5,0,5,10],duplicates='one physical outcome',
        initial_logit_gap=8.,initial_center_probability_interior=float(1/(1+4*np.exp(-8))),
        initialization_gate='greedy AND sampled: own-state agreement>=.99, teacher reward CI lower>=-1/day, fill CI lower>=-.005',
        learning_gate='greedy AND sampled: reward CI lower>0 and fill CI lower>=-.005 versus own initialization, teacher, constant, ss',
        lr=.0003,epochs=4,hidden=64,reward_scale=100,target_kl=.02,entropy_coefficient=.01,
        max_seconds=570,held_out_test_used=False,selection='final update only; no validation during training',
        expansion='none; single bounded diagnostic, not vanilla MAPPO')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    try:
        torch.set_num_threads(1);torch.manual_seed(11)
        params=load_identifiable_training_calibration(calibration)
        train=TrainingWindows(make_research_env('S2',split='train',start=15,params=params,pricing_scope='shared'),28)
        val=make_research_env('S2',split='validation',params=params,pricing_scope='shared')
        model=StructuredResidualMAPPO(train,center)
        rows,traces={},{}
        def evaluate(label,actor,greedy=True):
            r,t,_,_=measure(val,actor,protocol['validation_seeds'],label,greedy=greedy,
                deadline=deadline,trace_path=out/'traces.jsonl')
            rows[label],traces[label]=r,t
            return r
        for label,actor in [('teacher',None),('constant','constant'),('ss','ss')]:evaluate(label,actor)
        evaluate('initial_greedy',model.replenishment)
        evaluate('initial_sampled',model.replenishment,False)
        initial_gates={mode:equivalence(rows['initial_'+mode],rows['teacher']) for mode in ['greedy','sampled']}
        model.save(out/'initial.pt')
        before=[p.detach().clone() for p in model.replenishment.parameters()]
        history=[];comparisons={};learning_gate=False;replay_episodes=0
        if all(g['passed'] for g in initial_gates.values()):
            print('Both initialization gates passed. Starting 64 bounded PPO updates.',flush=True)
            for update,seed in enumerate(protocol['training_seeds']):
                if time.perf_counter()>deadline:raise TimeoutError('residual campaign deadline')
                history+=model.train(1,seed=seed)
                if not all(np.isfinite(v) for v in history[-1].values()):raise FloatingPointError('nonfinite PPO metric')
                if (update+1)%16==0:print(f'Completed {update+1}/64 updates',flush=True)
            model.save(out/'trained.pt')
            for mode in ['greedy','sampled']:
                evaluate('trained_'+mode,model.replenishment,mode=='greedy')
                comparisons[mode]={ref:compare(rows['trained_'+mode],rows[ref]) for ref in
                                  ['initial_'+mode,'teacher','constant','ss']}
            learning_gate=all(c['reward_ci95'][0]>0 and c['fill_ci95'][0]>=-.005
                              for cs in comparisons.values() for c in cs.values())
            restored=StructuredResidualMAPPO(val,center);restored.load(out/'trained.pt')
            for mode in ['greedy','sampled']:
                evaluate('replay_'+mode,restored.replenishment,mode=='greedy')
                assert [dict(r,label='trained_'+mode) for r in traces['replay_'+mode]]==traces['trained_'+mode]
                replay_episodes+=20
        else:
            print('Initialization gate failed: no PPO updates.',flush=True)
            restored=StructuredResidualMAPPO(val,center);restored.load(out/'initial.pt')
            for mode in ['greedy','sampled']:
                evaluate('replay_'+mode,restored.replenishment,mode=='greedy')
                assert [dict(r,label='initial_'+mode) for r in traces['replay_'+mode]]==traces['initial_'+mode]
                replay_episodes+=20
        reference=[r['demand'] for r in traces['teacher']]
        assert all([r['demand'] for r in ts]==reference for ts in traces.values())
        assert digest(center)==protocol['center_sha256'] and digest(calibration)==protocol['calibration_sha256']
        summary=dict(protocol=protocol,initial_gates=initial_gates,completed_updates=len(history),history=history,
            learning_gate=bool(learning_gate),comparisons=comparisons,
            actor_parameter_l2_change=float(sum((p-b).square().sum().item() for p,b in zip(model.replenishment.parameters(),before))**.5),
            means={label:{k:float(np.mean([r[k] for r in rs])) for k in ['reward_day','fill','teacher_agreement']} for label,rs in rows.items()},
            episode_rows=sum(rows.values(),[]),scored_episodes=20*(len(rows)-2),replay_episodes=replay_episodes,
            training_windows=train.windows,training_transitions=sum(h['daily_steps'] for h in history),
            paired_demand_verified=True,checkpoint_replay_verified=True,held_out_test_used=False,
            seconds=time.perf_counter()-start)
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        print(json.dumps({k:summary[k] for k in ['initial_gates','completed_updates','learning_gate','means','seconds']},indent=2))
    except Exception as exc:
        (out/'failed_status.json').write_text(json.dumps(dict(status='failed_partial',error=str(exc),held_out_test_used=False)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);run(p.parse_args().output)
