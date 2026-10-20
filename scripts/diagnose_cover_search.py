"""Predeclared small training-only cover search; frozen validation comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scripts.diagnose_teacher_ordering import measure,compare
from scripts.train_clean_research import TrainingWindows
from src.algos.imitation_ordering import teacher_indices
from src.experiments.research import make_research_env
from src.demand.identifiable_calibration import load_identifiable_training_calibration


class CoverPolicy:
    def __init__(self,n,pipe_horizon,cover):
        if not np.isfinite(cover) or cover<0:raise ValueError('nonnegative finite cover required')
        self.n,self.pipe_horizon,self.cover=n,pipe_horizon,float(cover)
    def act(self,obs,greedy=True):
        return teacher_indices(torch.as_tensor(obs),self.n,self.pipe_horizon,cover=self.cover).numpy(),0.


def select_candidate(candidates,teacher_fill):
    eligible=[r for r in candidates if r['fill']>=teacher_fill-.005]
    if not eligible:raise ValueError('no eligible candidate')
    # Ties favor smallest change from teacher, then smaller cover.
    return max(eligible,key=lambda r:(r['reward_day'],-abs(r['cover']-6),-r['cover']))


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();deadline=start+600
    calibration=Path('outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json')
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    protocol=dict(variant='bounded_uniform_cover_search_v1',covers=[4.,4.5,5.,5.5,6.,6.5,7.,7.5,8.],
        training_seeds=list(range(710000,710064)),validation_seeds=list(range(70000,70040)),
        training_end_day=1885,validation_days=[1886,1913],days=28,scenario='S2',price_multiplier=1.2,
        pricing_scope='shared_fixed_fixture',capacity=500,billing='accepted_delivered_units',
        selection='max training mean reward among candidates with mean fill >= cover6 mean fill-.005; ties nearest6 then smaller',
        validation='selected rule only against teacher6, constant, ss; no validation reselection',
        gate='positive paired reward lower95% bound and fill lower95% bound>=-.005 against EACH comparator',
        calibration=str(calibration),calibration_sha256=digest(calibration),runner_sha256=digest(__file__),
        max_seconds=600,held_out_test_used=False,rl_updates=0,
        interpretation='feasible rule search, not optimality or RL advantage; uniform cover only')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    try:
        torch.set_num_threads(1)
        params=load_identifiable_training_calibration(calibration)
        train=TrainingWindows(make_research_env('S2',split='train',start=15,params=params,pricing_scope='shared'),28)
        candidates=[];training_rows=[];reference=None;teacher_rows=None
        for cover in protocol['covers']:
            rows,traces,_,_=measure(train,CoverPolicy(train.n,train.obs_pipe_horizon,cover),protocol['training_seeds'],
                f'cover_{cover:g}',deadline=deadline,trace_path=out/'training_traces.jsonl')
            demands=[r['demand'] for r in traces]
            if reference is None:reference=demands
            else:assert demands==reference
            training_rows+=rows
            candidates.append(dict(cover=cover,reward_day=float(np.mean([r['reward_day'] for r in rows])),
                                   fill=float(np.mean([r['fill'] for r in rows]))))
            if cover==6:teacher_rows=rows
            (out/'training_progress.json').write_text(json.dumps(candidates,indent=2))
            print(json.dumps(candidates[-1]),flush=True)
        selected=select_candidate(candidates,next(r['fill'] for r in candidates if r['cover']==6))
        selection=dict(selected=selected,training_candidates=candidates,selection_split='train',validation_scored=False)
        (out/'selection.json').write_text(json.dumps(selection,indent=2))
        selection_hash=digest(out/'selection.json')
        # Only now construct validation environment and score frozen choice.
        val=make_research_env('S2',split='validation',params=params,pricing_scope='shared')
        rows_by_label={};traces_by_label={}
        for label,actor in [('selected',CoverPolicy(val.n,val.obs_pipe_horizon,selected['cover'])),
                            ('teacher',None),('constant','constant'),('ss','ss')]:
            rows,traces,_,_=measure(val,actor,protocol['validation_seeds'],label,deadline=deadline,
                                   trace_path=out/'validation_traces.jsonl')
            rows_by_label[label],traces_by_label[label]=rows,traces
        # Deserialize frozen choice and replay it; no trained checkpoint involved.
        restored=json.loads((out/'selection.json').read_text())['selected']['cover']
        _,replay,_,_=measure(val,CoverPolicy(val.n,val.obs_pipe_horizon,restored),protocol['validation_seeds'],
                             'replay',deadline=deadline,trace_path=out/'validation_traces.jsonl')
        assert traces_by_label['selected']==[dict(r,label='selected') for r in replay]
        ref=[r['demand'] for r in traces_by_label['teacher']]
        assert all([r['demand'] for r in ts]==ref for ts in traces_by_label.values())
        comparisons={label:compare(rows_by_label['selected'],rows) for label,rows in rows_by_label.items() if label!='selected'}
        gates={label:bool(r['reward_ci95'][0]>0 and r['fill_ci95'][0]>=-.005) for label,r in comparisons.items()}
        assert digest(out/'selection.json')==selection_hash and digest(calibration)==protocol['calibration_sha256']
        summary=dict(protocol=protocol,selection=selection,selection_sha256=selection_hash,
            training_episode_rows=training_rows,validation_episode_rows=sum(rows_by_label.values(),[]),
            training_windows=train.windows,comparisons=comparisons,gates=gates,all_comparators_gate=all(gates.values()),
            means={label:{key:float(np.mean([r[key] for r in rows])) for key in ['reward_day','fill','teacher_agreement']}
                   for label,rows in rows_by_label.items()},
            training_episodes=576,validation_episodes=160,replay_episodes=40,total_days=21728,
            paired_demands_verified=True,selection_replay_verified=True,held_out_test_used=False,rl_updates=0,
            seconds=time.perf_counter()-start)
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        print(json.dumps({k:summary[k] for k in ['selection','means','comparisons','gates','seconds']},indent=2),flush=True)
    except Exception as exc:
        (out/'failed_status.json').write_text(json.dumps(dict(status='failed_partial',error=str(exc),held_out_test_used=False)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);run(p.parse_args().output)
