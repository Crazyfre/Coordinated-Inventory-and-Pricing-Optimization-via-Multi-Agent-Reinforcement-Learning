"""Frozen selected-state pricing diagnostic, not policy training."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scripts.diagnose_conditional_pricing import advance, suffix, STUDY
from scripts.diagnose_counterfactual_rollouts import clone_environment
from scripts.audit_repeated_futures import snapshot, resample_future
from scripts.train_compact_residual import INPUTS, digest, write_json
from src.experiments.research import make_research_env
from src.demand.identifiable_calibration import load_identifiable_training_calibration

CANDIDATES = [(0,15,2),(3,15,4),(6,22,7),(7,8,9)]
SOURCE = Path('outputs/structured_mappo/conditional_pricing_signal_v1')
TERMS = ['revenue','ordering_cost','holding_cost','stockout_cost','price_change_cost']


def metrics(trace):
    demand = sum(sum(r['info']['demand']) for r in trace)
    return dict(value=sum(.99**i*r['reward'] for i,r in enumerate(trace)),
        fill=sum(sum(r['info']['sales']) for r in trace)/demand if demand else 1.,
        **{k:sum(.99**i*r['info']['reward_terms'][k] for i,r in enumerate(trace)) for k in TERMS})


def analyze(rows):
    if [tuple(r['candidate']) for r in rows] != CANDIDATES:
        raise ValueError('exact four predeclared candidates required')
    results=[]
    for j,r in enumerate(rows):
        x=np.asarray(r['differences'],float)
        if x.shape!=(128,) or not np.isfinite(x).all():raise ValueError('128 finite paired draws required')
        indices=np.random.default_rng(20261101+j).integers(128,size=(10000,128))
        ci=np.quantile(x[indices].mean(1),[.00625,.99375])
        results.append(dict(candidate=r['candidate'],mean=float(x.mean()),ci9875=ci.tolist(),
            positive=bool(ci[0]>0),mean_fill=float(np.mean(r['fill_differences'])),
            economics={k:float(np.mean(r['economics'][k])) for k in TERMS}))
    return dict(states=results,positive_states=sum(r['positive'] for r in results),policy_success=False)


def run(output):
    assert json.loads((SOURCE/'verification.json').read_text())['status']=='passed'
    prior=json.loads((SOURCE/'summary.json').read_text())
    for p,h in prior['trace_hashes'].items():assert digest(SOURCE/p)==h
    oldstates={}
    with (SOURCE/'states.jsonl').open() as f:
        for line in f:
            r=json.loads(line);key=(r['episode'],r['day'],r['sku'])
            if key in CANDIDATES:oldstates[key]=r
    originals={r['episode']:r for r in map(json.loads,(SOURCE/'originals.jsonl').read_text().splitlines())}
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    paths={str(p) for folder in ['src','scripts'] for p in Path(folder).rglob('*.py')}
    paths|={str(p) for p in Path('data').rglob('*') if p.is_file() and ('aligned' in str(p) or 'sku' in p.name.lower())}
    paths|={'src/config/skus.json',INPUTS['calibration'],str(STUDY/'sku_seed211_final.json'),
        'PRICING_CANDIDATE_CONFIRMATION_PROTOCOL.md'}
    paths|={str(SOURCE/p) for p in ['summary.json','verification.json','states.jsonl','originals.jsonl']}
    covers=np.array(json.loads((STUDY/'sku_seed211_final.json').read_text())['mean'])
    hashes={p:digest(p) for p in sorted(paths)}
    write_json(out/'manifest.json',dict(hashes=hashes,covers=covers,candidates=CANDIDATES,planned_days=14616,cap_seconds=600))
    torch.set_num_threads(1);starttime=time.perf_counter();deadline=starttime+600;counters=dict(days=0);rows=[]
    try:
        env=make_research_env('S2',split='train',start=15,pricing_scope='per_sku',
            params=load_identifiable_training_calibration(INPUTS['calibration']))
        with (out/'states.jsonl').open('w') as handle:
            for ep,day,sku in CANDIDATES:
                key=2310000+ep;old=oldstates[(ep,day,sku)]
                env.sim.start_day_index=originals[ep]['start'];env.reset(seed=key)
                original=[];record=None
                while not env._finished:
                    if int(env.day)==day:
                        before=snapshot(env);assert before==old['state_before']
                        probe=suffix(clone_environment(env),covers,sku,4,counters,deadline)
                        assert probe==originals[ep]['trace'][day-1:]
                        draws=[];replays=[];diff=[];fill=[];economics={k:[] for k in TERMS}
                        for rep in range(128):
                            keys=[int(np.random.SeedSequence([2430000,ep,day,rep,s]).generate_state(1,dtype=np.uint64)[0]) for s in [0,1]]
                            branches=[]
                            for action in [3,4]:
                                branch=resample_future(env,keys);initial=snapshot(branch)
                                trace=suffix(branch,covers,sku,action,counters,deadline)
                                branches.append(dict(action=action,initial=initial,trace=trace,metrics=metrics(trace)))
                                if rep==0:
                                    replay=suffix(resample_future(env,keys),covers,sku,action,counters,deadline)
                                    assert replay==trace;replays.append(replay)
                            a,b=[v['metrics'] for v in branches]
                            diff.append(a['value']-b['value']);fill.append(a['fill']-b['fill'])
                            for term in TERMS:economics[term].append(a[term]-b[term])
                            draws.append(dict(keys=keys,branches=branches))
                        assert snapshot(env)==before
                        row=dict(candidate=[ep,day,sku],differences=diff,fill_differences=fill,economics=economics)
                        rows.append(row)
                        record=dict(candidate=[ep,day,sku],before=before,after=snapshot(env),probe=probe,
                            draws=draws,replays=replays,statistics=row)
                    step,_=advance(env,covers,int(env.day),0,4,counters,deadline);original.append(step)
                assert original==originals[ep]['trace'];record['original']=original
                handle.write(json.dumps(record,allow_nan=False)+'\n');handle.flush()
                write_json(out/'progress.json',dict(states=len(rows),days=counters['days']))
                print(f'Pricing candidates: {len(rows)}/4',flush=True)
        assert counters['days']==14616 and all(digest(p)==h for p,h in hashes.items())
        elapsed=time.perf_counter()-starttime;assert elapsed<=600
        result=analyze(rows)
        write_json(out/'summary.json',dict(status='complete',analysis=result,statistics=rows,days=counters['days'],
            seconds=elapsed,test_scoring=False,trace_hashes={'states.jsonl':digest(out/'states.jsonl')}))
        print(json.dumps(result),flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json',dict(error=str(exc),counters=counters));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
