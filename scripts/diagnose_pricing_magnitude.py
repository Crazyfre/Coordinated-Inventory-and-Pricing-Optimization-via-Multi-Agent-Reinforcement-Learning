"""Fixed twenty-state, five-price fresh-future diagnostic; no actor fit."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scripts.confirm_pricing_candidates import metrics,TERMS
from scripts.diagnose_conditional_pricing import advance,suffix,STUDY
from scripts.diagnose_counterfactual_rollouts import clone_environment
from scripts.audit_repeated_futures import snapshot,resample_future
from scripts.screen_stratified_pricing import reset
from scripts.train_compact_residual import INPUTS,digest,write_json
from src.experiments.research import make_research_env
from src.demand.identifiable_calibration import load_identifiable_training_calibration

SOURCE=Path('outputs/structured_mappo/stratified_pricing_v1')


def select(values):
    v=np.asarray(values,float)
    if v.shape!=(64,5) or not np.isfinite(v).all():raise ValueError('64 finite draws by five actions required')
    return max([4,0,1,2,3],key=lambda a:v[:,a].mean())


def analyze(rows):
    if len(rows)!=20 or len({r['episode'] for r in rows})!=20:raise ValueError('twenty unique states required')
    results=[]
    for r in rows:
        v=np.array(r['values']);assert v.shape==(128,5)
        chosen=select(v[:64]);reverse=select(v[64:]);delta=v-v[:,4,None]
        results.append(dict(episode=r['episode'],chosen=chosen,reverse_chosen=reverse,
            selection_gain=float(delta[:64,chosen].mean()),transfer=float(delta[64:,chosen].mean()),
            reverse_transfer=float(delta[:64,reverse].mean()),
            action_mean=delta.mean(0).tolist(),action_se=(delta.std(0,ddof=1)/np.sqrt(128)).tolist(),
            action_economics={k:np.array(r['economics'][k]).mean(0).tolist() for k in TERMS}))
    x=np.array([r['transfer'] for r in results]);draws=np.random.default_rng(20261106).integers(20,size=(10000,20))
    ci=np.quantile(x[draws].mean(1),[.025,.975]).tolist();count=sum(r['chosen']!=4 for r in results)
    return dict(states=results,mean_transfer=float(x.mean()),ci95=ci,discount_states=count,
        action_counts=np.bincount([r['chosen'] for r in results],minlength=5).tolist(),
        promising_signal=bool(ci[0]>0 and count>=4),policy_success=False)


def run(output):
    assert json.loads((SOURCE/'verification.json').read_text())['status']=='passed'
    frozen=json.loads((SOURCE/'assessment_freeze.json').read_text())
    for name in ['policy.json','scan.jsonl','states.jsonl']:assert digest(SOURCE/name)==frozen['fit_hashes'][name]
    policy=json.loads((SOURCE/'policy.json').read_text())
    selected=sorted([s for s in policy['samples'] if s['sku']==4 and s['day']==22],key=lambda s:s['episode'])
    assert len(selected)==20 and len({s['episode'] for s in selected})==20
    ids={s['episode'] for s in selected};old={};originals={}
    with (SOURCE/'states.jsonl').open() as f:
        for line in f:
            r=json.loads(line);s=r['sample']
            if s['episode'] in ids and s['sku']==4 and s['day']==22:old[s['episode']]=r['before']
    with (SOURCE/'scan.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['episode'] in ids:originals[r['episode']]=r['trace']
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    paths={str(p) for folder in ['src','scripts'] for p in Path(folder).rglob('*.py')}
    paths|={str(p) for p in Path('data').rglob('*') if p.is_file() and ('aligned' in str(p) or 'sku' in p.name.lower())}
    paths|={'src/config/skus.json',INPUTS['calibration'],str(STUDY/'sku_seed211_final.json'),'PRICING_MAGNITUDE_PROTOCOL.md'}
    paths|={str(SOURCE/n) for n in ['policy.json','scan.jsonl','states.jsonl','assessment_freeze.json','verification.json']}
    hashes={p:digest(p) for p in sorted(paths)};covers=np.array(json.loads((STUDY/'sku_seed211_final.json').read_text())['mean'])
    write_json(out/'manifest.json',dict(hashes=hashes,covers=covers,
        selected=[dict(episode=s['episode'],rank=s['rank'],obs=s['obs']) for s in selected],planned_days=91560,cap_seconds=900))
    torch.set_num_threads(1);began=time.perf_counter();deadline=began+900;counters=dict(days=0);rows=[]
    try:
        env=make_research_env('S2',split='train',start=15,pricing_scope='per_sku',params=load_identifiable_training_calibration(INPUTS['calibration']))
        with (out/'states.jsonl').open('w') as handle:
            for s in selected:
                ep=s['episode'];key=2500000+ep;reset(env,key);original=[]
                while not env._finished:
                    if env.day==22:
                        before=snapshot(env);assert before==old[ep] and env.pricing_observation().tolist()==s['obs']
                        probe=suffix(clone_environment(env),covers,4,4,counters,deadline);assert probe==originals[ep][21:]
                        draws=[];replays=[];values=[];economics={k:[] for k in TERMS}
                        for rep in range(128):
                            keys=[int(np.random.SeedSequence([2530000,ep,22,4,rep,z]).generate_state(1,dtype=np.uint64)[0]) for z in [0,1]]
                            branches=[]
                            for action in range(5):
                                branch=resample_future(env,keys);initial=snapshot(branch)
                                trace=suffix(branch,covers,4,action,counters,deadline)
                                branches.append(dict(action=action,initial=initial,trace=trace,metrics=metrics(trace)))
                                if rep==0:
                                    replay=suffix(resample_future(env,keys),covers,4,action,counters,deadline)
                                    assert replay==trace;replays.append(replay)
                            values.append([b['metrics']['value'] for b in branches])
                            for k in TERMS:economics[k].append([b['metrics'][k]-branches[4]['metrics'][k] for b in branches])
                            draws.append(dict(keys=keys,branches=branches))
                        after=snapshot(env);assert after==before
                    step,_=advance(env,covers,int(env.day),0,4,counters,deadline);original.append(step)
                assert original==originals[ep];reset(env,key)
                replay=suffix(env,covers,0,4,counters,deadline);assert replay==original
                row=dict(episode=ep,values=values,economics=economics);rows.append(row)
                handle.write(json.dumps(dict(episode=ep,before=before,after=after,original=original,
                    original_replay=replay,probe=probe,draws=draws,replays=replays,statistics=row),allow_nan=False)+'\n');handle.flush()
                write_json(out/'progress.json',dict(states=len(rows),days=counters['days'],seconds=time.perf_counter()-began))
                print(f'Pricing magnitude: {len(rows)}/20',flush=True)
        assert counters['days']==91560 and all(digest(p)==h for p,h in hashes.items())
        elapsed=time.perf_counter()-began;assert elapsed<=900
        analysis=analyze(rows)
        write_json(out/'summary.json',dict(status='complete',analysis=analysis,statistics=rows,days=counters['days'],
            seconds=elapsed,test_scoring=False,trace_hash=digest(out/'states.jsonl')))
        print(json.dumps({k:v for k,v in analysis.items() if k!='states'}),flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json',dict(error=str(exc),counters=counters));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
