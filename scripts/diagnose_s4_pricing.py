"""Prospective static price diagnostic in the existing S4 scenario, not S2."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scripts.diagnose_frozen_pricing import collect,STUDY
from scripts.train_compact_residual import INPUTS,digest,write_json
from scripts.train_clean_research import TrainingWindows
from src.experiments.research import make_research_env
from src.demand.identifiable_calibration import load_identifiable_training_calibration


def analyze(rows):
    index={(r['label'],r['scenario']):r for r in rows}
    if len(index)!=len(rows) or set(index)!={(f'price{p}',k) for p in range(5) for k in range(2330000,2330064)}:
        raise ValueError('complete unique S4 diagnostic grid required')
    keys=range(2330000,2330064);draws=np.random.default_rng(20261029).integers(64,size=(10000,64))
    means={};effects={};eligible=[]
    for p in range(5):
        label=f'price{p}'
        means[label]={metric:float(np.mean([index[label,k][metric] for k in keys]))
            for metric in ['reward_day','fill','revenue','ordering_cost','holding_cost','stockout_cost','price_change_cost',
                           'lost_sales','clipped_units','stockout_sku_day_fraction']}
        for metric in ['reward_day','fill']:
            delta=np.array([index[label,k][metric]-index['price4',k][metric] for k in keys])
            effects[f'{label}:{metric}']=dict(mean=float(delta.mean()),ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist())
        if p!=4 and effects[f'{label}:reward_day']['mean']>=.5 and effects[f'{label}:fill']['mean']>=-.005:eligible.append(p)
    return dict(means=means,effects=effects,promising_price_indices=eligible,scenario='S4',s2_success_claim=False)


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    assert json.loads((STUDY/'verification.json').read_text())['confirmed_gate']
    covers=np.array(json.loads((STUDY/'sku_seed211_final.json').read_text())['mean'])
    sources={str(p) for folder in ['src','scripts'] for p in Path(folder).rglob('*.py')}
    sources|={INPUTS['calibration'],str(STUDY/'sku_seed211_final.json'),'S4_PRICING_OPPORTUNITY_PROTOCOL.md'}
    hashes={p:digest(p) for p in sorted(sources)}
    write_json(out/'manifest.json',dict(hashes=hashes,covers=covers,scenario='S4',elasticity=-3.,maximum_seconds=600))
    torch.set_num_threads(1);started=time.perf_counter();counters=dict(episodes=0,days=0)
    try:
        params=load_identifiable_training_calibration(INPUTS['calibration'])
        env=TrainingWindows(make_research_env('S4',split='train',start=15,days=28,params=params,pricing_scope='per_sku'),28)
        assert np.all(env.sim.elasticity==-3.)
        rows=[];targets={}
        with (out/'assessment.jsonl').open('w') as handle:
            for key in range(2330000,2330064):
                for p in range(5):
                    result,trace=collect(env,covers,np.full(10,p),key,f'price{p}',handle,counters,started+600)
                    rows.append(result)
                    if key==2330000 and p in [0,4]:targets[p]=trace
                if (key-2330000+1)%16==0:print(f'S4 prices: {key-2330000+1}/64 scenarios',flush=True)
        with (out/'replays.jsonl').open('w') as handle:
            for p in [0,4]:
                _,trace=collect(env,covers,np.full(10,p),2330000,f'price{p}',handle,counters,started+600)
                assert trace==targets[p]
        assert counters==dict(episodes=322,days=9016)
        assert all(digest(p)==h for p,h in hashes.items())
        elapsed=time.perf_counter()-started;assert elapsed<=600
        result=analyze(rows)
        write_json(out/'summary.json',dict(status='complete',analysis=result,rows=rows,counters=counters,
            seconds=elapsed,test_scoring=False,trace_hashes={p.name:digest(p) for p in out.glob('*.jsonl')}))
        print(json.dumps(result),flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json',dict(error=str(exc),counters=counters));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
