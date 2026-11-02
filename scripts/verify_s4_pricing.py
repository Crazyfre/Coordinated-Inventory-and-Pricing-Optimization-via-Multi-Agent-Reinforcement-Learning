"""Independent archived S4 static-action economic comparison audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.train_compact_residual import digest,write_json
from scripts.verify_parameter_confirmation import episodes,check_episode
from src.config.sku_config import load_skus


def verify(output):
    out=Path(output);load=lambda p:json.loads((out/p).read_text())
    manifest,summary=load('manifest.json'),load('summary.json')
    assert manifest['scenario']=='S4' and manifest['elasticity']==-3. and summary['status']=='complete'
    for p,h in manifest['hashes'].items():assert digest(p)==h,p
    for p,h in summary['trace_hashes'].items():assert digest(out/p)==h
    calibration=next(p for p in manifest['hashes'] if p.endswith('train_only_nb_identifiable_v2.json'))
    order=json.loads(Path(calibration).read_text())['series_order'];lookup={s.series_id:s.base_price for s in load_skus()}
    base=np.array([lookup[s] for s in order]);results={};noise={};targets={}
    for key,trace in episodes(out/'assessment.jsonl'):
        label,scenario=key;assert label in [f'price{p}' for p in range(5)] and scenario in range(2330000,2330064)
        assert key not in results;p=int(label[-1])
        start=int(np.random.default_rng(np.random.SeedSequence([scenario,314159])).integers(15,1859))
        result=check_episode(trace,manifest['covers'],start)
        for r in trace:
            if r['day'] in [1,8,15,22]:np.testing.assert_array_equal(r['price_actions'],np.full(10,p))
            else:assert r['price_actions'] is None
            np.testing.assert_allclose(r['info']['price'],base*[.8,.9,1,1.1,1.2][p],atol=1e-12,rtol=0)
        for metric in ['revenue','ordering_cost','holding_cost','stockout_cost','price_change_cost']:
            result[metric]=sum(r['info']['reward_terms'][metric] for r in trace)
        result['lost_sales']=sum(np.asarray(r['info']['lost_sales']).sum() for r in trace)
        result['clipped_units']=sum(np.asarray(r['info']['orders_clipped']).sum() for r in trace)
        result['stockout_sku_day_fraction']=float(np.mean([np.asarray(r['info']['lost_sales'])>0 for r in trace]))
        results[key]=result
        seeds=[(r['demand_seed'],r['supplier_seed']) for r in trace]
        if scenario in noise:assert seeds==noise[scenario]
        else:noise[scenario]=seeds
        if scenario==2330000 and p in [0,4]:targets[key]=trace
    assert len(results)==320 and len(noise)==64
    seen=set()
    for key,trace in episodes(out/'replays.jsonl'):
        assert key not in seen and trace==targets[key];seen.add(key)
    assert len(seen)==2
    assert len(summary['rows'])==len({(r['label'],r['scenario']) for r in summary['rows']})==320
    for r in summary['rows']:
        for metric,value in results[r['label'],r['scenario']].items():
            if metric!='discounted_day':assert abs(r[metric]-value)<1e-8
    draws=np.random.default_rng(20261029).integers(64,size=(10000,64));eligible=[]
    for p in range(5):
        label=f'price{p}'
        for metric,mean in summary['analysis']['means'][label].items():
            assert abs(mean-np.mean([results[label,k][metric] for k in range(2330000,2330064)]))<1e-8
        for metric in ['reward_day','fill']:
            delta=np.array([results[label,k][metric]-results['price4',k][metric] for k in range(2330000,2330064)])
            expected=summary['analysis']['effects'][f'{label}:{metric}']
            assert abs(expected['mean']-delta.mean())<1e-10
            np.testing.assert_allclose(expected['ci95'],np.quantile([delta[d].mean() for d in draws],[.025,.975]),atol=1e-10,rtol=0)
        if p!=4 and summary['analysis']['effects'][f'{label}:reward_day']['mean']>=.5 and summary['analysis']['effects'][f'{label}:fill']['mean']>=-.005:eligible.append(p)
    assert eligible==summary['analysis']['promising_price_indices']
    assert not summary['analysis']['s2_success_claim'] and summary['analysis']['scenario']=='S4'
    assert summary['counters']==dict(episodes=322,days=9016) and summary['seconds']<=600 and not summary['test_scoring']
    write_json(out/'verification.json',dict(status='passed',episodes=322,exact_replays=2,
        actions_economics_pairing_intervals=True,new_simulator_days=0,verifier_sha256=digest(__file__)))
    print('S4 pricing diagnostic verified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
