"""Independent archive verification of selected-state pricing effects."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.config.sku_config import load_skus
from scripts.train_compact_residual import digest,write_json


def check_statistics(rows,analysis):
    assert [r['candidate'] for r in rows]==[[0,15,2],[3,15,4],[6,22,7],[7,8,9]]
    assert len(analysis['states'])==4 and analysis['policy_success'] is False
    positives=0
    for j,(r,s) in enumerate(zip(rows,analysis['states'])):
        x=np.array(r['differences']);assert x.shape==(128,) and np.isfinite(x).all()
        rng=np.random.default_rng(20261101+j)
        sampled=[x[rng.integers(128,size=128)].mean() for _ in range(10000)]
        bounds=np.quantile(sampled,[.00625,.99375])
        assert s['candidate']==r['candidate'] and s['mean']==float(x.mean())
        np.testing.assert_allclose(bounds,s['ci9875'],rtol=0,atol=1e-12)
        assert s['positive']==bool(bounds[0]>0);positives+=s['positive']
        assert s['mean_fill']==float(np.mean(r['fill_differences']))
        assert s['economics']=={k:float(np.mean(v)) for k,v in r['economics'].items()}
    assert analysis['positive_states']==positives


def verify(output):
    out=Path(output);load=lambda p:json.loads((out/p).read_text())
    m,s=load('manifest.json'),load('summary.json');assert s['status']=='complete'
    for p,h in m['hashes'].items():assert digest(p)==h,p
    for p,h in s['trace_hashes'].items():assert digest(out/p)==h
    source=Path('outputs/structured_mappo/conditional_pricing_signal_v1')
    oldstates={}
    with (source/'states.jsonl').open() as f:
        for line in f:
            r=json.loads(line);oldstates[(r['episode'],r['day'],r['sku'])]=r
    originals={r['episode']:r for r in map(json.loads,(source/'originals.jsonl').read_text().splitlines())}
    calibration=next(p for p in m['hashes'] if p.endswith('train_only_nb_identifiable_v2.json'))
    order=json.loads(Path(calibration).read_text())['series_order'];lookup={x.series_id:x.base_price for x in load_skus()}
    base=np.array([lookup[x] for x in order]);covers=np.array(m['covers'],np.float32);days=0
    terms=['revenue','ordering_cost','holding_cost','stockout_cost','price_change_cost']

    def audit(trace,first,sku,alternative,keys,start):
        nonlocal days
        assert [r['day'] for r in trace]==list(range(first,29));prices=None
        for r in trace:
            assert r['start']==start and [r['demand_seed'],r['supplier_seed']]==keys
            x=np.array(r['obs'],np.float32);position=x[:10]+x[10:80].reshape(10,7).sum(1)
            action=np.ceil(np.maximum(covers.clip(0,20)*x[-10:]-position,0)/5).clip(0,40).astype(int)
            np.testing.assert_array_equal(action,r['actions'])
            if (r['day']-1)%7==0:
                pa=np.full(10,4)
                if r['day']==first:pa[sku]=alternative
                np.testing.assert_array_equal(pa,r['price_actions']);assert r['pricing_obs'] is not None
                prices=base*np.array([.8,.9,1.,1.1,1.2])[pa]
            else:assert r['price_actions'] is None and r['pricing_obs'] is None
            np.testing.assert_allclose(prices,r['info']['price'],rtol=0,atol=1e-12)
            t=r['info']['reward_terms']
            assert abs(t['revenue']-sum(t[k] for k in terms[1:])-r['reward'])<1e-8
            assert r['info']['total_on_hand']<=500+1e-8
        days+=len(trace);demand=sum(sum(r['info']['demand']) for r in trace)
        return dict(value=sum(.99**i*r['reward'] for i,r in enumerate(trace)),
            fill=sum(sum(r['info']['sales']) for r in trace)/demand if demand else 1.,
            **{k:sum(.99**i*r['info']['reward_terms'][k] for i,r in enumerate(trace)) for k in terms})

    statistics=[]
    with (out/'states.jsonl').open() as f:
        for index,line in enumerate(f):
            r=json.loads(line);candidate=[[0,15,2],[3,15,4],[6,22,7],[7,8,9]][index]
            assert r['candidate']==candidate;ep,day,sku=candidate
            old=oldstates[tuple(candidate)];original=originals[ep]
            key=2310000+ep;start=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
            assert original['start']==start and original['key']==key
            assert r['original']==original['trace'] and r['before']==r['after']==old['state_before']
            before=r['before'];assert before['pending'] is None and before['day']==day
            assert r['probe']==r['original'][day-1:]
            keys=[r['original'][0]['demand_seed'],r['original'][0]['supplier_seed']]
            audit(r['original'],1,0,4,keys,start);audit(r['probe'],day,sku,4,keys,start)
            assert len(r['draws'])==128 and len(r['replays'])==2
            differences=[];fills=[];economics={k:[] for k in terms}
            for rep,draw in enumerate(r['draws']):
                keys=[int(np.random.SeedSequence([2430000,ep,day,rep,z]).generate_state(1,dtype=np.uint64)[0]) for z in [0,1]]
                assert draw['keys']==keys and keys not in [d['keys'] for d in old['draws']]
                assert len(draw['branches'])==2;values=[]
                for action,b in zip([3,4],draw['branches']):
                    assert b['action']==action and b['initial']==before
                    first=b['trace'][0]
                    for section in [slice(0,220),slice(230,250)]:
                        np.testing.assert_array_equal(np.array(first['obs'])[section],np.array(before['obs'])[section])
                    np.testing.assert_array_equal(first['info']['orders_delivered'],before['accepted'])
                    np.testing.assert_array_equal(first['info']['orders_clipped'],before['clipped'])
                    value=audit(b['trace'],day,sku,action,keys,start);assert value==b['metrics'];values.append(value)
                    if rep==0:
                        assert r['replays'][action-3]==b['trace']
                        audit(r['replays'][action-3],day,sku,action,keys,start)
                a,b=values;differences.append(a['value']-b['value']);fills.append(a['fill']-b['fill'])
                for k in terms:economics[k].append(a[k]-b[k])
            expected=dict(candidate=candidate,differences=differences,fill_differences=fills,economics=economics)
            assert r['statistics']==expected;statistics.append(expected)
    assert statistics==s['statistics'];check_statistics(statistics,s['analysis'])
    assert days==s['days']==14616 and s['seconds']<=600 and not s['test_scoring']
    write_json(out/'verification.json',dict(status='passed',states=4,paired_draws=512,suffixes=1024,
        original_episodes=4,probes=4,replays=8,days=days,new_simulator_days=0,
        snapshots_actions_keys_economics_intervals_replays=True,positive_states=s['analysis']['positive_states'],
        verifier_sha256=digest(__file__)))
    print('Pricing candidate diagnostic independently verified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
