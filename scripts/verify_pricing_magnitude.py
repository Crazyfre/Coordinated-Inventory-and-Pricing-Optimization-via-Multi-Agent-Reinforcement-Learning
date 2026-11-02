"""Independent action/return/selection reconstruction; zero simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.config.sku_config import load_skus
from scripts.train_compact_residual import digest,write_json

TERMS=['revenue','ordering_cost','holding_cost','stockout_cost','price_change_cost']


def check_analysis(rows,a):
    assert len(rows)==len(a['states'])==20 and len({r['episode'] for r in rows})==20
    gains=[];counts=[0]*5
    for row,state in zip(rows,a['states']):
        v=np.array(row['values']);assert v.shape==(128,5) and np.isfinite(v).all()
        means=v[:64].mean(0);other=v[64:].mean(0)
        chosen=sorted(range(5),key=lambda j:(-means[j],0 if j==4 else j+1))[0]
        reverse=sorted(range(5),key=lambda j:(-other[j],0 if j==4 else j+1))[0]
        d=v-v[:,4,None];gain=float(d[64:,chosen].mean());gains.append(gain);counts[chosen]+=1
        expected=dict(episode=row['episode'],chosen=chosen,reverse_chosen=reverse,
            selection_gain=float(d[:64,chosen].mean()),transfer=gain,reverse_transfer=float(d[:64,reverse].mean()),
            action_mean=d.mean(0).tolist(),action_se=(np.std(d,axis=0,ddof=1)/np.sqrt(128)).tolist(),
            action_economics={k:np.mean(row['economics'][k],axis=0).tolist() for k in TERMS})
        assert state==expected
    x=np.array(gains);rng=np.random.default_rng(20261106)
    ci=np.quantile([x[rng.integers(20,size=20)].mean() for _ in range(10000)],[.025,.975])
    np.testing.assert_allclose(ci,a['ci95'],rtol=0,atol=1e-12)
    assert a['mean_transfer']==float(x.mean()) and a['action_counts']==counts
    assert a['discount_states']==20-counts[4] and a['promising_signal']==bool(ci[0]>0 and 20-counts[4]>=4)
    assert a['policy_success'] is False


def verify(output):
    out=Path(output);load=lambda name:json.loads((out/name).read_text())
    m,s=load('manifest.json'),load('summary.json');assert s['status']=='complete'
    for path,h in m['hashes'].items():assert digest(path)==h,path
    assert digest(out/'states.jsonl')==s['trace_hash']
    source=Path('outputs/structured_mappo/stratified_pricing_v1')
    policy=json.loads((source/'policy.json').read_text())
    selected=sorted([r for r in policy['samples'] if r['sku']==4 and r['day']==22],key=lambda r:r['episode'])
    assert len(selected)==20
    assert m['selected']==[dict(episode=r['episode'],rank=r['rank'],obs=r['obs']) for r in selected]
    old={};originals={}
    with (source/'states.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['sample']['sku']==4 and r['sample']['day']==22:old[r['sample']['episode']]=r['before']
    with (source/'scan.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['episode'] in old:originals[r['episode']]=r['trace']
    calibration=next(p for p in m['hashes'] if p.endswith('train_only_nb_identifiable_v2.json'))
    order=json.loads(Path(calibration).read_text())['series_order'];lookup={x.series_id:x.base_price for x in load_skus()}
    base=np.array([lookup[x] for x in order]);covers=np.array(m['covers'],np.float32);days=0
    def audit(trace,first,start,keys,action=4):
        nonlocal days
        assert [r['day'] for r in trace]==list(range(first,29));price=None
        for r in trace:
            assert r['start']==start and [r['demand_seed'],r['supplier_seed']]==keys
            x=np.array(r['obs'],np.float32);position=x[:10]+x[10:80].reshape(10,7).sum(1)
            oa=np.ceil(np.maximum(covers.clip(0,20)*x[-10:]-position,0)/5).clip(0,40).astype(int)
            np.testing.assert_array_equal(oa,r['actions'])
            if (r['day']-1)%7==0:
                pa=np.full(10,4)
                if r['day']==first:pa[4]=action
                np.testing.assert_array_equal(pa,r['price_actions']);assert r['pricing_obs'] is not None
                price=base*np.array([.8,.9,1.,1.1,1.2])[pa]
            else:assert r['price_actions'] is None and r['pricing_obs'] is None
            np.testing.assert_allclose(price,r['info']['price'],rtol=0,atol=1e-12)
            t=r['info']['reward_terms'];assert abs(t['revenue']-sum(t[k] for k in TERMS[1:])-r['reward'])<1e-8
            assert r['info']['total_on_hand']<=500+1e-8
        days+=len(trace);demand=sum(sum(r['info']['demand']) for r in trace)
        return dict(value=sum(.99**i*r['reward'] for i,r in enumerate(trace)),
            fill=sum(sum(r['info']['sales']) for r in trace)/demand if demand else 1.,
            **{k:sum(.99**i*r['info']['reward_terms'][k] for i,r in enumerate(trace)) for k in TERMS})
    rows=[];branches=0
    with (out/'states.jsonl').open() as f:
        for index,line in enumerate(f):
            r=json.loads(line);sample=selected[index];ep=sample['episode'];assert r['episode']==ep
            start=int(np.random.default_rng(np.random.SeedSequence([2500000+ep,314159])).integers(15,1859))
            assert r['before']==r['after']==old[ep] and r['before']['day']==22 and r['before']['pending'] is None
            assert r['original']==r['original_replay']==originals[ep] and r['probe']==r['original'][21:]
            keys=[r['original'][0]['demand_seed'],r['original'][0]['supplier_seed']]
            audit(r['original'],1,start,keys);audit(r['original_replay'],1,start,keys);audit(r['probe'],22,start,keys)
            assert len(r['draws'])==128 and len(r['replays'])==5;values=[];economics={k:[] for k in TERMS}
            for rep,draw in enumerate(r['draws']):
                keys=[int(np.random.SeedSequence([2530000,ep,22,4,rep,z]).generate_state(1,dtype=np.uint64)[0]) for z in [0,1]]
                assert draw['keys']==keys and len(draw['branches'])==5;ms=[]
                for action,b in enumerate(draw['branches']):
                    assert b['action']==action and b['initial']==r['before']
                    first=b['trace'][0];assert first['pricing_obs']==sample['obs']
                    for section in [slice(0,220),slice(230,250)]:
                        np.testing.assert_array_equal(np.array(first['obs'])[section],np.array(r['before']['obs'])[section])
                    np.testing.assert_array_equal(first['info']['orders_delivered'],r['before']['accepted'])
                    np.testing.assert_array_equal(first['info']['orders_clipped'],r['before']['clipped'])
                    value=audit(b['trace'],22,start,keys,action);assert value==b['metrics'];ms.append(value);branches+=1
                    if rep==0:
                        assert r['replays'][action]==b['trace'];audit(r['replays'][action],22,start,keys,action)
                values.append([v['value'] for v in ms])
                for k in TERMS:economics[k].append([v[k]-ms[4][k] for v in ms])
            row=dict(episode=ep,values=values,economics=economics);assert row==r['statistics'];rows.append(row)
    assert rows==s['statistics'];check_analysis(rows,s['analysis'])
    assert branches==12800 and days==s['days']==m['planned_days']==91560
    assert s['seconds']<=m['cap_seconds']==900 and not s['test_scoring']
    write_json(out/'verification.json',dict(status='passed',states=20,suffixes=branches,days=days,new_simulator_days=0,
        selection_snapshots_keys_actions_economics_intervals_replays=True,promising_signal=s['analysis']['promising_signal'],verifier_sha256=digest(__file__)))
    print('Pricing magnitude independently verified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
