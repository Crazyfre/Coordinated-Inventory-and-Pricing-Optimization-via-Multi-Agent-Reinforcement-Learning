"""Independent archive reconstruction of rollout fitting and full-episode effects."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.config.sku_config import load_skus
from scripts.train_compact_residual import digest,write_json


def independent_fit(rows):
    assert [(r['episode'],r['day']) for r in rows]==[(e,d) for e in range(16) for d in [1,8,15,22]]
    candidates=[None,3.,2.5,2.,1.5,1.,.5,0.];scores=[];counts=[];indices=[]
    for sku in range(10):
        sku_scores=[];sku_counts=[]
        for threshold in candidates:
            values=[];count=0
            for r in rows:
                x=np.array(r['obs'],np.float32);v=np.array(r['advantages'],float)
                assert v.shape==(16,10) and np.isfinite(v).all()
                chosen=threshold is not None and x[sku]>np.float32(threshold*7)*x[-20+sku]
                values.append(float(v.mean(0)[sku]) if chosen else 0.);count+=int(chosen)
            sku_scores.append(float(np.mean(values)));sku_counts.append(count)
        best=0
        for j in range(1,8):
            if sku_scores[j]>sku_scores[best]:best=j
        scores.append(sku_scores);counts.append(sku_counts);indices.append(best)
    return dict(thresholds=[candidates[j] for j in indices],scores=scores,
        discount_state_counts=counts,selected_indices=indices)


def independent_analysis(rows,a):
    ix={(r['label'],r['key']):r for r in rows};keys=list(range(2460000,2460064))
    assert len(ix)==len(rows)==192 and set(ix)=={(l,k) for l in ['learned','baseline','heuristic'] for k in keys}
    draws=np.random.default_rng(20261102).integers(64,size=(10000,64));effects={}
    for ref in ['baseline','heuristic']:
        for metric in ['reward_day','fill']:
            x=np.array([ix['learned',k][metric]-ix[ref,k][metric] for k in keys])
            bounds=np.quantile([x[d].mean() for d in draws],[.00625,.99375]);e=a['effects'][f'{ref}:{metric}']
            assert e['mean']==float(x.mean());np.testing.assert_allclose(e['ci9875'],bounds,rtol=0,atol=1e-12)
            effects[f'{ref}:{metric}']=(x.mean(),bounds)
    r,ci=effects['baseline:reward_day'];f=effects['baseline:fill'][1][0]
    assert a['primary_pass']==bool(r>=.5 and ci[0]>0 and f>=-.005)
    assert a['heuristic_pass']==bool(effects['heuristic:reward_day'][1][0]>0 and effects['heuristic:fill'][1][0]>=-.005)
    for label in ['learned','baseline','heuristic']:
        for metric in ['reward_day','fill']:
            assert a['means'][label][metric]==float(np.mean([ix[label,k][metric] for k in keys]))
        np.testing.assert_array_equal(a['price_counts'][label],np.sum([ix[label,k]['price_counts'] for k in keys],axis=0))


def verify(output):
    out=Path(output);load=lambda p:json.loads((out/p).read_text())
    m,s,freeze,policy=map(load,['manifest.json','summary.json','assessment_freeze.json','policy.json'])
    assert s['status']=='complete' and freeze['source_hashes']==m['hashes']
    for p,h in m['hashes'].items():assert digest(p)==h,p
    for p,h in s['trace_hashes'].items():assert digest(out/p)==h
    for p,h in freeze['fit_hashes'].items():assert digest(out/p)==h
    assert digest(out/'assessment_freeze.json')==s['freeze_hash']
    assert freeze['thresholds']==policy['thresholds'] and freeze['days']==200928
    cal=next(p for p in m['hashes'] if p.endswith('train_only_nb_identifiable_v2.json'))
    order=json.loads(Path(cal).read_text())['series_order'];lookup={x.series_id:x.base_price for x in load_skus()}
    base=np.array([lookup[x] for x in order]);covers=np.array(m['covers'],np.float32);days=0

    def price_rule(obs,thresholds):
        x=np.array(obs,np.float32);pa=np.full(10,4)
        for i,t in enumerate(thresholds):
            if t is not None and x[i]>np.float32(t*7)*x[-20+i]:pa[i]=3
        return pa

    def audit(trace,first,keys,start,sku=-1,thresholds=None):
        nonlocal days
        assert [r['day'] for r in trace]==list(range(first,29));prices=None
        for r in trace:
            assert r['start']==start and [r['demand_seed'],r['supplier_seed']]==keys
            x=np.array(r['obs'],np.float32)
            position=x[:10]+x[10:80].reshape(10,7).sum(1)
            oa=np.ceil(np.maximum(covers.clip(0,20)*x[-10:]-position,0)/5).clip(0,40).astype(int)
            np.testing.assert_array_equal(oa,r['actions'])
            if (r['day']-1)%7==0:
                assert r['pricing_obs'] is not None
                pa=np.full(10,4) if thresholds is None else price_rule(r['pricing_obs'],thresholds)
                if thresholds is None and r['day']==first and sku>=0:pa[sku]=3
                np.testing.assert_array_equal(pa,r['price_actions'])
                prices=base*np.array([.8,.9,1.,1.1,1.2])[pa]
            else:assert r['price_actions'] is None and r['pricing_obs'] is None
            np.testing.assert_allclose(prices,r['info']['price'],rtol=0,atol=1e-12)
            t=r['info']['reward_terms'];reward=t['revenue']-sum(t[k] for k in ['ordering_cost','holding_cost','stockout_cost','price_change_cost'])
            assert abs(reward-r['reward'])<1e-8 and r['info']['total_on_hand']<=500+1e-8
        days+=len(trace)
        return sum(.99**i*r['reward'] for i,r in enumerate(trace))

    originals={}
    with (out/'originals.jsonl').open() as f:
        for ep,line in enumerate(f):
            r=json.loads(line);assert r['episode']==ep and r['key']==2440000+ep
            start=int(np.random.default_rng(np.random.SeedSequence([r['key'],314159])).integers(15,1859))
            keys=[r['trace'][0]['demand_seed'],r['trace'][0]['supplier_seed']]
            audit(r['trace'],1,keys,start);originals[ep]=r
    assert len(originals)==16;rows=[];suffixes=0
    with (out/'fit_states.jsonl').open() as f:
        for index,line in enumerate(f):
            r=json.loads(line);ep=index//4;day=1+7*(index%4)
            assert (r['episode'],r['key'],r['day'])==(ep,2440000+ep,day)
            original=originals[ep]['trace'];before=r['before'];start=original[0]['start']
            assert before==r['after'] and before['day']==day and before['start']==start and before['pending'] is None
            assert r['probe']==original[day-1:]
            keys=[original[0]['demand_seed'],original[0]['supplier_seed']];audit(r['probe'],day,keys,start)
            np.testing.assert_array_equal(np.array(before['inventory'],np.float32),np.array(before['obs'],np.float32)[:10])
            np.testing.assert_array_equal(np.array(before['history'],np.float32).T.ravel(),np.array(before['obs'])[80:220])
            assert r['statistics']['obs']==r['probe'][0]['pricing_obs']
            assert len(r['draws'])==16 and len(r['replays'])==2
            replay_sku=(ep*4+(day-1)//7)%10;advantages=[]
            assert [v['sku'] for v in r['replays']]==[-1,replay_sku]
            for rep,draw in enumerate(r['draws']):
                keys=[int(np.random.SeedSequence([2450000,ep,day,rep,z]).generate_state(1,dtype=np.uint64)[0]) for z in [0,1]]
                assert draw['keys']==keys and len(draw['branches'])==11;values=[]
                for sku,b in zip([-1]+list(range(10)),draw['branches']):
                    assert b['sku']==sku and b['initial']==before
                    first=b['trace'][0]
                    for section in [slice(0,220),slice(230,250)]:
                        np.testing.assert_array_equal(np.array(first['obs'])[section],np.array(before['obs'])[section])
                    assert first['pricing_obs']==r['statistics']['obs']
                    np.testing.assert_array_equal(first['info']['orders_delivered'],before['accepted'])
                    np.testing.assert_array_equal(first['info']['orders_clipped'],before['clipped'])
                    value=audit(b['trace'],day,keys,start,sku);assert value==b['value'];values.append(value);suffixes+=1
                    if rep==0 and sku in [-1,replay_sku]:
                        replay=r['replays'][0 if sku==-1 else 1]['trace'];assert replay==b['trace'];audit(replay,day,keys,start,sku)
                advantages.append([v-values[0] for v in values[1:]])
            expected=dict(episode=ep,day=day,obs=r['probe'][0]['pricing_obs'],advantages=advantages)
            assert r['statistics']==expected;rows.append(expected)
    assert days==200928 and suffixes==11264
    fitted=independent_fit(rows)
    for k,v in fitted.items():
        if k=='scores':np.testing.assert_allclose(v,policy[k],rtol=0,atol=1e-10)
        else:assert v==policy[k]
    controllers={'learned':policy['thresholds'],'baseline':[None]*10,'heuristic':[1.5]*10}
    results=[];firsts={};noises={}
    with (out/'assessment.jsonl').open() as f:
        for line in f:
            r=json.loads(line);key=r['key'];label=r['label'];trace=r['trace']
            assert key in range(2460000,2460064) and label in controllers
            start=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
            keys=[trace[0]['demand_seed'],trace[0]['supplier_seed']]
            if key in noises:assert noises[key]==keys
            else:noises[key]=keys
            audit(trace,1,keys,start,thresholds=controllers[label])
            demand=sum(sum(v['info']['demand']) for v in trace)
            result=dict(label=label,key=key,reward_day=sum(v['reward'] for v in trace)/28,
                fill=sum(sum(v['info']['sales']) for v in trace)/demand if demand else 1.,
                price_counts=np.bincount(np.concatenate([v['price_actions'] for v in trace if v['price_actions'] is not None]),minlength=5).tolist())
            assert result==r['result'];results.append(result)
            if key==2460000:firsts[label]=trace
    seen=[]
    with (out/'replays.jsonl').open() as f:
        for line in f:
            r=json.loads(line);label=r['label'];assert r['key']==2460000 and r['trace']==firsts[label]
            assert label not in seen;seen.append(label);trace=r['trace']
            audit(trace,1,noises[2460000],trace[0]['start'],thresholds=controllers[label])
    assert set(seen)==set(controllers) and results==s['rows'];independent_analysis(results,s['analysis'])
    assert days==s['days']==206388 and s['fit_seconds']==freeze['fit_seconds']<=1800
    assert s['assessment_seconds']<=600 and s['seconds']<=2400 and not s['test_scoring']
    write_json(out/'verification.json',dict(status='passed',fit_states=64,fit_suffixes=11264,assessment_episodes=192,
        fit_replays=128,assessment_replays=3,days=days,new_simulator_days=0,
        snapshots_actions_keys_economics_fit_freeze_intervals_replays=True,
        primary_pass=s['analysis']['primary_pass'],heuristic_pass=s['analysis']['heuristic_pass'],verifier_sha256=digest(__file__)))
    print('Rollout pricing screen independently verified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
