"""Independent archive checks for stratified rollout pricing, zero simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.config.sku_config import load_skus
from scripts.train_compact_residual import digest,write_json


def selection(obs,ep):
    candidates=[(d,i) for d in [1,8,15,22] for i in range(10)]
    ratios=[]
    for d,i in candidates:
        x=np.asarray(obs[d],np.float32)
        ratios.append(float(x[i]/(np.float32(1)+np.float32(7)*x[-20+i])))
    best=max(range(40),key=lambda j:ratios[j]);remaining=[j for j in range(40) if j!=best]
    other=remaining[int(np.random.default_rng(np.random.SeedSequence([2500000,ep,991])).integers(39))]
    return [list(candidates[best]),list(candidates[other])]


def feature_matrix(obs,day,base):
    x=np.asarray(obs,np.float32);b=np.asarray(base,np.float32);forecast=x[-20:-10]
    columns=[np.clip(x[:10]/(1+7*forecast),0,3),np.log1p(np.maximum(forecast,0))/5,
        np.clip(x[10:150].reshape(10,14)[:,-7:].mean(1)/(1+forecast),0,3),x[150:160]/b-1,
        np.full(10,(29-day)/28),np.clip(x[-10:]/500,0,1)]
    f=np.stack(columns,axis=1).astype(np.float32).astype(float)
    out=np.zeros((10,37));out[:,:10]=np.eye(10);out[:,10:16]=f
    col=16
    for i in range(6):
        for j in range(i,6):out[:,col]=f[:,i]*f[:,j];col+=1
    return out


def check_ridge(x,y,weights):
    x=np.asarray(x);y=np.asarray(y);w=np.asarray(weights)
    assert x.shape==(256,37) and y.shape==(256,) and w.shape==(37,)
    independent=np.linalg.lstsq(np.vstack([x,np.eye(37)]),np.r_[y,np.zeros(37)],rcond=None)[0]
    np.testing.assert_allclose(w,independent,rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose((x.T@x+np.eye(37))@w,x.T@y,rtol=1e-9,atol=1e-8)


def check_analysis(rows,a):
    ix={(r['label'],r['key']):r for r in rows};keys=list(range(2520000,2520064));labels=['learned','baseline','heuristic']
    assert len(ix)==len(rows)==192 and set(ix)=={(l,k) for l in labels for k in keys}
    draws=np.random.default_rng(20261104).integers(64,size=(10000,64));effects={}
    for ref in ['baseline','heuristic']:
        for metric in ['reward_day','fill']:
            x=np.array([ix['learned',k][metric]-ix[ref,k][metric] for k in keys]);bounds=np.quantile([x[d].mean() for d in draws],[.00625,.99375])
            e=a['effects'][f'{ref}:{metric}'];assert e['mean']==float(x.mean())
            np.testing.assert_allclose(e['ci9875'],bounds,rtol=0,atol=1e-12);effects[f'{ref}:{metric}']=(x.mean(),bounds)
    mean,ci=effects['baseline:reward_day']
    assert a['primary_pass']==bool(mean>=.5 and ci[0]>0 and effects['baseline:fill'][1][0]>=-.005)
    assert a['heuristic_pass']==bool(effects['heuristic:reward_day'][1][0]>0 and effects['heuristic:fill'][1][0]>=-.005)
    for l in labels:
        for metric in ['reward_day','fill']:assert a['means'][l][metric]==float(np.mean([ix[l,k][metric] for k in keys]))
        np.testing.assert_array_equal(a['price_counts'][l],np.sum([ix[l,k]['price_counts'] for k in keys],axis=0))


def verify(output):
    out=Path(output);load=lambda p:json.loads((out/p).read_text())
    m,s,sel,freeze,policy=map(load,['manifest.json','summary.json','selection_freeze.json','assessment_freeze.json','policy.json'])
    assert s['status']=='complete' and sel['source_hashes']==freeze['source_hashes']==m['hashes']
    for p,h in m['hashes'].items():assert digest(p)==h,p
    for p,h in s['trace_hashes'].items():assert digest(out/p)==h
    for p,h in freeze['fit_hashes'].items():assert digest(out/p)==h
    assert digest(out/'selection_freeze.json')==s['selection_hash'] and digest(out/'assessment_freeze.json')==s['freeze_hash']
    assert digest(out/'scan.jsonl')==sel['scan_sha256'] and sel['days']==3584
    cal=next(p for p in m['hashes'] if p.endswith('train_only_nb_identifiable_v2.json'))
    order=json.loads(Path(cal).read_text())['series_order'];lookup={v.series_id:v.base_price for v in load_skus()}
    base=np.array([lookup[v] for v in order]);np.testing.assert_array_equal(base,policy['base_prices'])
    covers=np.array(m['covers'],np.float32);weights=np.array(policy['weights']);days=0

    def audit(trace,first,start,keys,sku=-1,action=4,label='baseline'):
        nonlocal days
        assert [r['day'] for r in trace]==list(range(first,29));prices=None
        for r in trace:
            assert r['start']==start and [r['demand_seed'],r['supplier_seed']]==keys
            x=np.array(r['obs'],np.float32);position=x[:10]+x[10:80].reshape(10,7).sum(1)
            oa=np.ceil(np.maximum(covers.clip(0,20)*x[-10:]-position,0)/5).clip(0,40).astype(int)
            np.testing.assert_array_equal(oa,r['actions'])
            if (r['day']-1)%7==0:
                assert r['pricing_obs'] is not None;pa=np.full(10,4);pobs=np.array(r['pricing_obs'],np.float32)
                if label=='learned':
                    gain=(feature_matrix(pobs,r['day'],base)@weights)*base;best=int(np.argmax(gain))
                    if gain[best]>0:pa[best]=3
                elif label=='heuristic':pa[pobs[:10]>np.float32(10.5)*pobs[-20:-10]]=3
                elif sku>=0 and r['day']==first:pa[sku]=action
                np.testing.assert_array_equal(pa,r['price_actions']);prices=base*np.array([.8,.9,1.,1.1,1.2])[pa]
            else:assert r['price_actions'] is None and r['pricing_obs'] is None
            np.testing.assert_allclose(prices,r['info']['price'],rtol=0,atol=1e-12)
            t=r['info']['reward_terms'];reward=t['revenue']-sum(t[k] for k in ['ordering_cost','holding_cost','stockout_cost','price_change_cost'])
            assert abs(reward-r['reward'])<1e-8 and r['info']['total_on_hand']<=500+1e-8
        days+=len(trace);return sum(.99**i*r['reward'] for i,r in enumerate(trace))

    scans={};length=0;assert len(sel['selections'])==128
    with (out/'scan.jsonl').open() as f:
        for ep,line in enumerate(f):
            r=json.loads(line);assert r['episode']==ep and r['key']==2500000+ep
            start=int(np.random.default_rng(np.random.SeedSequence([r['key'],314159])).integers(15,1859));trace=r['trace']
            keys=[trace[0]['demand_seed'],trace[0]['supplier_seed']];audit(trace,1,start,keys)
            obs={v['day']:v['pricing_obs'] for v in trace if v['pricing_obs'] is not None};chosen=selection(obs,ep)
            stored=sel['selections'][ep];assert stored['episode']==ep and stored['key']==r['key'] and stored['selected']==chosen
            assert {int(k):v for k,v in stored['observations'].items()}==obs
            length+=sum(29-day for day,sku in chosen);scans[ep]=r
    assert len(scans)==128 and length==sel['suffix_length_sum']==s['suffix_length_sum']
    assert sel['planned_days']==12628+67*length<=492884
    with (out/'original_replays.jsonl').open() as f:
        count=0
        for ep,line in enumerate(f):
            r=json.loads(line);assert r==scans[ep];trace=r['trace'];audit(trace,1,trace[0]['start'],[trace[0]['demand_seed'],trace[0]['supplier_seed']]);count+=1
    assert count==128;samples={};branches=0
    with (out/'states.jsonl').open() as f:
        for line in f:
            r=json.loads(line);sample=r['sample'];ep=sample['episode'];rank=sample['rank'];day=sample['day'];sku=sample['sku']
            assert (ep,rank) not in samples and rank in [0,1] and [day,sku]==sel['selections'][ep]['selected'][rank]
            original=scans[ep]['trace'];before=r['before'];assert before==r['after'] and before['day']==day and before['pending'] is None
            assert before['start']==original[0]['start'] and r['probe']==original[day-1:]
            assert sample['obs']==original[day-1]['pricing_obs']
            np.testing.assert_array_equal(np.array(before['inventory'],np.float32),np.array(before['obs'],np.float32)[:10])
            np.testing.assert_array_equal(np.array(before['history'],np.float32).T.ravel(),np.array(before['obs'])[80:220])
            audit(r['probe'],day,before['start'],[original[0]['demand_seed'],original[0]['supplier_seed']])
            assert len(r['draws'])==32 and len(r['replays'])==2;gains=[]
            for rep,draw in enumerate(r['draws']):
                keys=[int(np.random.SeedSequence([2510000,ep,day,sku,rep,z]).generate_state(1,dtype=np.uint64)[0]) for z in [0,1]]
                assert draw['keys']==keys and len(draw['branches'])==2;values=[]
                for action,b in zip([3,4],draw['branches']):
                    assert b['action']==action and b['initial']==before
                    first=b['trace'][0];assert first['pricing_obs']==sample['obs']
                    for section in [slice(0,220),slice(230,250)]:np.testing.assert_array_equal(np.array(first['obs'])[section],np.array(before['obs'])[section])
                    np.testing.assert_array_equal(first['info']['orders_delivered'],before['accepted'])
                    np.testing.assert_array_equal(first['info']['orders_clipped'],before['clipped'])
                    value=audit(b['trace'],day,before['start'],keys,sku,action);assert value==b['value'];values.append(value);branches+=1
                    if rep==0:
                        assert r['replays'][action-3]==b['trace'];audit(r['replays'][action-3],day,before['start'],keys,sku,action)
                gains.append(values[0]-values[1])
            assert gains==sample['gains'] and sample['target']==float(np.mean(gains)/base[sku])
            np.testing.assert_array_equal(sample['features'],feature_matrix(sample['obs'],day,base)[sku]);samples[ep,rank]=sample
    ordered=[samples[ep,j] for ep in range(128) for j in [0,1]]
    assert len(samples)==256 and branches==16384 and days==freeze['days']==7168+67*length
    assert ordered==policy['samples'];x=np.array([r['features'] for r in ordered]);y=np.array([r['target'] for r in ordered])
    np.testing.assert_array_equal(x,policy['matrix']);np.testing.assert_array_equal(y,policy['targets']);check_ridge(x,y,weights)
    rows=[];firsts={};noise={}
    with (out/'assessment.jsonl').open() as f:
        for line in f:
            r=json.loads(line);key=r['key'];label=r['label'];trace=r['trace'];assert label in ['learned','baseline','heuristic'] and key in range(2520000,2520064)
            start=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859));keys=[trace[0]['demand_seed'],trace[0]['supplier_seed']]
            if key in noise:assert noise[key]==keys
            else:noise[key]=keys
            audit(trace,1,start,keys,label=label)
            demand=sum(sum(v['info']['demand']) for v in trace)
            result=dict(key=key,label=label,reward_day=sum(v['reward'] for v in trace)/28,
                fill=sum(sum(v['info']['sales']) for v in trace)/demand if demand else 1.,
                price_counts=np.bincount(np.concatenate([v['price_actions'] for v in trace if v['price_actions'] is not None]),minlength=5).tolist())
            assert result==r['result'];rows.append(result)
            if key==2520000:firsts[label]=trace
    seen=[]
    with (out/'assessment_replays.jsonl').open() as f:
        for line in f:
            r=json.loads(line);label=r['label'];assert r['key']==2520000 and label not in seen and r['trace']==firsts[label]
            seen.append(label);audit(r['trace'],1,r['trace'][0]['start'],noise[2520000],label=label)
    assert set(seen)=={'learned','baseline','heuristic'} and rows==s['rows'];check_analysis(rows,s['analysis'])
    assert days==s['days']==sel['planned_days'] and s['fit_seconds']==freeze['fit_seconds']<=7200
    assert s['assessment_seconds']<=600 and s['seconds']<=7800 and not s['test_scoring']
    write_json(out/'verification.json',dict(status='passed',scan_episodes=128,original_replays=128,states=256,
        paired_suffixes=16384,probes=256,fit_suffix_replays=512,assessment_episodes=192,assessment_replays=3,
        days=days,new_simulator_days=0,selection_model_freezes_features_ridge_actions_economics_intervals_replays=True,
        primary_pass=s['analysis']['primary_pass'],heuristic_pass=s['analysis']['heuristic_pass'],verifier_sha256=digest(__file__)))
    print('Stratified pricing independently verified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
