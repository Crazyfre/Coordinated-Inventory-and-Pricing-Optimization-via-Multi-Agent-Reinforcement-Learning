"""Reconstruct frozen predictions, returns and grouped diagnostic statistics."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.stats import t
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from scripts.diagnose_cover_search import CoverPolicy
from src.algos.rollout_correction import RolloutCorrection


def verify(output):
    out=Path(output);read=lambda name:json.loads((out/name).read_text())
    manifest=read('manifest.json');data=read('traces.json');summary=read('summary.json')
    for k,p in manifest['paths'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==manifest['hashes'][k]
    ck=json.loads(Path(manifest['paths']['checkpoint']).read_text());n,h=ck['n'],ck['horizon']
    center=InventoryPositionLinearPolicy.load(manifest['paths']['center'],n,h)
    target=CoverPolicy(n,h,ck['cover']);model=RolloutCorrection(center,target,n,h,ck['weights'],ck['scale'])
    originals={(r['source'],r['seed']):r for r in data['originals']}
    groups={};branch_count=branch_days=0
    assert len(data['records'])==128 and len(originals)==32
    for index,r in enumerate(data['records']):
        source=['reference','learned'][index//64];ep=(index%64)//4;day=1+7*(index%4);seed=950000+ep
        assert (r['source'],r['seed'],r['day'])==(source,seed,day)
        start=int(np.random.default_rng(np.random.SeedSequence([seed,314159])).integers(15,1859))
        assert r['start']==originals[(source,seed)]['start']==start and start+27<=1885
        obs=np.asarray(r['obs'],dtype=np.float32);c=center.predict(obs)
        ref=np.clip(target.act(obs)[0],np.maximum(0,c-2),np.minimum(40,c+2))
        chosen=model.action(obs,day);changed=np.flatnonzero(chosen!=ref)
        sku=int(changed[0]) if len(changed) else (ep*4+(day-1)//7)%n
        assert len(changed)<=1 and r['sku']==sku and r['center']==c[sku]
        assert r['reference_action']==ref[sku] and r['chosen_action']==chosen[sku]
        candidates=list(range(max(0,int(c[sku])-2),min(40,int(c[sku])+2)+1))
        f=obs[-n:].astype(float);pos=obs[:n]+obs[n:n+n*h].reshape(n,h).sum(1)
        log=lambda x:np.sign(x)*np.log1p(abs(x))
        base=np.array([1.,log(f[sku]/5),log(pos[sku]/5),float(obs[-2*n+sku])/500,(29-day)/28,f[sku]/max(float(f.sum()),1e-12)])
        predictions=[]
        for a in candidates:
            delta=a-int(ref[sku]);x=np.r_[delta*base,delta*delta*base]
            predictions.append(float((x/np.array(ck['scale']))@np.array(ck['weights'])))
        np.testing.assert_allclose(predictions,r['predictions'],rtol=1e-12,atol=1e-12)
        for cont,bs in r['branches'].items():
            assert [b['action'] for b in bs]==candidates
            refbranch=next(b for b in bs if b['action']==ref[sku])
            for b in bs:
                trace=b['trace'];branch_count+=1;branch_days+=len(trace)
                assert [row['day'] for row in trace]==list(range(day,29))
                assert b['value']==float(np.dot(.99**np.arange(len(trace)),[row['reward'] for row in trace]))
                assert trace[0]['reward']==refbranch['trace'][0]['reward']
                for step,(row,rr) in enumerate(zip(trace,refbranch['trace'])):
                    assert row['demand']==rr['demand'] and 0<=row['stock']<=500+1e-8
                    if step==0:
                        action=ref.copy();action[sku]=b['action']
                        assert row['actions']==action.tolist()
                    for a,ar,l,lr in zip(row['actions'],rr['actions'],row['leads'],rr['leads']):
                        if a>0 and ar>0:assert l==lr
            if cont==source:
                a=ref[sku] if source=='reference' else chosen[sku]
                assert next(b['trace'] for b in bs if b['action']==a)==originals[(source,seed)]['trace'][day-1:]
            y=np.array([b['value']-refbranch['value'] for b in bs]);p=np.array(r['predictions'])
            mask=np.array(candidates)!=ref[sku];rank=int(np.argmax(p));pick=candidates.index(int(chosen[sku]))
            group=groups.setdefault((source,cont),dict(p=[],y=[],regret=[],best=[],gain=[],active=0))
            group['p'].extend(p[mask]);group['y'].extend(y[mask]);group['regret'].append(float(y.max()-y[rank]))
            group['best'].append(bool(y[rank]>=y.max()-1e-9));group['gain'].append(float(y[pick]));group['active']+=int(chosen[sku]!=ref[sku])
    def check_interval(values,reported):
        a=np.asarray(values);mean=a.mean();half=t.ppf(.975,15)*a.std(ddof=1)/4
        np.testing.assert_allclose([mean,mean-half,mean+half],[reported['mean']]+reported['ci95'],rtol=1e-12,atol=1e-12)
    for (source,cont),g in groups.items():
        m=summary['metrics'][source][cont];p,y=np.array(g['p']),np.array(g['y'])
        expected=dict(nonreference_actions=len(y),rmse=float(np.sqrt(np.mean((p-y)**2))),zero_rmse=float(np.sqrt(np.mean(y*y))),
                      bias=float(np.mean(p-y)),mean_realized_ranking_regret=float(np.mean(g['regret'])),
                      realized_tied_best_rate=float(np.mean(g['best'])),active_states=g['active'])
        for k,v in expected.items():assert m[k]==v
        correlation=float(np.corrcoef(p,y)[0,1]) if p.std()>0 and y.std()>0 else None
        assert m['correlation']==correlation
        means=np.array(g['gain']).reshape(16,4).mean(1)
        np.testing.assert_array_equal(means,m['scenario_mean_chosen_gain']);check_interval(means,m['chosen_gain'])
    for source in ['reference','learned']:
        delta=np.array(groups[source,'learned']['gain']).reshape(16,4).mean(1)-np.array(groups[source,'reference']['gain']).reshape(16,4).mean(1)
        check_interval(delta,summary['metrics'][source]['continuation_delta'])
    for seed in range(950000,950016):
        assert [r['demand'] for r in originals['reference',seed]['trace']]==[r['demand'] for r in originals['learned',seed]['trace']]
    assert summary['branches']==branch_count and summary['branch_days']==branch_days and summary['states']==128
    assert summary['fits']==summary['validation_episodes']==0 and not summary['held_out_test_used']
    print('Verified frozen hashes, prescribed states/actions, predictions, paired traces, returns and scenario-clustered metrics.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
