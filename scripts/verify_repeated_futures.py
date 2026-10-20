"""Verify saved repeated-future draws and independently recompute variance terms."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.algos.rollout_correction import RolloutCorrection,features
from scripts.diagnose_cover_search import CoverPolicy


def verify(output):
    out=Path(output);read=lambda name:json.loads((out/name).read_text())
    manifest=read('manifest.json');data=read('traces.json');summary=read('summary.json');stats=summary['statistics']
    for k,p in manifest['paths'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==manifest['hashes'][k]
    ck=json.loads(Path(manifest['paths']['checkpoint']).read_text());n,h=ck['n'],ck['horizon']
    center=InventoryPositionLinearPolicy.load(manifest['paths']['center'],n,h);target=CoverPolicy(n,h,ck['cover'])
    model=RolloutCorrection(center,target,n,h,ck['weights'],ck['scale'])
    assert len(data['records'])==8 and len(data['originals'])==4
    pred_all=[];samples_all=[];regrets=[];correct=[];days=branches=0
    for index,r in enumerate(data['records']):
        ep=index//2;day=8+7*(index%2);seed=960000+ep
        assert (r['seed'],r['day'])==(seed,day)
        start=int(np.random.default_rng(np.random.SeedSequence([seed,314159])).integers(15,1859))
        assert r['start']==start and start+27<=1885
        state=r['state'];obs=np.asarray(state['obs'],dtype=np.float32)
        assert state['day']==day and state['start']==start
        c=center.predict(obs);ref=np.clip(target.act(obs)[0],np.maximum(0,c-2),np.minimum(40,c+2))
        chosen=model.action(obs,day);changed=np.flatnonzero(chosen!=ref)
        sku=int(changed[0]) if len(changed) else (ep*2+index%2)%n
        assert r['sku']==sku and len(changed)<=1
        assert r['reference_action']==ref[sku] and r['chosen_action']==chosen[sku]
        actions=list(range(max(0,int(c[sku])-2),min(40,int(c[sku])+2)+1));assert r['actions']==actions
        p=np.array([float((features(obs,n,h,sku,a-int(ref[sku]),day)/model.scale)@model.weights) for a in actions])
        np.testing.assert_array_equal(p,r['predictions']);refidx=actions.index(int(ref[sku]));pick=actions.index(int(chosen[sku]))
        assert len(r['draws'])==16
        values=[];demand_paths=set()
        for rep,draw in enumerate(r['draws']):
            expected=[int(np.random.SeedSequence([960100,ep,day,rep,s]).generate_state(1,dtype=np.uint64)[0]) for s in (0,1)]
            assert draw['replicate']==rep and draw['keys']==expected
            bs=draw['branches'];assert [b['action'] for b in bs]==actions
            baseline=bs[refidx]['trace'];demand_paths.add(json.dumps([row['demand'] for row in baseline]))
            vals=[]
            for b in bs:
                trace=b['trace'];days+=len(trace);branches+=1
                assert [row['day'] for row in trace]==list(range(day,29))
                first=ref.copy();first[sku]=b['action'];assert trace[0]['actions']==first.tolist()
                assert trace[0]['reward']==baseline[0]['reward']
                for row,rr in zip(trace,baseline):
                    assert row['demand']==rr['demand'] and 0<=row['stock']<=500+1e-8
                    for a,ar,l,lr in zip(row['actions'],rr['actions'],row['leads'],rr['leads']):
                        if a>0 and ar>0:assert l==lr
                val=float(np.dot(.99**np.arange(len(trace)),[row['reward'] for row in trace]));assert val==b['value'];vals.append(val)
            values.append(vals)
            if rep==0:assert baseline==r['replicate0_replay']
        assert len(demand_paths)>1
        original=data['originals'][ep];assert original['seed']==seed and original['start']==start
        assert r['unresampled_probe']==original['trace'][day-1:]
        a=np.array(values);g=a-a[:,refidx,None];means=g.mean(0);variance=g.var(0,ddof=1);se=np.sqrt(variance/16)
        reported=stats['states'][index]
        for key,value in [('mean_gain',means),('variance',variance),('mc_se',se)]:np.testing.assert_array_equal(value,reported[key])
        half=t.ppf(.975,15)*se[pick]
        np.testing.assert_allclose(reported['chosen_ci95'],[means[pick]-half,means[pick]+half],rtol=1e-12,atol=1e-12)
        assert reported['chosen_gain']==means[pick]
        rank=int(np.argmax(p));regret=float(means.max()-means[rank]);best=bool(means[rank]>=means.max()-1e-9)
        assert reported['mean_rank_regret']==regret and reported['mean_rank_tied_best']==best
        regrets.append(regret);correct.append(best)
        mask=np.array(actions)!=ref[sku];pred_all.extend(p[mask]);samples_all.extend(g[:,mask].T)
    p=np.array(pred_all);y=np.array(samples_all);m=y.mean(1);v=y.var(1,ddof=1).mean()
    assert stats['nonreference_actions']==len(p)
    assert stats['mean_within_state_action_variance']==v and stats['mean_target_noise_variance']==v/16
    for label,pred in [('model',p),('zero',np.zeros_like(p))]:
        mean_mse=float(np.mean((pred-m)**2));single=float(np.mean((pred[:,None]-y)**2))
        expected=dict(mean_target_mse=mean_mse,single_future_mse=single,persistent_mse_estimate=mean_mse-v/16)
        assert expected==stats['errors'][label]
        np.testing.assert_allclose(single,mean_mse+15*v/16,rtol=1e-12)
    assert stats['mean_rank_regret']==np.mean(regrets) and stats['mean_rank_tied_best_rate']==np.mean(correct)
    assert summary['branch_days']==days and summary['branches']==branches
    assert summary['fits']==summary['validation_episodes']==0 and not summary['held_out_test_used']
    print('Verified hashes, state selection, future keys, paired traces, frozen predictions, replay and noise/error decomposition.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
