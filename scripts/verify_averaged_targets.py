"""Reconstruct matched labels, ridge fits and independent prediction comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.algos.rollout_correction import RolloutCorrection
from scripts.diagnose_cover_search import CoverPolicy


def verify(output):
    out=Path(output);read=lambda name:json.loads((out/name).read_text())
    manifest=read('manifest.json');summary=read('summary.json')
    for k,p in manifest['paths'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==manifest['hashes'][k]
    ck=json.loads(Path(manifest['paths']['checkpoint']).read_text());n,h=ck['n'],ck['horizon']
    center=InventoryPositionLinearPolicy.load(manifest['paths']['center'],n,h);target=CoverPolicy(n,h,ck['cover'])
    policy=RolloutCorrection(center,target,n,h,ck['weights'],ck['scale']);arrays={};datasets={}
    for label,seed0,count,reps,namespace in [('fit',970000,16,8,972100),('assessment',971000,8,16,973100)]:
        data=read(label+'_traces.json');datasets[label]=data;assert len(data['records'])==2*count and len(data['originals'])==count
        xs=[];ys=[];avgs=[];noise=[];mask=[];branches=days=0
        for index,r in enumerate(data['records']):
            ep=index//2;seed=seed0+ep;day=8+7*(index%2)
            assert (r['seed'],r['day'])==(seed,day)
            start=int(np.random.default_rng(np.random.SeedSequence([seed,314159])).integers(15,1859))
            assert r['start']==start and start+27<=1885
            obs=np.asarray(r['state']['obs'],dtype=np.float32);c=center.predict(obs)
            ref=np.clip(target.act(obs)[0],np.maximum(0,c-2),np.minimum(40,c+2));chosen=policy.action(obs,day)
            changed=np.flatnonzero(chosen!=ref);assert len(changed)<=1
            sku=int(changed[0]) if len(changed) else (ep*2+index%2)%n
            assert r['sku']==sku and r['chosen_action']==chosen[sku] and r['reference_action']==ref[sku]
            actions=list(range(max(0,int(c[sku])-2),min(40,int(c[sku])+2)+1));assert r['actions']==actions
            ri=actions.index(int(ref[sku]));values=[];assert len(r['draws'])==reps
            for rep,draw in enumerate(r['draws']):
                keys=[int(np.random.SeedSequence([namespace,ep,day,rep,s]).generate_state(1,dtype=np.uint64)[0]) for s in (0,1)]
                assert draw['keys']==keys and draw['replicate']==rep
                bs=draw['branches'];assert [b['action'] for b in bs]==actions
                baseline=bs[ri]['trace'];vals=[]
                for b in bs:
                    trace=b['trace'];branches+=1;days+=len(trace)
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
            original=data['originals'][ep];assert original['seed']==seed and original['start']==start
            assert r['unresampled_probe']==original['trace'][day-1:]
            v=np.asarray(values);g=v-v[:,ri,None]
            f=obs[-n:].astype(float);pos=obs[:n]+obs[n:n+n*h].reshape(n,h).sum(1);log=lambda x:np.sign(x)*np.log1p(abs(x))
            base=np.array([1.,log(f[sku]/5),log(pos[sku]/5),float(obs[-2*n+sku])/500,(29-day)/28,f[sku]/max(float(f.sum()),1e-12)])
            for j,a in enumerate(actions):
                delta=a-int(ref[sku]);xs.append(np.r_[delta*base,delta*delta*base]);ys.append(g[0,j]);avgs.append(g[:,j].mean())
                noise.append(g[:,j].var(ddof=1)/reps);mask.append(j!=ri)
        arrays[label]=tuple(map(np.array,[xs,ys,avgs,noise,mask]))
        assert summary['counts'][label]==dict(states=2*count,branches=branches,branch_days=days)
    fx,fy,fa,_,_=arrays['fit'];fit=np.load(out/'fit_data.npz')
    for key,value in [('x',fx),('single',fy),('averaged',fa)]:np.testing.assert_array_equal(value,fit[key])
    ex,_,ey,noise,mask=arrays['assessment'];saved=np.load(out/'assessment_data.npz')
    for key,value in [('x',ex),('y',ey),('noise',noise),('mask',mask)]:np.testing.assert_array_equal(value,saved[key])
    scale=np.maximum(np.sqrt(np.mean(fx*fx,axis=0)),1e-8);z=fx/scale;predictions={'zero':np.zeros(len(ey))}
    for label,labels in [('single',fy),('averaged',fa)]:
        checkpoint=read(label+'_checkpoint.json');w=np.linalg.solve(z.T@z+10*np.eye(12),z.T@labels)
        np.testing.assert_array_equal(w,checkpoint['weights']);np.testing.assert_array_equal(scale,checkpoint['scale'])
        assert hashlib.sha256((out/(label+'_checkpoint.json')).read_bytes()).hexdigest()==summary['checkpoint_hashes'][label]
        predictions[label]=ex/scale@w;np.testing.assert_array_equal(predictions[label],saved[label])
    episode_errors={};result=summary['result']
    for label,p in predictions.items():
        offset=0;errors=[];regrets=[];best=[]
        for r in datasets['assessment']['records']:
            end=offset+len(r['actions']);m=mask[offset:end];pr=p[offset:end];y=ey[offset:end]
            errors.append(float(np.mean((pr[m]-y[m])**2)));rank=int(np.argmax(pr));regrets.append(float(y.max()-y[rank]));best.append(y[rank]>=y.max()-1e-9);offset=end
        ep=np.array(errors).reshape(8,2).mean(1);episode_errors[label]=ep;reported=result['models'][label]
        assert reported['episode_equal_mse']==ep.mean() and reported['action_pooled_mse']==np.mean((p[mask]-ey[mask])**2)
        np.testing.assert_array_equal(errors,reported['state_mse']);np.testing.assert_array_equal(ep,reported['episode_mse'])
        assert reported['mean_rank_regret']==np.mean(regrets) and reported['tied_best_rate']==np.mean(best)
    delta=episode_errors['averaged']-episode_errors['single'];half=t.ppf(.975,7)*delta.std(ddof=1)/np.sqrt(8)
    np.testing.assert_allclose(result['difference_ci95'],[delta.mean()-half,delta.mean()+half],rtol=1e-12)
    assert result['averaged_minus_single_mse']==delta.mean() and result['prediction_gate']==bool(delta.mean()+half<0)
    assert result['mean_target_noise']==noise[mask].mean() and result['nonreference_actions']==int(mask.sum())
    assert summary['fits']==2 and summary['validation_episodes']==0 and not summary['held_out_test_used']
    print('Verified provenance, independent states/futures, full traces, both exact ridge fits, predictions and episode-paired gate.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
