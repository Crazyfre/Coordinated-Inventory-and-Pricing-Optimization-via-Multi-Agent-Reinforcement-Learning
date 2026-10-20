"""Archive verification for full-horizon matched arrival features."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from scripts.train_rollout_correction import digest
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from scripts.diagnose_cover_search import CoverPolicy
from src.algos.rollout_correction import RolloutCorrection


def verify(output):
    out=Path(output);read=lambda f:json.loads((out/f).read_text());manifest=read('manifest.json');summary=read('summary.json')
    for k,p in manifest['paths'].items():assert digest(p)==manifest['hashes'][k]
    hashes=read('trace_hashes.json');ck=json.loads(Path(manifest['paths']['checkpoint']).read_text());n,h=ck['n'],ck['horizon']
    center=InventoryPositionLinearPolicy.load(manifest['paths']['center'],n,h);target=CoverPolicy(n,h,ck['cover']);policy=RolloutCorrection(center,target,n,h,ck['weights'],ck['scale'])
    datasets={};metadata={}
    for phase,base,reps,namespace in [('fit',990000,4,992100),('assessment',991000,8,993100)]:
        xs={'legacy':[],'arrival':[]};ys=[];mask=[];records=[];branches=days=probes=0
        for ep in range(16):
            path=out/f'{phase}_{ep:03d}.json';assert digest(path)==hashes[path.name];data=json.loads(path.read_text());assert len(data['records'])==5
            chosen_days=[1+ep%7,8+ep%7,15+ep%7,22+ep%5,27+ep%2]
            for slot,r in enumerate(data['records']):
                day=chosen_days[slot];seed=base+ep
                start=int(np.random.default_rng(np.random.SeedSequence([seed,314159])).integers(15,1859))
                assert (r['seed'],r['day'],r['start'])==(seed,day,start) and start+27<=1885
                obs=np.array(r['state']['obs'],dtype=np.float32);c=center.predict(obs);ref=np.clip(target.act(obs)[0],np.maximum(c-2,0),np.minimum(c+2,40));chosen=policy.action(obs,day)
                changed=np.flatnonzero(chosen!=ref);sku=int(changed[0]) if len(changed) else (ep*5+slot)%n
                assert len(changed)<=1 and r['sku']==sku and r['reference_action']==ref[sku] and r['chosen_action']==chosen[sku]
                actions=list(range(max(0,int(c[sku])-2),min(40,int(c[sku])+2)+1));assert actions==r['actions'];ri=actions.index(int(ref[sku]));values=[]
                assert len(r['draws'])==reps
                for rep,d in enumerate(r['draws']):
                    assert d['keys']==[int(np.random.SeedSequence([namespace,ep,day,rep,s]).generate_state(1,dtype=np.uint64)[0]) for s in (0,1)]
                    bs=d['branches'];assert [b['action'] for b in bs]==actions;baseline=bs[ri]['trace'];v=[]
                    for b in bs:
                        trace=b['trace'];branches+=1;days+=len(trace);assert [row['day'] for row in trace]==list(range(day,29))
                        first=ref.copy();first[sku]=b['action'];assert trace[0]['actions']==first.tolist() and trace[0]['reward']==baseline[0]['reward']
                        for row,rr in zip(trace,baseline):
                            assert row['demand']==rr['demand'] and 0<=row['stock']<=500+1e-8
                            for a,ar,l,lr in zip(row['actions'],rr['actions'],row['leads'],rr['leads']):
                                if a>0 and ar>0:assert l==lr
                        value=float(np.dot(.99**np.arange(len(trace)),[row['reward'] for row in trace]));assert b['value']==value;v.append(value)
                    if rep==0:assert baseline==r['replicate0_replay']
                    values.append(v)
                original=data['originals'][0];assert original['seed']==seed and original['start']==start
                assert r['unresampled_probe']==original['trace'][day-1:];probes+=len(r['unresampled_probe'])
                v=np.array(values);g=v-v[:,ri,None]
                if day>=27:np.testing.assert_array_equal(g,np.zeros_like(g))
                log=lambda x:np.sign(x)*np.log1p(np.abs(x))
                stock=obs[:n].astype(float);pipe=obs[n:n+n*h].reshape(n,h);forecast=obs[-n:].astype(float);price=obs[-4*n:-3*n].astype(float)
                per=np.column_stack([log(stock/5),log(pipe.astype(float)/5),log(forecast/5),log(price)])
                context=np.r_[1.,per.ravel(),np.eye(n)[sku],per[sku],float(obs[-2*n+sku])/500,(29-day)/28]
                position=obs[:n]+pipe.sum(1)
                old=np.array([1.,log(forecast[sku]/5),log(position[sku]/5),float(obs[-2*n+sku])/500,(29-day)/28,forecast[sku]/max(float(forecast.sum()),1e-12)])
                for j,a in enumerate(actions):
                    delta=a-int(ref[sku])
                    for name,ctx in [('legacy',old),('arrival',context)]:
                        feature=np.r_[delta*ctx,delta*delta*ctx];xs[name].append(np.zeros_like(feature) if day>=27 else feature)
                    ys.append(float(g[:,j].mean()));mask.append(j!=ri)
                records.append(dict(actions=actions,day=day))
        datasets[phase]=({k:np.array(v) for k,v in xs.items()},np.array(ys),np.array(mask));metadata[phase]=records
        assert summary['counts'][phase]==dict(states=80,branches=branches,branch_days=days,probe_days=probes,replay_days=probes,original_days=448)
        assert set(r['day'] for r in records)==set(range(1,29))
    fx,fy,fmask=datasets['fit'];ex,ey,mask=datasets['assessment'];fit=np.load(out/'fit_data.npz');ev=np.load(out/'assessment_data.npz');predictions={'zero':np.zeros(len(ey))}
    for name in fx:
        np.testing.assert_array_equal(fx[name],fit[name]);np.testing.assert_array_equal(ex[name],ev['x_'+name])
        scale=np.maximum(np.sqrt(np.mean(fx[name]**2,axis=0)),1e-8);z=fx[name]/scale;w=np.linalg.solve(z.T@z+10*np.eye(z.shape[1]),z.T@fy)
        m=read(name+'_checkpoint.json');np.testing.assert_array_equal(w,m['weights']);np.testing.assert_array_equal(scale,m['scale'])
        assert digest(out/(name+'_checkpoint.json'))==summary['checkpoint_hashes'][name]
        predictions[name]=ex[name]/scale@w;np.testing.assert_array_equal(predictions[name],ev[name])
    np.testing.assert_array_equal(fy,fit['y']);np.testing.assert_array_equal(ey,ev['y']);np.testing.assert_array_equal(mask,ev['mask'])
    errors={}
    for name,p in predictions.items():
        offset=0;state=[]
        for r in metadata['assessment']:
            end=offset+len(r['actions']);m=mask[offset:end];state.append(float(np.mean((p[offset:end][m]-ey[offset:end][m])**2)));offset=end
        strata=np.array(state).reshape(16,5);episode=strata[:,:4].mean(1);errors[name]=episode;reported=summary['result']['models'][name]
        np.testing.assert_array_equal(strata.mean(0),reported['stratum_mse']);np.testing.assert_array_equal(episode,reported['episode_mse']);assert episode.mean()==reported['primary_mse']
    passes=[]
    for name in ['legacy','zero']:
        d=errors['arrival']-errors[name];half=t.ppf(.975,15)*d.std(ddof=1)/4;gate=summary['result']['gates'][name]
        np.testing.assert_allclose(gate['ci95'],[d.mean()-half,d.mean()+half],rtol=1e-12);assert gate['delta']==d.mean() and gate['passed']==bool(d.mean()+half<0);passes.append(gate['passed'])
    assert summary['result']['diagnostic_gate']==all(passes) and summary['validation_episodes']==0 and not summary['held_out_test_used']
    print('Verified full day coverage, traces, independent feature reconstruction, exact fits, terminal zeros and paired gates.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
