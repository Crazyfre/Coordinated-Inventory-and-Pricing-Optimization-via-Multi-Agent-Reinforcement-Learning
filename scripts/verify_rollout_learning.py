"""Independent archive recomputation for the fixed reward-learning pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t


def verify(output):
    out=Path(output)
    read=lambda name:json.loads((out/name).read_text())
    manifest=read('manifest.json');summary=read('summary.json');ck=read('checkpoint.json')
    for key,path in manifest['paths'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==manifest['hashes'][key]
    assert hashlib.sha256((out/'checkpoint.json').read_bytes()).hexdigest()==summary['checkpoint_sha256']
    data=read('training_traces.json');fit=np.load(out/'fit_data.npz')
    assert len(data['records'])==256 and len(data['originals'])==64
    originals={r['seed']:r for r in data['originals']}
    xs=[];ys=[];branch_days=0
    n,h=ck['n'],ck['horizon']
    for index,rec in enumerate(data['records']):
        seed=930000+index//4;day=1+7*(index%4);sku=index%n
        assert (rec['seed'],rec['day'],rec['sku'])==(seed,day,sku)
        start=int(np.random.default_rng(np.random.SeedSequence([seed,314159])).integers(15,1859))
        assert rec['start']==originals[seed]['start']==start and start+27<=1885
        ref=next(b for b in rec['branches'] if b['action']==rec['reference_action'])
        assert ref['trace']==originals[seed]['trace'][day-1:]
        assert [b['action'] for b in rec['branches']]==list(range(max(0,rec['center']-2),min(40,rec['center']+2)+1))
        obs=np.array(rec['obs'],dtype=np.float32)
        forecast=obs[-n:].astype(float)
        pos=obs[:n]+obs[n:n+n*h].reshape(n,h).sum(1)
        log=lambda x:np.sign(x)*np.log1p(abs(x))
        base=np.array([1,log(forecast[sku]/5),log(pos[sku]/5),obs[-2*n+sku]/500,
                       (29-day)/28,forecast[sku]/max(float(forecast.sum()),1e-12)])
        for b in rec['branches']:
            trace=b['trace'];branch_days+=len(trace)
            assert [r['day'] for r in trace]==list(range(day,29))
            value=float(np.dot(.99**np.arange(len(trace)),[r['reward'] for r in trace]))
            assert value==b['value']
            assert trace[0]['reward']==ref['trace'][0]['reward']
            for step,(r,rr) in enumerate(zip(trace,ref['trace'])):
                assert r['demand']==rr['demand'] and 0<=r['stock']<=500+1e-8
                for j,(a,ar) in enumerate(zip(r['actions'],rr['actions'])):
                    if step==0:assert a==(b['action'] if j==sku else ar)
                    if a>0 and ar>0:assert r['leads'][j]==rr['leads'][j]
            delta=b['action']-rec['reference_action']
            xs.append(np.r_[delta*base,delta*delta*base]);ys.append(value-ref['value'])
    x,y=np.array(xs),np.array(ys)
    np.testing.assert_allclose(x,fit['x'],rtol=1e-7,atol=1e-8)
    np.testing.assert_array_equal(y,fit['y'])
    scale=np.maximum(np.sqrt(np.mean(fit['x']**2,axis=0)),1e-8)
    z=fit['x']/scale
    weights=np.linalg.solve(z.T@z+10*np.eye(12),z.T@y)
    np.testing.assert_array_equal(scale,ck['scale']);np.testing.assert_array_equal(weights,ck['weights'])
    assert summary['training_branch_days']==branch_days and summary['fit_samples']==len(y)
    assert summary['training_rmse']==float(np.sqrt(np.mean((z@weights-y)**2)))
    traces=read('validation_traces.json');rows={}
    assert traces['learned']==traces['replay']
    for label,trace in traces.items():
        assert [r['demand'] for r in trace]==[r['demand'] for r in traces['initial']]
        rows[label]=[]
        for i,seed in enumerate(range(940000,940040)):
            rs=trace[i*28:(i+1)*28]
            assert len(rs)==28 and all(r['seed']==seed for r in rs)
            assert [r['day'] for r in rs]==list(range(1,29))
            rows[label].append(dict(seed=seed,reward_day=sum(r['reward'] for r in rs)/28,
                                    fill=sum(sum(r['sales']) for r in rs)/sum(sum(r['demand']) for r in rs)))
        assert rows[label]==summary['episode_rows'][label]
        for metric in ['reward_day','fill']:
            assert summary['means'][label][metric]==float(np.mean([r[metric] for r in rows[label]]))
    gates=[]
    for label,comparison in summary['comparisons'].items():
        lows={}
        for key,metric in [('reward','reward_day'),('fill','fill')]:
            delta=np.array([a[metric]-b[metric] for a,b in zip(rows['learned'],rows[label])])
            width=t.ppf(.975,39)*delta.std(ddof=1)/np.sqrt(40)
            bounds=[delta.mean()-width,delta.mean()+width]
            np.testing.assert_allclose(bounds,comparison[key+'_ci95'],rtol=0,atol=1e-12)
            assert comparison[key+'_delta']==float(delta.mean())
            lows[key]=bounds[0]
        gate=bool(lows['reward']>0 and lows['fill']>=-.005)
        assert summary['gates'][label]==gate;gates.append(gate)
    assert summary['all_comparators_gate']==all(gates)
    assert summary['ppo_updates']==0 and summary['reward_regression_fits']==1 and not summary['held_out_test_used']
    print('Verified provenance, train-only windows, branch pairing, reward targets, ridge checkpoint, replay and all paired gates.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
