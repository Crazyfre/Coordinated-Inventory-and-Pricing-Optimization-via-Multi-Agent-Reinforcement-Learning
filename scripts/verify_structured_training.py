"""Independently reconstruct all newly trained structured-cohort updates."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.train_compact_residual import digest,write_json
from scripts.verify_parameter_confirmation import episodes,check_episode


def verify(output):
    root=Path(output);load=lambda p:json.loads(p.read_text())
    manifest=load(root/'manifest.json');summary=load(root/'summary.json')
    for p,h in manifest['hashes'].items():assert digest(p)==h
    for r in manifest['existing_main'].values():assert digest(r['path'])==r['sha256']
    total=0;updates=0
    assert len(summary['runs'])==25
    for run in summary['runs']:
        condition,seed=run['condition'],run['seed'];j=manifest['seeds'].index(seed)
        folder=root/f'{condition}_{seed}';d=1 if condition=='pooled' else 10
        for name,field in [('traces.jsonl','trace_sha256'),('history.json','history_sha256'),('final.json','checkpoint_sha256')]:
            assert digest(folder/name)==run[field]
        rng=np.random.default_rng(seed);theta=np.full(d,7.5);m=np.zeros(d);v=np.zeros(d)
        history=load(folder/'history.json');assert len(history)==64;params={};expected={}
        for u,record in enumerate(history):
            e=rng.normal(size=(8,d));np.testing.assert_array_equal(e,record['epsilon'])
            np.testing.assert_array_equal(theta,record['mean_before']);diff=[]
            for pair,r in enumerate(record['records']):
                t=u*8+pair;base={'main':1800000,'pooled':1900000,'independent':2000000}[condition]+j*2000
                plus=base+(2*t if condition=='independent' else t);minus=plus+1 if condition=='independent' else plus
                assert (r['plus_key'],r['minus_key'])==(plus,minus)
                for sign,direction,key in [('plus',1,plus),('minus',-1,minus)]:
                    values=theta+direction*.75*e[pair]
                    label=f'u{u}_p{pair}_{sign}'
                    params[label,key]=np.repeat(values,10) if d==1 else values.copy();expected[label,key]=r[sign]
                diff.append(r['plus']['discounted_day']-r['minus']['discounted_day'])
            np.testing.assert_array_equal(diff,record['differences'])
            g=np.mean(np.array(diff)[:,None]*e/1.5,axis=0);np.testing.assert_array_equal(g,record['gradient'])
            m=.9*m+.1*g;v=.999*v+.001*g**2
            theta=(theta+.1*(m/(1-.9**(u+1)))/(np.sqrt(v/(1-.999**(u+1)))+1e-8)).clip(0,16)
            for field,value in [('mean',theta),('m',m),('v',v)]:np.testing.assert_array_equal(value,record['state'][field])
            updates+=1
        np.testing.assert_array_equal(theta,load(folder/'final.json')['mean'])
        count=0;paired=None;seen=set()
        for key,trace in episodes(folder/'traces.jsonl'):
            assert key not in seen;seen.add(key)
            start=int(np.random.default_rng(np.random.SeedSequence([key[1],314159])).integers(15,1859))
            metrics=check_episode(trace,params[key],start)
            for metric,value in metrics.items():assert abs(value-expected[key][metric])<1e-10
            if condition!='independent':
                noise=[r['info']['demand'] for r in trace]
                if key[0].endswith('_plus'):paired=noise
                else:assert noise==paired
            count+=1
        assert count==1024 and seen==params.keys() and run['seconds']<=1200
        total+=count;print(f'Verified {condition} seed{seed}',flush=True)
    assert total==25600 and updates==1600 and summary['counters']==dict(episodes=25600,days=716800)
    assert summary['seconds']<=30000 and not summary['test_scoring']
    write_json(root/'verification.json',dict(status='passed',new_training_episodes=total,reconstructed_updates=updates,
        source_checkpoint_trace_hashes=True,independent_actions_returns_gradients=True,new_simulator_days=0,
        test_scoring=False,verifier_sha256=digest(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
