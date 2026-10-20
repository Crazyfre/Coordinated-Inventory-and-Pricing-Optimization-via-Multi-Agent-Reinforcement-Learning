"""Reconstruct subsets and final predictions for head-sharing diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.algos.local_feature_residual import LocalFeatureResidualActor


def verify(output):
    out=Path(output);s=json.loads((out/'summary.json').read_text());p=s['protocol']
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert all(sha(v)==p['hashes'][k] for k,v in p['paths'].items())
    assert all(sha(out/k)==v for k,v in s['hashes'].items())
    assert sha(out/'probabilities.npz')==s['probabilities_sha256']
    ds=np.load(p['paths']['data']);saved=np.load(out/'probabilities.npz')
    torch.set_num_threads(1)
    ck=torch.load(out/'initial.pt',weights_only=True);spec=ck['spec']
    torch.manual_seed(11);initial=LocalFeatureResidualActor(ck['obs_dim'],spec['n'],spec['pipe_horizon'],spec['coefficients'])
    assert all(torch.equal(v,initial.state_dict()[k]) for k,v in ck['state_dict'].items())
    with torch.no_grad():centers=initial.centers(torch.tensor(ds['fit_x'])).numpy()
    for i in range(spec['n']):
        candidates=[[j for j in range(len(centers)) if centers[j,i]<2 and ds['fit_y'][j,i]-centers[j,i]==k] for k in [0,1,2]]
        assert [len(c) for c in candidates]==p['availability'][str(i)]
        if min(map(len,candidates))<8:assert i in p['excluded']
        else:assert sum([c[:8] for c in candidates],[])==p['selected'][str(i)]
    for arm,result in s['results'].items():
        ck=torch.load(out/(arm+'.pt'),weights_only=True)
        assert ck['variant']==LocalFeatureResidualActor.variant
        actor=LocalFeatureResidualActor(ck['obs_dim'],spec['n'],spec['pipe_horizon'],spec['coefficients']);actor.load_state_dict(ck['state_dict'])
        # Match original union-batch shape for exact floating point checkpoint replay.
        heads=[int(i) for i in result['heads']];union=np.unique(np.concatenate([p['selected'][str(i)] for i in heads]))
        with torch.no_grad():distributions=actor(torch.tensor(ds['fit_x'][union]))
        for key,metrics in result['heads'].items():
            i=int(key);rows=np.array(p['selected'][key]);prob=distributions[i].probs[np.searchsorted(union,rows)].numpy()
            np.testing.assert_array_equal(prob,saved[arm+'_head_'+key]);np.testing.assert_allclose(prob.sum(-1),1,atol=1e-6)
            c=centers[rows,i];support=(np.arange(41)>=np.maximum(c-2,0)[:,None])&(np.arange(41)<=np.minimum(c+2,40)[:,None])
            np.testing.assert_array_equal(prob>0,support)
            y=ds['fit_y'][rows,i];pred=prob.argmax(-1);pt=prob[np.arange(24),y]
            for k,v in dict(accuracy=(pred==y).mean(),target_probability=pt.mean(),nll=-np.log(pt).mean()).items():np.testing.assert_allclose(metrics[k],v)
            for k in range(3):
                mask=y-c==k;r=metrics['classes'][str(k)];assert r['count']==8==mask.sum()
                np.testing.assert_allclose(r['accuracy'],(pred[mask]==y[mask]).mean())
                np.testing.assert_allclose(r['target_probability'],pt[mask].mean())
            assert result['gates'][key]==bool(metrics['accuracy']>=.99 and metrics['target_probability']>=.99)
        assert [r['update'] for r in s['histories'][arm]]==list(range(250,2001,250))
    assert s['joint_gate']==all(s['results']['joint']['gates'].values())
    assert s['separate_gate']==all(s['results']['separate_'+i]['gates'][i] for i in p['selected'])
    assert s['total_decisions']==24*len(p['selected'])
    assert s['total_optimizer_updates']==2000*(1+len(p['selected']))
    assert s['unique_states']==len(set(sum(p['selected'].values(),[])))
    result=dict(passed=True,heads=len(p['selected']),decisions=s['total_decisions'],
        checks='hashes, availability/selection, seeded initialization, exact checkpoints/probabilities, support, every final metric/gate and budgets')
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');verify(p.parse_args().output)
