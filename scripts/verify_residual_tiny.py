"""Independent selection and checkpoint audit for tiny-set fitting."""
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
    assert all(sha(path)==p['hashes'][key] for key,path in p['paths'].items())
    assert all(sha(out/name)==digest for name,digest in s['artifact_hashes'].items())
    data=np.load(p['paths']['data']);subset=np.load(out/'subset.npz');saved=np.load(out/'probabilities.npz')
    torch.set_num_threads(1)
    for name in ['initial','final']:
        ck=torch.load(out/(name+'.pt'),weights_only=True);spec=ck['spec']
        assert ck['variant']==LocalFeatureResidualActor.variant
        actor=LocalFeatureResidualActor(ck['obs_dim'],spec['n'],spec['pipe_horizon'],spec['coefficients'])
        actor.load_state_dict(ck['state_dict']);sku=p['sku']
        with torch.no_grad():
            centers=actor.centers(torch.tensor(data['fit_x'])).numpy()
            probabilities=actor(torch.tensor(subset['observations']))[sku].probs.numpy()
        selected=[]
        for offset in [0,1,2]:
            selected.extend([i for i in range(len(centers)) if centers[i,sku]<2 and data['fit_y'][i,sku]-centers[i,sku]==offset][:8])
        assert len(selected)==24 and selected==p['indices']
        np.testing.assert_array_equal(selected,subset['indices'])
        np.testing.assert_array_equal(data['fit_x'][selected],subset['observations'])
        y=data['fit_y'][selected,sku];np.testing.assert_array_equal(y,subset['targets'])
        offsets=y-centers[selected,sku];np.testing.assert_array_equal(offsets,subset['offsets'])
        np.testing.assert_array_equal(probabilities,saved[name]);np.testing.assert_allclose(probabilities.sum(-1),1,atol=1e-6)
        c=centers[selected,sku];support=(np.arange(41)>=np.maximum(c-2,0)[:,None])&(np.arange(41)<=np.minimum(c+2,40)[:,None])
        np.testing.assert_array_equal(probabilities>0,support)
        pred=probabilities.argmax(-1);target_p=probabilities[np.arange(24),y]
        for key,value in dict(accuracy=(pred==y).mean(),target_probability=target_p.mean(),nll=-np.log(target_p).mean()).items():
            np.testing.assert_allclose(s[name][key],value)
        for offset in [0,1,2]:
            mask=offsets==offset;row=s[name]['classes'][str(offset)];assert row['count']==8==mask.sum()
            np.testing.assert_allclose(row['accuracy'],(pred[mask]==y[mask]).mean())
            np.testing.assert_allclose(row['target_probability'],target_p[mask].mean())
    assert s['gate']==bool(s['final']['accuracy']>=.99 and s['final']['target_probability']>=.99)
    assert [r['update'] for r in s['history']]==list(range(100,2001,100))
    assert len(set(p['indices']))==24
    result=dict(passed=True,selected_decisions=24,updates=2000,new_episodes=0,
        checks='hashes, deterministic balanced selection, archived rows/labels, initial/final checkpoint predictions, support, metrics and gate')
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');verify(p.parse_args().output)
