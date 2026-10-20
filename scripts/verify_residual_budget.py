"""Audit fixed-budget checkpoints, natural sampling and fidelity metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.algos.local_feature_residual import LocalFeatureResidualActor


def verify(output):
    out=Path(output);s=json.loads((out/'summary.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert all(sha(v)==s['hashes'][k] for k,v in s['paths'].items())
    assert all(sha(out/k)==v for k,v in s['artifact_hashes'].items())
    ds=np.load(s['paths']['data']);saved=np.load(out/'probabilities.npz');order=np.load(out/'batches.npy')
    g=torch.Generator().manual_seed(11)
    expected=torch.cat([torch.randperm(1792,generator=g).reshape(14,128) for _ in range(143)])[:2000]
    np.testing.assert_array_equal(order,expected)
    prior=torch.load(s['paths']['prior'],weights_only=True);torch.set_num_threads(1)
    for update in [560,2000]:
        ck=torch.load(out/f'update_{update}.pt',weights_only=True);spec=ck['spec']
        assert ck['variant']==LocalFeatureResidualActor.variant and ck['update']==update
        if update==560:assert all(torch.equal(v,prior['state_dict'][k]) for k,v in ck['state_dict'].items())
        assert all(int(v['step'])==update for v in ck['optimizer']['state'].values())
        actor=LocalFeatureResidualActor(ck['obs_dim'],spec['n'],spec['pipe_horizon'],spec['coefficients']);actor.load_state_dict(ck['state_dict'])
        for split in ['fit','holdout']:
            x=torch.tensor(ds[split+'_x']);y=ds[split+'_y']
            with torch.no_grad():p=np.stack([d.probs.numpy() for d in actor(x)],1);c=actor.centers(x).numpy()
            np.testing.assert_array_equal(p,saved[f'{update}_{split}']);np.testing.assert_allclose(p.sum(-1),1,atol=1e-6)
            support=(np.arange(41)>=np.maximum(c-2,0)[...,None])&(np.arange(41)<=np.minimum(c+2,40)[...,None])
            np.testing.assert_array_equal(p>0,support)
            pred=p.argmax(-1);pt=np.take_along_axis(p,y[...,None],-1)[...,0];low=c<2;middle=y-c==1
            values=dict(accuracy=(pred==y).mean(),target_probability=pt.mean(),lower_center_accuracy=(pred[low]==y[low]).mean(),middle_offset_recall=(pred[middle]==y[middle]).mean())
            for k,v in values.items():np.testing.assert_allclose(s['results'][str(update)][split][k],v,atol=1e-7)
    assert len(s['losses'])==2000 and np.isfinite(s['losses']).all()
    final=s['results']['2000']['holdout'];assert s['gate']==bool(final['accuracy']>=.99 and final['target_probability']>=.99)
    result=dict(passed=True,updates=2000,checkpoints=[560,2000],new_episodes=0,
        checks='hashes, seeded natural-sampling batches, exact prior560 state replay, optimizer steps, checkpoint probabilities/support, all metrics and gate')
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');verify(p.parse_args().output)
