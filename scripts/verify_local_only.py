"""Verify local-only checkpoints and archived comparator metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.algos.local_only_residual import LocalOnlyResidualActor


def verify(output):
    out=Path(output);s=json.loads((out/'summary.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert all(sha(v)==s['hashes'][k] for k,v in s['paths'].items())
    assert all(sha(out/k)==v for k,v in s['artifact_hashes'].items())
    ds=np.load(s['paths']['data']);saved=np.load(out/'probabilities.npz');reference=np.load(s['paths']['comparator_probabilities'])
    torch.set_num_threads(1)
    ck=torch.load(out/'initial.pt',weights_only=True);spec=ck['spec']
    torch.manual_seed(11);initial=LocalOnlyResidualActor(ck['obs_dim'],spec['n'],spec['pipe_horizon'],spec['coefficients'])
    assert all(torch.equal(v,initial.state_dict()[k]) for k,v in ck['state_dict'].items())
    assert sum(p.numel() for p in initial.parameters())==s['parameters']['local_only']
    for update in [560,2000]:
        ck=torch.load(out/f'update_{update}.pt',weights_only=True)
        assert ck['variant']==LocalOnlyResidualActor.variant and ck['update']==update
        assert all(int(v['step'])==update for v in ck['optimizer']['state'].values())
        actor=LocalOnlyResidualActor(ck['obs_dim'],spec['n'],spec['pipe_horizon'],spec['coefficients']);actor.load_state_dict(ck['state_dict'])
        for split in ['fit','holdout']:
            x=torch.tensor(ds[split+'_x']);y=ds[split+'_y']
            with torch.no_grad():p=np.stack([d.probs.numpy() for d in actor(x)],1);c=actor.centers(x).numpy()
            np.testing.assert_array_equal(p,saved[f'{update}_{split}']);np.testing.assert_allclose(p.sum(-1),1,atol=1e-6)
            support=(np.arange(41)>=np.maximum(c-2,0)[...,None])&(np.arange(41)<=np.minimum(c+2,40)[...,None]);np.testing.assert_array_equal(p>0,support)
            for probs,table in [(p,s['results']),(reference[f'{update}_{split}'],s['comparator'])]:
                pred=probs.argmax(-1);pt=np.take_along_axis(probs,y[...,None],-1)[...,0];low=c<2;middle=y-c==1
                values=dict(accuracy=(pred==y).mean(),target_probability=pt.mean(),lower_center_accuracy=(pred[low]==y[low]).mean(),middle_offset_recall=(pred[middle]==y[middle]).mean())
                for k,v in values.items():np.testing.assert_allclose(table[str(update)][split][k],v,atol=1e-7)
    assert len(s['losses'])==2000 and np.isfinite(s['losses']).all()
    h=s['results']['2000']['holdout'];assert s['gate']==bool(h['accuracy']>=.99 and h['target_probability']>=.99)
    result=dict(passed=True,updates=2000,new_episodes=0,checks='source/data/comparator hashes, seeded initial state, checkpoints/optimizer steps, exact probabilities/support, both-model metrics and gate')
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');verify(p.parse_args().output)
