"""Verify matched-feature experiment checkpoints and archived-data metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.algos.structured_residual import StructuredResidualActor
from src.algos.local_feature_residual import LocalFeatureResidualActor


def verify(output):
    out=Path(output);s=json.loads((out/'summary.json').read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert all(sha(v)==s['hashes'][k] for k,v in s['paths'].items())
    assert sha(out/'probabilities.npz')==s['probabilities_sha256']
    assert sha(out/'permutations.npy')==s['permutations_sha256']
    data=np.load(s['paths']['data']);probs=np.load(out/'probabilities.npz');orders=np.load(out/'permutations.npy')
    g=torch.Generator().manual_seed(11)
    np.testing.assert_array_equal(orders,torch.stack([torch.randperm(1792,generator=g) for _ in range(40)]).numpy())
    torch.set_num_threads(1)
    for label,cls in [('control',StructuredResidualActor),('augmented',LocalFeatureResidualActor)]:
        r=s['results'][label];path=out/(label+'.pt');assert sha(path)==r['checkpoint_sha256']
        ck=torch.load(path,weights_only=True);spec=ck['spec']
        assert ck['variant']==('structured_residual_actor_only_v1' if label=='control' else LocalFeatureResidualActor.variant)
        actor=cls(ck['obs_dim'],spec['n'],spec['pipe_horizon'],spec['coefficients']);actor.load_state_dict(ck['state_dict'])
        assert r['parameters']==sum(p.numel() for p in actor.parameters())
        for split in ['fit','holdout']:
            x=torch.tensor(data[split+'_x']);y=data[split+'_y'];p=probs[label+'_'+split]
            with torch.no_grad():
                actual=np.stack([d.probs.numpy() for d in actor(x)],1);c=actor.centers(x).numpy()
            np.testing.assert_array_equal(p,actual);np.testing.assert_allclose(p.sum(-1),1,atol=1e-6)
            support=(np.arange(41)>=np.maximum(c-2,0)[...,None])&(np.arange(41)<=np.minimum(c+2,40)[...,None])
            np.testing.assert_array_equal(p>0,support)
            pred=p.argmax(-1);target_p=np.take_along_axis(p,y[...,None],-1)[...,0];low=c<2;middle=y-c==1
            values=dict(accuracy=np.mean(pred==y),target_probability=target_p.mean(),
                lower_center_accuracy=np.mean(pred[low]==y[low]),middle_offset_recall=np.mean(pred[middle]==y[middle]))
            for k,v in values.items():np.testing.assert_allclose(r['metrics'][split][k],v,atol=1e-7)
        assert len(r['history'])==40
        h=r['metrics']['holdout'];assert r['gate']==bool(h['accuracy']>=.99 and h['target_probability']>=.99)
    result=dict(passed=True,arms=2,fit_states=1792,holdout_states=448,new_episodes=0,
        checks='source/checkpoint/data hashes, exact seeded permutations, checkpoint predictions, physical support, metrics and gates')
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');verify(p.parse_args().output)
