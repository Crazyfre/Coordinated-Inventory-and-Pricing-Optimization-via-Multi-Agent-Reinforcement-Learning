"""Frozen local-only boundary errors and confidence decomposition."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.algos.local_only_residual import LocalOnlyResidualActor


def describe(y,pred,pt,confidence):
    correct=pred==y;count=y.size
    return dict(count=int(count),errors=int((~correct).sum()),accuracy=float(correct.mean()) if count else None,
        mean_target_probability=float(pt.mean()) if count else None,expected_sampled_errors=float((1-pt).sum()),
        correct_below_99=int((correct&(pt<.99)).sum()),high_confidence_errors=int((~correct&(confidence>=.9)).sum()),
        mean_absolute_units=float((5*np.abs(pred-y)).mean()) if count else None,
        under=int((pred<y).sum()),over=int((pred>y).sum()))


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);root=Path('outputs/local_only/seed11_v1')
    original=json.loads((root/'summary.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    paths=dict(original=str(root/'summary.json'),data=original['paths']['data'],probabilities=str(root/'probabilities.npz'),
        checkpoint=str(root/'update_2000.pt'),calibration='outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json',
        actor='src/algos/local_only_residual.py',runner=__file__,protocol='LOCAL_ONLY_INSPECTION_PROTOCOL.md')
    hashes={k:sha(v) for k,v in paths.items()}
    assert hashes['data']==original['hashes']['data']
    assert hashes['probabilities']==original['artifact_hashes']['probabilities.npz'] and hashes['checkpoint']==original['artifact_hashes']['update_2000.pt']
    ds=np.load(paths['data']);probs=np.load(paths['probabilities']);ck=torch.load(paths['checkpoint'],weights_only=True);spec=ck['spec']
    ids=json.loads(Path(paths['calibration']).read_text())['series_order'];torch.set_num_threads(1)
    actor=LocalOnlyResidualActor(ck['obs_dim'],spec['n'],spec['pipe_horizon'],spec['coefficients']);actor.load_state_dict(ck['state_dict'])
    tables={};arrays={}
    for split in ['fit','holdout']:
        x=torch.tensor(ds[split+'_x']);y=ds[split+'_y'];p=probs['2000_'+split]
        with torch.no_grad():actual=np.stack([d.probs.numpy() for d in actor(x)],1);center=actor.centers(x).numpy()
        np.testing.assert_array_equal(p,actual)
        pred=p.argmax(-1);pt=np.take_along_axis(p,y[...,None],-1)[...,0];confidence=p.max(-1)
        top=np.sort(p,axis=-1)[...,-2:];margin=top[...,1]-top[...,0];correct=pred==y
        masks=dict(lower=center<2,interior=(center>=2)&(center<=38),upper=center>38,correct=correct,wrong=~correct,
            margin_low=margin<.1,margin_medium=(margin>=.1)&(margin<.5),margin_high=margin>=.5)
        masks.update({f'offset_{k}':y-center==k for k in range(-2,3)})
        groups={k:describe(y[m],pred[m],pt[m],confidence[m]) for k,m in masks.items()}
        by_sku={name:describe(y[:,i],pred[:,i],pt[:,i],confidence[:,i]) for i,name in enumerate(ids)}
        confusion=np.zeros((5,5),dtype=int);np.add.at(confusion,((y-center+2).ravel(),(pred-center+2).ravel()),1)
        overall=describe(y,pred,pt,confidence)
        tables[split]=dict(overall=overall,groups=groups,by_sku=by_sku,offset_order=list(range(-2,3)),confusion=confusion.tolist(),
            temperature_probability_upper_bound=float((1+correct.mean())/2))
        for k,v in dict(target=y,predicted=pred,target_probability=pt,confidence=confidence,center=center,margin=margin).items():arrays[split+'_'+k]=v
        for keys in [['lower','interior','upper'],['correct','wrong'],['margin_low','margin_medium','margin_high'],[f'offset_{k}' for k in range(-2,3)]]:
            assert sum(groups[k]['count'] for k in keys)==overall['count']
            assert sum(groups[k]['errors'] for k in keys)==overall['errors']
            np.testing.assert_allclose(sum(groups[k]['expected_sampled_errors'] for k in keys),overall['expected_sampled_errors'],rtol=1e-6)
    np.savez_compressed(out/'decisions.npz',**arrays)
    assert all(sha(v)==hashes[k] for k,v in paths.items())
    summary=dict(paths=paths,hashes=hashes,tables=tables,sku_ids=ids,arrays_sha256=sha(out/'decisions.npz'),new_episodes=0,training_updates=0,validation_used=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    lines=['# Local-only errors and confidence','', '| Group | Decisions | Errors | Mean target probability | Expected sampled errors | Correct below99% |','|---|---:|---:|---:|---:|---:|']
    for split,t in tables.items():
        for name,r in dict(overall=t['overall'],**t['groups'],**t['by_sku']).items():
            prob='n/a' if r['mean_target_probability'] is None else f"{r['mean_target_probability']:.6f}"
            lines.append(f"| {split}/{name} | {r['count']} | {r['errors']} | {prob} | {r['expected_sampled_errors']:.3f} | {r['correct_below_99']} |")
    (out/'TABLES.md').write_text('\n'.join(lines));print(json.dumps({k:dict(overall=v['overall'],groups=v['groups'],bound=v['temperature_probability_upper_bound']) for k,v in tables.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
