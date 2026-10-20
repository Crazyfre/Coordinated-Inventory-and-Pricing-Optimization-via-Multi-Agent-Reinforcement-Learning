"""Training-data-only error decomposition of the frozen residual imitation actor."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.diagnose_cover_search import CoverPolicy
from src.algos.structured_residual import StructuredResidualActor
from src.config.sku_config import load_skus


def statistics(target, predicted, probability):
    target,predicted,probability=map(np.asarray,(target,predicted,probability))
    if target.size==0:
        return dict(count=0,errors=0,accuracy=None,target_probability=None,nll=None,mae_units=None,under=0,over=0)
    assert target.shape==predicted.shape==probability.shape
    assert np.isfinite(probability).all() and ((probability>0)&(probability<=1)).all()
    return dict(count=int(target.size),errors=int((target!=predicted).sum()),
        accuracy=float((target==predicted).mean()),target_probability=float(probability.mean()),
        nll=float(-np.log(probability).mean()),mae_units=float((5*np.abs(target-predicted)).mean()),
        under=int((predicted<target).sum()),over=int((predicted>target).sum()))


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    source=Path('outputs/residual_imitation/seed11_v1')
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    paths={name:str(source/name) for name in ['summary.json','datasets.npz','fitted.pt']}
    paths.update(actor='src/algos/structured_residual.py',runner=__file__,protocol='RESIDUAL_ERROR_PROTOCOL.md',skus='src/config/skus.json')
    hashes={k:digest(v) for k,v in paths.items()}
    original=json.loads((source/'summary.json').read_text())
    assert hashes['fitted.pt']==original['checkpoint_sha256']
    assert hashes['datasets.npz']==original['dataset_sha256']
    torch.set_num_threads(1)
    ck=torch.load(source/'fitted.pt',map_location='cpu',weights_only=True)
    spec=ck['spec'];n,h=spec['n'],spec['pipe_horizon']
    ds=np.load(source/'datasets.npz')
    actor=StructuredResidualActor(ds['fit_x'].shape[1],n,h,spec['coefficients'])
    actor.load_state_dict(ck['replenishment']);actor.eval()
    teacher=CoverPolicy(n,h,original['protocol']['cover'])
    sku_ids=[s.series_id for s in load_skus()];assert len(sku_ids)==n
    tables={};saved={};majority=None
    for split in ['fit','holdout']:
        x=torch.tensor(ds[split+'_x']);y=ds[split+'_y']
        with torch.no_grad():
            center=actor.centers(x).numpy()
            distributions=actor(x)
            probs=np.stack([d.probs.numpy() for d in distributions],axis=1)
        full=np.asarray([teacher.act(row)[0] for row in x.numpy()])
        expected=np.clip(full,np.maximum(center-2,0),np.minimum(center+2,40))
        np.testing.assert_array_equal(y,expected)
        pred=probs.argmax(-1);p=np.take_along_axis(probs,y[...,None],axis=-1)[...,0]
        residual=y-center;pred_residual=pred-center
        assert ((residual>=-2)&(residual<=2)).all()
        if split=='fit':
            majority=np.asarray([np.bincount(residual[:,i]+2,minlength=5).argmax()-2 for i in range(n)])
        baseline=np.clip(center+majority,0,40)
        groups={}
        masks=dict(center_low=center<2,center_interior=(center>=2)&(center<=38),center_high=center>38,
            target_zero=y==0,target_interior=(y>0)&(y<40),target_max=y==40)
        masks.update({f'offset_{k}':residual==k for k in range(-2,3)})
        for name,mask in masks.items():groups[name]=statistics(y[mask],pred[mask],p[mask])
        by_sku={name:statistics(y[:,i],pred[:,i],p[:,i]) for i,name in enumerate(sku_ids)}
        confusion=np.zeros((5,5),dtype=int)
        np.add.at(confusion,((residual+2).ravel(),(pred_residual+2).ravel()),1)
        overall=statistics(y,pred,p)
        np.testing.assert_allclose(overall['accuracy'],original['history'][-1][split],atol=1e-7)
        tables[split]=dict(overall=overall,groups=groups,by_sku=by_sku,
            confusion_rows_target_columns_predicted_offsets=list(range(-2,3)),confusion=confusion.tolist(),
            majority_baseline_accuracy=float((baseline==y).mean()),
            target_probability_on_errors=float(p[pred!=y].mean()),
            fraction_errors_predicting_center=float(((pred==center)&(pred!=y)).sum()/(pred!=y).sum()))
        saved.update({split+'_'+k:v for k,v in dict(target=y,predicted=pred,target_probability=p,center=center,probabilities=probs).items()})
        # Reconcile all disjoint partitions and the confusion matrix to totals.
        for keys in [['center_low','center_interior','center_high'],['target_zero','target_interior','target_max'],[f'offset_{k}' for k in range(-2,3)]]:
            assert sum(groups[k]['count'] for k in keys)==overall['count']
            assert sum(groups[k]['errors'] for k in keys)==overall['errors']
        assert confusion.sum()==overall['count'] and np.trace(confusion)==overall['count']-overall['errors']
    np.savez_compressed(out/'predictions.npz',**saved)
    summary=dict(paths=paths,hashes=hashes,tables=tables,sku_ids=sku_ids,
        fit_only_majority_offsets=majority.tolist(),new_episodes=0,training_updates=0,
        validation_arrays_used=False,held_out_test_used=False,
        predictions_sha256=digest(out/'predictions.npz'),checks='source hashes, projected labels, archived accuracy, partition/count/confusion reconciliation')
    assert all(digest(v)==hashes[k] for k,v in paths.items())
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    lines=['# Archived residual error tables','', 'Only saved training and training-scenario holdout states; no new episodes or updates.','']
    for split,table in tables.items():
        lines+=['## '+split,'','| Group | Decisions | Errors | Accuracy | Mean target probability | Mean absolute error (units) |','|---|---:|---:|---:|---:|---:|']
        for name,row in dict(overall=table['overall'],**table['groups'],**table['by_sku']).items():
            fmt=lambda v:'n/a' if v is None else f'{v:.4f}'
            lines.append(f"| {name} | {row['count']} | {row['errors']} | {fmt(row['accuracy'])} | {fmt(row['target_probability'])} | {fmt(row['mae_units'])} |")
        lines+=['',f"Fit-only per-SKU majority-offset baseline accuracy: {table['majority_baseline_accuracy']:.6f}.",'']
    (out/'TABLES.md').write_text('\n'.join(lines))
    print(json.dumps(tables,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
