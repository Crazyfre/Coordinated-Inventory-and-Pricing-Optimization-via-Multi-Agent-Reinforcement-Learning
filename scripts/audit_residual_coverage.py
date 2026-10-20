"""Archived fitting-error transitions and per-SKU local-state coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist
from src.algos.local_feature_residual import LocalFeatureResidualActor


def transition_counts(y,before,after):
    a=before==y;b=after==y
    return dict(count=int(y.size),retained_correct=int((a&b).sum()),corrected=int((~a&b).sum()),
        regressed=int((a&~b).sum()),persistent_error=int((~a&~b).sum()))


def local_coordinates(x,n,h):
    position=x[:,:n]+x[:,n:n*(1+h)].reshape(-1,n,h).sum(-1)
    return np.stack([x[:,-n:]/5,position/5],-1).astype(np.float64)


def neighbors(fit,holdout,episode_days=28):
    mean=fit.mean(0);scale=fit.std(0);scale=np.where(scale==0,1.,scale)
    a=(fit-mean)/scale;b=(holdout-mean)/scale
    distances=cdist(a,a)
    same_episode=np.arange(len(a))[:,None]//episode_days==np.arange(len(a))[None,:]//episode_days
    distances[same_episode]=np.inf
    fit_index=distances.argmin(1);fit_distance=distances[np.arange(len(a)),fit_index]
    distances=cdist(b,a);index=distances.argmin(1);distance=distances[np.arange(len(b)),index]
    return dict(mean=mean,scale=scale,fit_index=fit_index,fit_distance=fit_distance,
        holdout_index=index,holdout_distance=distance,threshold=float(np.quantile(fit_distance,.95)))


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    source=Path('outputs/residual_budget/seed11_v1');old=json.loads((source/'summary.json').read_text())
    paths=dict(summary=str(source/'summary.json'),data=old['paths']['data'],probabilities=str(source/'probabilities.npz'),
        checkpoint=str(source/'update_2000.pt'),calibration='outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json',
        protocol='RESIDUAL_COVERAGE_PROTOCOL.md',runner=__file__)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();hashes={k:sha(v) for k,v in paths.items()}
    assert hashes['data']==old['hashes']['data'] and hashes['probabilities']==old['artifact_hashes']['probabilities.npz']
    assert hashes['checkpoint']==old['artifact_hashes']['update_2000.pt']
    ds=np.load(paths['data']);p=np.load(paths['probabilities']);ck=torch.load(paths['checkpoint'],weights_only=True);spec=ck['spec']
    torch.set_num_threads(1);actor=LocalFeatureResidualActor(ck['obs_dim'],spec['n'],spec['pipe_horizon'],spec['coefficients']);actor.load_state_dict(ck['state_dict'])
    ids=json.loads(Path(paths['calibration']).read_text())['series_order']
    arrays={};tables={};coverage={};coords={}
    for split in ['fit','holdout']:
        x=ds[split+'_x'];y=ds[split+'_y'];before=p['560_'+split].argmax(-1);after=p['2000_'+split].argmax(-1)
        with torch.no_grad():c=actor.centers(torch.tensor(x)).numpy()
        coords[split]=local_coordinates(x,spec['n'],spec['pipe_horizon'])
        masks=dict(lower=c<2,interior=(c>=2)&(c<=38),upper=c>38)
        masks.update({f'offset_{k}':y-c==k for k in range(-2,3)})
        tables[split]=dict(overall=transition_counts(y,before,after),
            groups={k:transition_counts(y[m],before[m],after[m]) for k,m in masks.items()},
            by_sku={name:transition_counts(y[:,i],before[:,i],after[:,i]) for i,name in enumerate(ids)})
        for k,v in dict(target=y,before=before,after=after,center=c,coordinates=coords[split]).items():arrays[split+'_'+k]=v
    far=np.zeros_like(ds['holdout_y'],dtype=bool);matching=far.copy()
    for i,name in enumerate(ids):
        row=neighbors(coords['fit'][:,i],coords['holdout'][:,i]);far[:,i]=row['holdout_distance']>row['threshold']
        fit_offset=arrays['fit_target'][:,i]-arrays['fit_center'][:,i]
        matching[:,i]=(arrays['holdout_target'][:,i]-arrays['holdout_center'][:,i])==fit_offset[row['holdout_index']]
        coverage[name]=dict(threshold=row['threshold'],mean=row['mean'].tolist(),scale=row['scale'].tolist(),
            far_count=int(far[:,i].sum()),same_neighbor_residual_fraction=float(matching[:,i].mean()),
            fit_range=np.stack([coords['fit'][:,i].min(0),coords['fit'][:,i].max(0)]).tolist(),
            holdout_range=np.stack([coords['holdout'][:,i].min(0),coords['holdout'][:,i].max(0)]).tolist())
        for k,v in row.items():arrays[f'sku_{i}_{k}']=v
    arrays['holdout_far']=far;arrays['holdout_neighbor_same_residual']=matching
    coverage_groups={}
    for name,mask in dict(near=~far,far=far,neighbor_same=matching,neighbor_different=~matching).items():
        row=transition_counts(arrays['holdout_target'][mask],arrays['holdout_before'][mask],arrays['holdout_after'][mask])
        row['final_error_rate']=(row['regressed']+row['persistent_error'])/row['count'] if row['count'] else None
        coverage_groups[name]=row
    np.savez_compressed(out/'audit_arrays.npz',**arrays)
    summary=dict(paths=paths,hashes=hashes,sku_ids=ids,tables=tables,coverage=coverage,coverage_groups=coverage_groups,
        arrays_sha256=sha(out/'audit_arrays.npz'),new_episodes=0,training_updates=0,validation_used=False)
    assert all(sha(v)==hashes[k] for k,v in paths.items())
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    lines=['# Error transitions and local coverage','', '| Split/group | Decisions | Retained correct | Corrected | Regressed | Persistent error |','|---|---:|---:|---:|---:|---:|']
    for split,t in tables.items():
        for name,r in dict(overall=t['overall'],**t['groups'],**t['by_sku']).items():
            lines.append(f"| {split}/{name} | {r['count']} | {r['retained_correct']} | {r['corrected']} | {r['regressed']} | {r['persistent_error']} |")
    for name,r in coverage_groups.items():lines.append(f"| holdout/{name} | {r['count']} | {r['retained_correct']} | {r['corrected']} | {r['regressed']} | {r['persistent_error']} |")
    (out/'TABLES.md').write_text('\n'.join(lines))
    print(json.dumps(dict(overall={k:v['overall'] for k,v in tables.items()},coverage_groups=coverage_groups),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
