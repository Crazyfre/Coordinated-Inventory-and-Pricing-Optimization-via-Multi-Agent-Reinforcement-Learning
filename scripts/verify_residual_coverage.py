"""Independent array/count and brute-force local-neighbor audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def verify(output):
    out=Path(output);s=json.loads((out/'summary.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert all(sha(v)==s['hashes'][k] for k,v in s['paths'].items())
    assert sha(out/'audit_arrays.npz')==s['arrays_sha256']
    a=np.load(out/'audit_arrays.npz');ds=np.load(s['paths']['data']);probs=np.load(s['paths']['probabilities'])
    def check(row,y,before,after):
        assert row['count']==len(y)
        flags=[(before==y)&(after==y),(before!=y)&(after==y),(before==y)&(after!=y),(before!=y)&(after!=y)]
        for key,flag in zip(['retained_correct','corrected','regressed','persistent_error'],flags):assert row[key]==int(flag.sum())
        if 'final_error_rate' in row:np.testing.assert_allclose(row['final_error_rate'],(after!=y).mean())
    for split in ['fit','holdout']:
        y=a[split+'_target'];b=a[split+'_before'];f=a[split+'_after'];c=a[split+'_center']
        np.testing.assert_array_equal(y,ds[split+'_y']);np.testing.assert_array_equal(b,probs['560_'+split].argmax(-1));np.testing.assert_array_equal(f,probs['2000_'+split].argmax(-1))
        # Original observation has ten stock entries followed by ten seven-day pipelines.
        x=ds[split+'_x'];n=y.shape[1];h=7
        position=x[:,:n]+x[:,n:n*(h+1)].reshape(-1,n,h).sum(-1)
        expected=np.stack([x[:,-n:]/5,position/5],-1).astype(np.float64)
        np.testing.assert_array_equal(expected,a[split+'_coordinates'])
        t=s['tables'][split];check(t['overall'],y.ravel(),b.ravel(),f.ravel())
        masks=dict(lower=c<2,interior=(c>=2)&(c<=38),upper=c>38)
        masks.update({f'offset_{k}':y-c==k for k in range(-2,3)})
        for k,m in masks.items():check(t['groups'][k],y[m],b[m],f[m])
        for i,name in enumerate(s['sku_ids']):check(t['by_sku'][name],y[:,i],b[:,i],f[:,i])
    for i,name in enumerate(s['sku_ids']):
        fit=a['fit_coordinates'][:,i];hold=a['holdout_coordinates'][:,i];mean=fit.mean(0);scale=fit.std(0);scale[scale==0]=1
        np.testing.assert_allclose(mean,a[f'sku_{i}_mean']);np.testing.assert_allclose(scale,a[f'sku_{i}_scale'])
        fit=(fit-mean)/scale;hold=(hold-mean)/scale
        for split,queries in [('fit',fit),('holdout',hold)]:
            indices=[];distances=[]
            for j,query in enumerate(queries):
                d=np.sum((fit-query)**2,axis=1)
                if split=='fit':d[(j//28)*28:(j//28+1)*28]=np.inf
                index=int(d.argmin());indices.append(index);distances.append(float(np.sqrt(d[index])))
            np.testing.assert_array_equal(indices,a[f'sku_{i}_{split}_index'])
            np.testing.assert_allclose(distances,a[f'sku_{i}_{split}_distance'],atol=1e-12)
        threshold=np.quantile(a[f'sku_{i}_fit_distance'],.95);row=s['coverage'][name]
        np.testing.assert_allclose(threshold,row['threshold'])
        far=a[f'sku_{i}_holdout_distance']>threshold;np.testing.assert_array_equal(far,a['holdout_far'][:,i]);assert int(far.sum())==row['far_count']
        offset=a['fit_target'][:,i]-a['fit_center'][:,i]
        same=a['holdout_target'][:,i]-a['holdout_center'][:,i]==offset[a[f'sku_{i}_holdout_index']]
        np.testing.assert_array_equal(same,a['holdout_neighbor_same_residual'][:,i]);np.testing.assert_allclose(same.mean(),row['same_neighbor_residual_fraction'])
    far=a['holdout_far'];same=a['holdout_neighbor_same_residual']
    for name,m in dict(near=~far,far=far,neighbor_same=same,neighbor_different=~same).items():check(s['coverage_groups'][name],a['holdout_target'][m],a['holdout_before'][m],a['holdout_after'][m])
    result=dict(passed=True,decisions=22400,new_episodes=0,checks='hashes, archived labels/predictions, local coordinates, every transition table, brute-force nearest distances/indices with episode exclusion, coverage thresholds/groups')
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');verify(p.parse_args().output)
