"""Independently check every archived split and held-out prediction."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.train_compact_residual import digest,write_json


def check_node(node,x,y,depth=0):
    assert node['depth']==depth and node['count']==len(y)
    assert node['mean']==float(np.mean(y))
    best=float(np.square(y-np.mean(y)).sum());choice=None
    if depth<4 and len(y)>=16:
        for col in range(x.shape[1]):
            values=sorted(set(x[:,col].tolist()))
            for low,high in zip(values,values[1:]):
                threshold=low+(high-low)/2
                mask=x[:,col]<=threshold
                if np.count_nonzero(mask)<8 or np.count_nonzero(~mask)<8:continue
                a,b=y[mask],y[~mask]
                loss=float(np.square(a-np.mean(a)).sum()+np.square(b-np.mean(b)).sum())
                if loss<best:best=loss;choice=(col,threshold,mask)
    if choice is None:
        assert set(node)=={'depth','count','mean'}
        return depth
    col,threshold,mask=choice
    assert node['feature']==col and node['threshold']==threshold
    return max(check_node(node['left'],x[mask],y[mask],depth+1),
               check_node(node['right'],x[~mask],y[~mask],depth+1))


def verify(source,output):
    root,out=Path(source),Path(output)
    manifest=json.loads((out/'manifest.json').read_text())
    for path,h in manifest['hashes'].items():assert digest(path)==h
    p=json.loads((root/'policy.json').read_text());summary=json.loads((out/'summary.json').read_text())
    assert summary['status']=='complete' and summary['seconds']<=120
    assert digest(out/'folds.json')==summary['folds_hash']
    folds=json.loads((out/'folds.json').read_text());assert len(folds)==128
    x=np.array(p['matrix'])[:,:16];y=np.array(p['targets']);pred=[];depths=[]
    for ep,fold in enumerate(folds):
        train=[j for j in range(256) if j//2!=ep];test=[2*ep,2*ep+1]
        assert fold['episode']==ep and fold['train_rows']==train and fold['test_rows']==test
        assert hashlib.sha256(json.dumps(fold['tree'],sort_keys=True).encode()).hexdigest()==fold['tree_hash']
        depths.append(check_node(fold['tree'],x[train],y[train]))
        for j in test:
            node=fold['tree']
            while 'feature' in node:
                node=node['right'] if x[j,node['feature']]>node['threshold'] else node['left']
            pred.append(node['mean'])
    np.testing.assert_array_equal(pred,summary['predictions']);pred=np.array(pred)
    chosen=np.flatnonzero(pred>0);count=len(chosen);windows=len(set(chosen//2))
    half=np.array([[np.mean(r['gains'][:16])/p['base_prices'][r['sku']],
                    np.mean(r['gains'][16:])/p['base_prices'][r['sku']]] for r in p['samples']])
    m=summary['metrics'];assert m['positive_predictions']==count and m['positive_windows']==windows
    assert m['positive_targets']==int(sum(y[chosen]>0))
    assert m['mse']==float(np.square(pred-y).mean())
    assert m['mean_target']==(float(y[chosen].mean()) if count else None)
    expected=half[chosen].mean(0).tolist() if count else [None,None]
    assert m['half_means']==expected
    assert m['continuation_flag']==bool(count>=8 and windows>=8 and y[chosen].mean()>0 and min(expected)>0)
    assert summary['fits']==128 and summary['new_simulator_days']==0
    write_json(out/'verification.json',dict(status='passed',folds=128,maximum_depth=max(depths),
        diagnostic_fit_reconstructions=128,new_simulator_days=0,metrics=m,
        verifier_sha256=digest(__file__)))
    print('Nonlinear diagnostic independently verified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();verify(a.source,a.output)
