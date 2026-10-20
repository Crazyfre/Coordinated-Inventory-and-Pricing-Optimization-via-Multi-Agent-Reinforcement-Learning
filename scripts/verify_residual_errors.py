"""Verify archived error tables from serialized physical probabilities."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def verify(output):
    out=Path(output);s=json.loads((out/'summary.json').read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert all(sha(path)==s['hashes'][key] for key,path in s['paths'].items())
    assert sha(out/'predictions.npz')==s['predictions_sha256']
    original=json.loads(Path(s['paths']['summary.json']).read_text())
    calibration=Path(original['protocol']['paths']['calibration'])
    assert sha(calibration)==original['protocol']['hashes']['calibration']
    assert json.loads(calibration.read_text())['series_order']==s['sku_ids']
    data=np.load(out/'predictions.npz');majority=None
    for split in ['fit','holdout']:
        y,pred,c,prob=[data[split+'_'+k] for k in ['target','predicted','center','probabilities']]
        np.testing.assert_allclose(prob.sum(-1),1,atol=1e-6)
        np.testing.assert_array_equal(pred,prob.argmax(-1))
        assert ((pred>=np.maximum(c-2,0))&(pred<=np.minimum(c+2,40))).all()
        pt=prob[np.arange(len(y))[:,None],np.arange(y.shape[1]),y]
        np.testing.assert_array_equal(pt,data[split+'_target_probability'])
        table=s['tables'][split];err=pred!=y;offset=y-c
        masks=dict(overall=np.ones_like(y,dtype=bool),center_low=c<2,center_interior=(c>=2)&(c<=38),center_high=c>38,
            target_zero=y==0,target_interior=(y>0)&(y<40),target_max=y==40)
        masks.update({f'offset_{k}':offset==k for k in range(-2,3)})
        for i,name in enumerate(s['sku_ids']):
            masks[name]=np.zeros_like(y,dtype=bool);masks[name][:,i]=True
        rows=dict(overall=table['overall'],**table['groups'],**table['by_sku'])
        for key,m in masks.items():
            r=rows[key];assert r['count']==int(m.sum()) and r['errors']==int(err[m].sum())
            if not m.any():
                assert all(r[k] is None for k in ['accuracy','target_probability','nll','mae_units']);continue
            values=dict(accuracy=1-err[m].mean(),target_probability=pt[m].mean(),nll=-np.log(pt[m]).mean(),
                mae_units=(5*np.abs(pred[m]-y[m])).mean(),under=(pred[m]<y[m]).sum(),over=(pred[m]>y[m]).sum())
            for k,v in values.items():np.testing.assert_allclose(r[k],v)
        confusion=[[int(((offset==a)&(pred-c==b)).sum()) for b in range(-2,3)] for a in range(-2,3)]
        assert confusion==table['confusion']
        if split=='fit':majority=np.array([np.argmax([(offset[:,i]==k).sum() for k in range(-2,3)])-2 for i in range(y.shape[1])])
        np.testing.assert_allclose(table['majority_baseline_accuracy'],(np.clip(c+majority,0,40)==y).mean())
        np.testing.assert_allclose(table['target_probability_on_errors'],pt[err].mean())
        np.testing.assert_allclose(table['fraction_errors_predicting_center'],((pred==c)&err).sum()/err.sum())
    assert majority.tolist()==s['fit_only_majority_offsets']
    result=dict(passed=True,decisions=22400,new_episodes=0,checks='hashes, simulator SKU ordering, probability normalization, support, every aggregate and confusion cell, fit-only baseline')
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');verify(p.parse_args().output)
