"""Recompute confidence tables independently from archived probabilities."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def verify(output):
    out=Path(output);s=json.loads((out/'summary.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert all(sha(v)==s['hashes'][k] for k,v in s['paths'].items())
    assert sha(out/'decisions.npz')==s['arrays_sha256']
    a=np.load(out/'decisions.npz');ds=np.load(s['paths']['data']);archive=np.load(s['paths']['probabilities'])
    assert json.loads(Path(s['paths']['calibration']).read_text())['series_order']==s['sku_ids']
    for split in ['fit','holdout']:
        y=ds[split+'_y'];p=archive['2000_'+split];pred=p.argmax(-1);pt=np.take_along_axis(p,y[...,None],-1)[...,0]
        confidence=p.max(-1);top=np.sort(p,axis=-1)[...,-2:];margin=top[...,1]-top[...,0];center=a[split+'_center']
        for k,v in dict(target=y,predicted=pred,target_probability=pt,confidence=confidence,margin=margin).items():np.testing.assert_array_equal(v,a[split+'_'+k])
        masks=dict(overall=np.ones_like(y,dtype=bool),lower=center<2,interior=(center>=2)&(center<=38),upper=center>38,
            correct=pred==y,wrong=pred!=y,margin_low=margin<.1,margin_medium=(margin>=.1)&(margin<.5),margin_high=margin>=.5)
        masks.update({f'offset_{k}':y-center==k for k in range(-2,3)})
        table=s['tables'][split];rows=dict(overall=table['overall'],**table['groups'],**table['by_sku'])
        for i,name in enumerate(s['sku_ids']):masks[name]=np.broadcast_to(np.arange(y.shape[1])==i,y.shape)
        for key,m in masks.items():
            r=rows[key];count=int(m.sum());correct=pred[m]==y[m]
            assert r['count']==count and r['errors']==int((~correct).sum())
            values=dict(expected_sampled_errors=(1-pt[m]).sum(),correct_below_99=(correct&(pt[m]<.99)).sum(),
                high_confidence_errors=((~correct)&(confidence[m]>=.9)).sum(),under=(pred[m]<y[m]).sum(),over=(pred[m]>y[m]).sum())
            if count:values.update(accuracy=correct.mean(),mean_target_probability=pt[m].mean(),mean_absolute_units=(5*np.abs(pred[m]-y[m])).mean())
            else:assert all(r[k] is None for k in ['accuracy','mean_target_probability','mean_absolute_units'])
            for k,v in values.items():np.testing.assert_allclose(r[k],v)
        matrix=[[int(((y-center==i)&(pred-center==j)).sum()) for j in range(-2,3)] for i in range(-2,3)]
        assert table['confusion']==matrix
        np.testing.assert_allclose(table['temperature_probability_upper_bound'],1-(pred!=y).mean()/2)
    result=dict(passed=True,decisions=22400,new_episodes=0,checks='hashes, SKU order, archived physical predictions/probabilities, all group metrics, confusion counts and temperature bound')
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');verify(p.parse_args().output)
