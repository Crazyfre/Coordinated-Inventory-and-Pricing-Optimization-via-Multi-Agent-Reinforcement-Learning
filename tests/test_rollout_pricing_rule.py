import copy
import numpy as np
import pytest
from src.algos.rollout_pricing_rule import THRESHOLDS,act,fit


def data():
    rows=[]
    for ep in range(16):
        for day in [1,8,15,22]:
            obs=np.zeros(180,np.float32);obs[-20:-10]=1;obs[:10]=ep
            gain=np.full((16,10),1. if ep>10 else -3.)
            rows.append(dict(episode=ep,day=day,obs=obs.tolist(),advantages=gain.tolist()))
    return rows


def test_score_selection_independently_reconstructed():
    rows=data();result=fit(rows)
    for sku in range(10):
        scores=[]
        for t in THRESHOLDS:
            total=0.
            for r in rows:
                if t is not None and r['obs'][sku]>t*7*r['obs'][160+sku]:
                    total+=sum(v[sku] for v in r['advantages'])/16
            scores.append(total/64)
        assert result['scores'][sku]==scores
        assert result['thresholds'][sku]==THRESHOLDS[max(range(8),key=lambda i:scores[i])]
    assert result['thresholds']==[1.5]*10


def test_ties_empty_regions_and_observable_only_actions():
    rows=data()
    for r in rows:r['advantages']=np.zeros((16,10)).tolist()
    assert fit(rows)['thresholds']==[None]*10
    obs=np.zeros(180);obs[:10]=12;obs[-20:-10]=1
    np.testing.assert_array_equal(act(obs,[1.5]*10),[3]*10)
    obs[10:160]=999;obs[-10:]=999
    np.testing.assert_array_equal(act(obs,[1.5]*10),[3]*10)
    obs[:10]=10.5
    np.testing.assert_array_equal(act(obs,[1.5]*10),[4]*10)
    expanded=np.r_[obs[:160],np.zeros(24),obs[160:]]
    np.testing.assert_array_equal(act(expanded,[1.5]*10),[4]*10)


def test_missing_states_or_futures_rejected():
    rows=data()
    with pytest.raises(ValueError):fit(rows[:-1])
    rows[0]['advantages'].pop()
    with pytest.raises(ValueError):fit(rows)
