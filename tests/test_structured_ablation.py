import numpy as np
import pytest
from src.algos.structured_ablation import StructuredControl,pair_keys
from src.algos.parameter_policy_gradient import ParameterPolicyGradient,order_action


def test_main_update_matches_verified_estimator_and_both_starts():
    a=StructuredControl('main',266);b=ParameterPolicyGradient('sku',266)
    for _ in range(4):
        e=a.rng.normal(size=(8,10));np.testing.assert_array_equal(e,b.rng.normal(size=(8,10)))
        a.update(e,np.arange(8));b.update(e,np.arange(8))
        np.testing.assert_array_equal(a.mean,b.mean)
    x=np.random.default_rng(4).uniform(0,30,260).astype(np.float32)
    for condition in ['main','pooled','independent']:
        p=StructuredControl(condition,1)
        np.testing.assert_array_equal(order_action(x,1,p.expand(p.mean)),order_action(x,1,np.full(10,7.5)))


def test_scalar_gradient_and_key_separation():
    a=StructuredControl('pooled',1)
    gradient=a.update(np.array([[1.],[-1.]]),np.array([1.5,-1.5]))
    np.testing.assert_allclose(gradient,[1.])
    assert len(np.unique(a.expand(a.mean)))==1
    keys=set()
    for condition in ['main','pooled','independent']:
        for j in range(10):
            for t in range(512):
                plus,minus=pair_keys(condition,j,t)
                assert (plus!=minus)==(condition=='independent')
                current={plus,minus}
                assert keys.isdisjoint(current);keys.update(current)
    with pytest.raises(ValueError):pair_keys('other',1,1)
