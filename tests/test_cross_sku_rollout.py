import numpy as np
import pytest
from src.algos.cross_sku_rollout import selection,matrix,fit,act


def observation():
    x=np.zeros(204);x[:10]=20;x[-20:-10]=2;x[-10:]=50;x[150:160]=1.2
    return x


def test_selection_ties_and_new_namespace():
    x=observation();obs={d:x.copy() for d in [1,8,15,22]}
    chosen=selection(obs,3,np.ones(10));assert chosen[0]==(1,0) and chosen[1]!=chosen[0]
    assert chosen==selection(obs,3,np.ones(10))
    with pytest.raises(ValueError):selection({1:x},3,np.ones(10))


def test_matched_features_and_independent_normalized_ridge():
    x=observation();base=np.ones(10)
    np.testing.assert_array_equal(matrix(x,22,base,'local'),matrix(x,22,base,'cross')[:,:16])
    rng=np.random.default_rng(82)
    for arm,width in [('local',16),('cross',20)]:
        raw=np.column_stack([np.eye(10)[np.arange(256)%10],rng.normal(size=(256,width-10))])
        raw[:,12]=3;y=rng.normal(size=256);m=fit(raw,y,arm)
        assert m['mean'][:10]==[0]*10 and m['scale'][:10]==[1]*10 and m['scale'][12]==1
        z=raw.copy()
        for j in range(10,width):
            center=sum(raw[:,j])/256
            sd=np.sqrt(sum((raw[:,j]-center)**2)/256)
            z[:,j]=(raw[:,j]-center)/(sd if sd>=1e-12 else 1)
        expected=np.linalg.lstsq(np.vstack([z,np.eye(width)]),np.r_[y,np.zeros(width)],rcond=None)[0]
        np.testing.assert_allclose(m['weights'],expected,rtol=1e-9,atol=1e-10)


def test_actor_single_discount_and_exact_budget():
    x=observation();model=dict(arm='cross',mean=[0]*20,scale=[1]*20,weights=[0]*20)
    assert np.all(act(x,22,np.ones(10),model)==4)
    model['weights'][4]=2;model['weights'][5]=2
    a=act(x,22,np.ones(10),model);assert a[4]==0 and sum(a==0)==1
    assert 128*28*2+(128*4+4)*28==21616
    assert 21616+67*256*28==501872
