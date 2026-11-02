import numpy as np
import pytest
from src.algos.stratified_pricing import select_candidates,features,fit,act


def test_selection_ties_and_uniform_comparison_without_returns():
    obs={d:np.zeros(204,np.float32) for d in [1,8,15,22]}
    selected=select_candidates(obs,0)
    assert selected[0]==(1,0) and selected[1]!=selected[0]
    assert selected==select_candidates(obs,0)
    obs[22][9]=100
    assert select_candidates(obs,0)[0]==(22,9)
    with pytest.raises(ValueError):select_candidates({1:obs[1]},0)


def test_ridge_matches_augmented_least_squares():
    rng=np.random.default_rng(87);x=rng.normal(size=(256,37));y=rng.normal(size=256)
    expected=np.linalg.lstsq(np.vstack([x,np.eye(37)]),np.r_[y,np.zeros(37)],rcond=None)[0]
    np.testing.assert_allclose(fit(x,y),expected,rtol=1e-10,atol=1e-12)
    with pytest.raises(ValueError):fit(x[:-1],y[:-1])


def test_features_and_single_discount_use_current_observation():
    obs=np.zeros(204);obs[-20:-10]=2.;base=np.ones(10)
    f=features(obs,8,base);assert f.shape==(10,37)
    np.testing.assert_array_equal(f[:,:10],np.eye(10))
    w=np.zeros(37);np.testing.assert_array_equal(act(obs,8,base,w),[4]*10)
    w[0]=w[1]=1
    np.testing.assert_array_equal(act(obs,8,base,w),[3]+[4]*9)
    base[1]=2
    np.testing.assert_array_equal(act(obs,8,base,w),[4,3]+[4]*8)
    # Calendar context is not an input to the regression representation.
    changed=obs.copy();changed[160:184]=999
    np.testing.assert_array_equal(features(changed,8,base),features(obs,8,base))


def test_maximum_day_accounting():
    assert 128*28*2+256*28*(32*2+1+2)==487424
    assert 487424+(64*3+3)*28==492884
