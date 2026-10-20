import numpy as np
from src.algos.rollout_correction import RolloutCorrection,features,fit_ridge


class Center:
    def predict(self,obs):return np.array([2,4])


class Target:
    def act(self,obs):return np.array([9,9]),0


def observation():
    # stock, one pipeline bucket per SKU, price, lead, headroom, forecast
    return np.array([5,10,15,20,1,1,3,3,400,400,5,10],dtype=float)


def test_zero_reward_correction_preserves_exact_projected_rule():
    model=RolloutCorrection(Center(),Target(),2,1)
    for day in [1,8,28]:
        np.testing.assert_array_equal(model.action(observation(),day),[4,6])
        np.testing.assert_array_equal(features(observation(),2,1,0,0,day),np.zeros(12))


def test_correction_changes_only_one_sku_and_keeps_support():
    w=np.zeros(12);w[0]=-10
    model=RolloutCorrection(Center(),Target(),2,1,w)
    action=model.action(observation(),1)
    np.testing.assert_array_equal(action,[0,6])


def test_features_use_pipeline_headroom_and_remaining_horizon():
    x=features(observation(),2,1,0,-2,28)
    np.testing.assert_allclose(x[:6],-2*np.array([1,np.log(2),np.log(5),.8,1/28,1/3]))
    np.testing.assert_allclose(x[6:],-2*x[:6])


def test_ridge_fits_reward_direction_and_zero_targets_preserve_start():
    rng=np.random.default_rng(11);x=rng.normal(size=(100,12));y=3*x[:,0]
    w,s=fit_ridge(x,y)
    assert np.corrcoef(x/s@w,y)[0,1]>.99
    w0,_=fit_ridge(x,np.zeros(100))
    np.testing.assert_array_equal(w0,np.zeros(12))
