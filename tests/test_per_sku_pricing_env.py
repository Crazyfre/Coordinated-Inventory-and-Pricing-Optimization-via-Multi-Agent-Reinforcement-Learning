"""Per-SKU pricing contracts without calibration fitting or training."""
import numpy as np
import pytest
from src.env.joint_env import JointRetailEnv
from src.env.pricing_env import PricingEnv
from src.env.interface import JointAction
from src.env.forecast import DemandForecast

class PriceSource:
    def __init__(self): self.price=np.ones(10); self.seen=[]
    def set_current_price(self, prices): self.price=np.array(prices,copy=True)
    def reset(self, rng): self.seen=[]
    def __call__(self, day):
        if day>0: self.seen.append(self.price.copy())
        return np.zeros(10)

class PriceForecast(DemandForecast):
    def __init__(self, source): self.source=source
    def forecast(self, day): return 100/self.source.price
    def update(self, day, demand): pass
    def reset(self): pass
    def set_current_price(self, price): pass

def joint(scope='per_sku', days=8):
    source=PriceSource()
    env=JointRetailEnv(pricing_scope=scope, episode_length_days=days,
        demand_source=source,forecaster=PriceForecast(source),base_price=np.arange(1,11),
        initial_inventory=0)
    return env,source

def test_joint_vector_staged_forecast_weekly_hold_and_cost():
    env,source=joint(); env.reset(seed=1)
    indices=np.arange(10)%5; expected=env.base_price*np.asarray(env.config.pricing.multipliers)[indices]
    env.select_price(indices)
    np.testing.assert_allclose(env.replenishment_observation()[-10:],100/expected)
    np.testing.assert_array_equal(source.price,env.base_price)
    total_cost=0
    for day in range(1,8):
        _,_,_,info=env.step(JointAction(None,np.zeros(10,dtype=int)))
        np.testing.assert_allclose(info['price'],expected)
        total_cost+=info['reward_terms']['price_change_cost']
        if day==1: env.select_price(np.array([999])) # ignored outside cadence
    assert total_cost==pytest.approx(env.config.reward.price_change_cost*np.abs(expected-env.base_price).sum())
    env.select_price(4)
    env.step(JointAction(None,np.zeros(10,dtype=int)))
    np.testing.assert_allclose(source.seen[-1],env.base_price*1.2)

@pytest.mark.parametrize('action',[np.zeros(9,dtype=int),np.zeros((10,1),dtype=int),np.zeros(10),np.full(10,5),np.full(10,-1),True])
def test_joint_rejects_invalid_due_vector(action):
    env,_=joint();env.reset(seed=0)
    with pytest.raises(ValueError): env.select_price(action)

def test_shared_and_uniform_vector_replay_identical():
    shared,_=joint('shared');vector,_=joint()
    assert shared.pricing_nvec.tolist()==[5]
    assert vector.pricing_nvec.tolist()==[5]*10
    shared.reset(seed=3);vector.reset(seed=3)
    for day in range(8):
        if shared.pricing_due:
            shared.select_price(4);vector.select_price(np.full(10,4,dtype=int))
        a=shared.step(JointAction(None,np.ones(10,dtype=int)))
        b=vector.step(JointAction(None,np.ones(10,dtype=int)))
        assert a[1:3]==b[1:3]
        for key in ('demand','sales','orders_delivered','orders_clipped','price'):
            np.testing.assert_array_equal(a[3][key],b[3][key])

def test_p2_vector_prices_hold_and_sum_adjustment_cost():
    source=PriceSource();env=PricingEnv(pricing_scope='per_sku',episode_length_days=8,
        demand_source=source,forecaster=PriceForecast(source),base_price=np.arange(1,11),
        replenishment_hook=lambda d,v:np.zeros(10),initial_inventory=0)
    env.reset(seed=4);indices=np.arange(10)%5
    assert env.pricing_nvec.tolist()==[5]*10
    assert env.action_space.contains(indices)
    expected=env.base_price*np.array(env.config.pricing.multipliers)[indices]
    _,reward,_,done,info=env.step(indices)
    assert not done
    for row in info['days']: np.testing.assert_allclose(row['price'],expected)
    assert -reward==pytest.approx(env.config.reward.price_change_cost*np.abs(expected-env.base_price).sum())
    _,reward,_,done,info=env.step(indices)
    assert done and reward==0

@pytest.mark.parametrize('cls',[JointRetailEnv,PricingEnv])
def test_unknown_scope_fails(cls):
    with pytest.raises(ValueError,match='pricing_scope'): cls(pricing_scope='unknown')

def test_shared_scope_rejects_vector_actions_and_preserves_discrete_space():
    env,_=joint('shared');env.reset(seed=1)
    assert env.pricing_action_space.n==5
    with pytest.raises(ValueError): env.select_price(np.full(10,4,dtype=int))
    legacy=PricingEnv()
    assert legacy.pricing_scope=='shared' and legacy.action_space.n==5

@pytest.mark.parametrize('action',[np.zeros(9,dtype=int),np.zeros(10),np.full(10,5)])
def test_p2_rejects_invalid_vector(action):
    env=PricingEnv(pricing_scope='per_sku');env.reset(seed=0)
    with pytest.raises(ValueError): env.step(action)

def test_factories_expose_per_sku_and_evaluator_accepts_vectors():
    from src.experiments.phase4 import make_env,evaluate
    from src.experiments.research import make_research_env
    env=make_env(days=2,pricing_scope='per_sku')
    assert env.pricing_nvec.tolist()==[5]*10
    result=evaluate(env,lambda e,o:np.zeros(e.n,dtype=int),episodes=1,
        price_policy=lambda o:np.arange(10)%5)
    assert np.isfinite(result['mean_daily_reward'])
    research=make_research_env(days=2,pricing_scope='per_sku')
    assert research.pricing_scope=='per_sku'
    research.reset(seed=2);research.select_price(np.arange(10)%5)
    expected=research.sim.mu_for(1,research._pending_price)
    np.testing.assert_allclose(research.replenishment_observation()[-10:],expected,rtol=1e-6)
