"""Same action freedom for the per-SKU myopic baseline; shared compatibility."""
from types import SimpleNamespace
import numpy as np
from src.baselines.retail import MyopicPricingPolicy
from src.config.locked_config import get_locked_config


def fixture(scope):
    prices=np.ones(10)
    inventory=np.array([0.]+[100.]*9)
    sim=SimpleNamespace(mu_for=lambda day,p:10*np.power(p,-3.),current_price=prices.copy())
    env=SimpleNamespace(base_price=prices,day=1,pricing_scope=scope,config=get_locked_config(),sim=sim,
                        observation_components=lambda:dict(inventory=inventory,price=prices))
    return env


def test_myopic_per_sku_can_choose_different_prices_without_mutation():
    env=fixture('per_sku')
    action=MyopicPricingPolicy(env)()
    np.testing.assert_array_equal(action,[4]+[0]*9)
    assert action.dtype==np.int64
    np.testing.assert_array_equal(env.base_price,np.ones(10))
    np.testing.assert_array_equal(env.sim.current_price,np.ones(10))


def test_myopic_shared_retains_global_immediate_reward_choice():
    env=fixture('shared')
    action=MyopicPricingPolicy(env)()
    assert isinstance(action,int) and action==0


def test_myopic_vector_reward_dominates_any_uniform_price_in_additive_fixture():
    env=fixture('per_sku');action=MyopicPricingPolicy(env)()
    cfg=env.config.reward;inventory=env.observation_components()['inventory']
    def reward(indices):
        p=np.asarray(cfg_dummy.pricing.multipliers)[indices]
        mean=env.sim.mu_for(1,p);sales=np.minimum(mean,inventory)
        return (p*sales-cfg.holding_cost_per_unit*(inventory-sales)
                -cfg.stockout_cost_per_unit*(mean-sales)-cfg.price_change_cost*np.abs(p-1)).sum()
    cfg_dummy=env.config
    assert all(reward(action)>=reward(np.full(10,i)) for i in range(5))
