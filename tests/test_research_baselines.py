import numpy as np
import pytest
from src.experiments.phase4 import make_env


def test_eoq_zero_setup_cost_uses_smallest_positive_grid_batch():
    from src.baselines.retail import EOQPolicy
    env = make_env(days=7)
    env.reset(seed=3)
    policy = EOQPolicy(setup_cost=0)
    action = policy(env, env.replenishment_observation())
    assert action.shape == (10,)
    assert np.all((action == 0) | (action == 1))
    assert np.any(action == 1)


def test_eoq_does_not_order_when_forecast_is_zero():
    from src.baselines.retail import EOQPolicy
    env = make_env(days=7)
    env.observation_components = lambda: dict(forecast=np.zeros(10),inventory=np.zeros(10),pipeline=np.zeros((10,7)))
    assert np.array_equal(EOQPolicy()(env),np.zeros(10,dtype=np.int64))


def test_myopic_prices_maximize_forecast_day_reward_without_mutating_simulator():
    from src.baselines.retail import MyopicPricingPolicy
    env = make_env(days=7)
    env.reset(seed=4)
    original = env.sim._price.copy()
    policy = MyopicPricingPolicy(env)
    scores = []
    b = env.observation_components()
    for multiplier in env.config.pricing.multipliers:
        price = env.base_price*multiplier
        mean = env.sim.mu_for(env.day, price)
        sales = np.minimum(mean, b['inventory'])
        scores.append(np.sum(price*sales - env.config.reward.holding_cost_per_unit*(b['inventory']-sales)
                             - env.config.reward.stockout_cost_per_unit*(mean-sales)
                             - env.config.reward.price_change_cost*np.abs(price-b['price'])))
    assert policy(env.pricing_observation()) == int(np.argmax(scores))
    assert np.array_equal(original, env.sim._price)
