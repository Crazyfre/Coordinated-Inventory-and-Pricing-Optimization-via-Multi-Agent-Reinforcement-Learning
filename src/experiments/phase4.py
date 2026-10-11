"""Small reproducible development experiments; not final research evaluation."""
import numpy as np
from src.env.joint_env import JointRetailEnv
from src.env.replenishment_env import ReplenishmentEnv
from src.env.interface import JointAction
from src.demand.simulator import CalibratedDemandSimulator
from src.demand.forecaster import CalibratedMeanForecaster
from src.env.pipeline import OrderPipeline
from src.config.sku_config import load_skus


class ScenarioPipeline(OrderPipeline):
    """Pair potential supplier lead times by episode seed, day and SKU.

    Reuse R1's distribution and bookkeeping; sparse orders do not shift another
    SKU's scenario. Zero orders still create no shipment or lead-time draw.
    """
    def __init__(self, n_skus, cfg, seed):
        super().__init__(n_skus, cfg)
        self.scenario_seed = seed

    def place_orders(self, qty, day):
        qty = np.asarray(qty, dtype=float)
        leads = np.zeros(self.n_skus, dtype=int)
        for i in np.nonzero(qty > 0)[0]:
            self.rng = np.random.default_rng(np.random.SeedSequence(
                [self.scenario_seed, int(day), int(i)]))
            single = np.zeros(self.n_skus)
            single[i] = qty[i]
            leads += super().place_orders(single, day)
        return leads


class PairedSupplierScenario:
    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        supplier_seed = int(self.np_random.integers(0, 2**63-1))
        # The just-reset pipeline has no orders; observations remain unchanged.
        self.pipeline = ScenarioPipeline(self.n, self.config.lead_time, supplier_seed)
        return result


class CalibratedJointEnv(PairedSupplierScenario, JointRetailEnv):
    pass


class IndependentDemandStream:
    """Split demand RNG once per reset; order-dependent lead draws cannot move it."""
    def __init__(self, sim, prices):
        self.sim, self.prices = sim, prices

    def reset(self, rng):
        self.sim.reset(np.random.default_rng(int(rng.integers(0,2**63-1))))
        self.sim.set_current_price(self.prices)

    def set_current_price(self, prices):
        self.sim.set_current_price(prices)

    def __call__(self, day):
        return self.sim(day)


class CalibratedReplenishmentEnv(PairedSupplierScenario, ReplenishmentEnv):
    def step(self, action):
        # R2 prepares tomorrow's arrivals before returning. Keep development
        # diagnostics aligned with today's reward without changing R2 semantics.
        day = self.day
        arrivals = self._arrivals_today.copy()
        accepted = self._accepted_today.copy()
        clipped = self._clipped_today.copy()
        events = self.inventory.capacity_violation_events
        units = self.inventory.capacity_clipped_units
        obs, reward, term, trunc, info = super().step(action)
        info.update(day=day, arrivals=arrivals, orders_delivered=accepted,
                    orders_clipped=clipped, capacity_violation_events=events,
                    capacity_clipped_units=units)
        if not (term or trunc):
            info['total_on_hand'] -= self._accepted_today.sum()
        return obs, reward, term, trunc, info

    def _push_price_to_demand_source(self):
        super()._push_price_to_demand_source()
        self.forecaster.set_current_price(self._current_price)


def make_env(days=28, capacity=500., joint=True):
    sim = CalibratedDemandSimulator(start_day_index=1800, shock_std=.15)
    lookup = {s.series_id:s.base_price for s in load_skus()}
    prices = np.array([lookup[s] for s in sim.series_ids])
    source = IndependentDemandStream(sim,prices)
    cls = CalibratedJointEnv if joint else CalibratedReplenishmentEnv
    env = cls(episode_length_days=days,capacity=capacity,demand_source=source,
              forecaster=CalibratedMeanForecaster(sim),base_price=prices,
              unit_cost=.6*prices,initial_inventory=np.minimum(sim.level_anchor*2,capacity/10))
    env.sim = sim
    return env


def heuristic(env, kind='ss', s_days=7., S_days=8.):
    b = env.observation_components()
    mean = b['forecast']
    pos = b['inventory']+b['pipeline'].sum(axis=1)
    if kind == 'zero':
        qty = np.zeros(env.n)
    elif kind == 'constant':
        qty = mean
    elif kind == 'ss':
        qty = np.where(pos < s_days*mean, np.maximum(S_days*mean-pos,0),0)
    else:
        raise ValueError('unknown heuristic')
    return np.rint(np.clip(qty/env.config.replenishment.order_step,0,len(env.grid)-1)).astype(np.int64)


def evaluate(env, order_policy, episodes=4, seed=1000, price_policy=None):
    rewards, demands = [], []
    demand_sum = sales_sum = clipped = 0.
    joint = isinstance(env,JointRetailEnv)
    for ep in range(episodes):
        obs,_ = env.reset(seed=seed+ep)
        total, path, done = 0.,[],False
        while not done:
            if joint and env.pricing_due:
                index = 2 if price_policy is None else int(price_policy(obs['pricing']))
                env.select_price(index)
            robs = env.replenishment_observation() if joint else obs
            action = order_policy(env,robs)
            if joint:
                obs,r,done,info = env.step(JointAction(None,action))
            else:
                obs,r,term,trunc,info = env.step(action)
                done = term or trunc
            total += r
            path.append(info['demand'].tolist())
            demand_sum += info['demand'].sum(); sales_sum += info['sales'].sum()
            if info['total_on_hand'] > env.capacity+1e-6:
                raise AssertionError('capacity invariant violated')
        clipped += info['capacity_clipped_units']
        rewards.append(total)
        demands.append(path)
    return {'mean_daily_reward':float(np.mean(rewards)/env.episode_length_days),
            'episode_rewards':rewards,'fill_rate':float(sales_sum/max(demand_sum,1)),
            'clipped_units':float(clipped),'demands':demands}
