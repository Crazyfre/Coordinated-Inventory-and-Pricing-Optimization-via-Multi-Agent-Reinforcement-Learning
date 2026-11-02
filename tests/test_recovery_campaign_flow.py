"""Complete2436-callback synthetic flow; not simulator or economic evidence."""
import copy
import time
import numpy as np
from scripts import recovery_campaign_dispatch as dispatcher
from scripts import recovery_campaign_science as science
from scripts import recovery_scientific_units as grouped
from scripts.recovery_campaign_plan import build
from src.experiments.recovery_execution import StepBudget
from tests.test_recovery_campaign_plan import study


class FakeEnvironment:
    base_price = np.ones(10)
    day = 1
    _finished = False

    @property
    def pricing_due(self):
        return (self.day-1)%7 == 0

    def pricing_observation(self):
        obs = np.zeros(204)
        obs[:10] = 50
        obs[150:160] = 1.2
        obs[-20:-10] = 2
        return obs

    def step(self, action):
        self.day += 1
        self._finished = self.day == 29


def test_full_callback_sequence(monkeypatch):
    def reset(env,key):
        # Forward reset through wrapper via the underlying object, as real reset
        # forwards the bound env.reset method rather than assigning proxy fields.
        raw = env.env if isinstance(env,science.CountedEnvironment) else env
        raw.day = 1
        raw._finished = False
    def advance(env,covers,first,sku,action,counters,deadline):
        day = env.day
        obs = env.pricing_observation().tolist() if env.pricing_due else None
        env.step(action)
        counters['days'] += 1
        return dict(day=day,reward=0.,pricing_obs=obs),day==28
    def suffix(env,covers,sku,action,counters,deadline):
        first = env.day
        rows = []
        while not env._finished:
            rows.append(advance(env,covers,first,sku,action,counters,deadline)[0])
        return rows
    def collect(env,covers,key,label,models,counters,deadline):
        reset(env,key)
        trace = suffix(env,covers,0,4,counters,deadline)
        return dict(key=key,label=label,reward_day=0.,fill=1.,price_counts=[0,0,0,0,40]),trace
    for module in (science,grouped):
        monkeypatch.setattr(module,'reset',reset)
        monkeypatch.setattr(module,'advance',advance)
        monkeypatch.setattr(module,'snapshot',lambda e:dict(day=e.day))
        monkeypatch.setattr(module,'resample_future',lambda e,k:copy.deepcopy(e))
    monkeypatch.setattr(science,'suffix',suffix)
    monkeypatch.setattr(science,'clone_environment',copy.deepcopy)
    monkeypatch.setattr(science,'collect',collect)
    env = FakeEnvironment()
    monkeypatch.setattr(dispatcher,'environment',lambda s,c:(env,np.ones(10)))
    # Binding/hash validation has separate tests. Here retain full dispatch and
    # science wiring while keeping the synthetic archive in memory.
    monkeypatch.setattr(dispatcher,'validate_request',lambda r:(r['study'],r['unit'],r['dependencies']))
    monkeypatch.setattr(dispatcher,'load_dependency',lambda record:record)
    config = study()
    plan = build(config)
    archive = {}
    total = 0
    for unit in plan['units']:
        deps = {item:archive[item] for item in unit['specification']['dependencies']}
        budget = StepBudget(unit['days'],time.perf_counter()+120,time.perf_counter)
        archive[unit['item']] = dispatcher.dispatch(dict(study=config,unit=unit,dependencies=deps),budget,{})
        total += budget.days
    assert total == archive['selection']['successful_days']
    assert total == 78960+59*archive['selection']['suffix_length_sum']
    assert len(archive['fit']['samples']) == 256
    assert set(archive['fit']['models']) == {'local','cross'}
    assert all(len(s['gains'])==32 for s in archive['fit']['samples'])
    assert archive['replays'] == archive['assess_000']
    assert len(archive['summary']['effects']) == 6
    assert not archive['summary']['primary_pass']
    assert not archive['summary']['representation_pass']
    assert not archive['summary']['heuristic_pass']
