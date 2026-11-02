import copy
import time
import pytest
from scripts import recovery_scientific_units as science
from src.experiments.recovery_execution import StepBudget


class Env:
    day = 1
    pricing_due = True
    state = 0


@pytest.mark.parametrize('day', [1, 8, 15, 22])
def test_group_exact_steps_and_independent_futures(monkeypatch, day):
    def reset(env, key):
        env.day = 1
        env.state = key
    def advance(env, covers, first, sku, action, counters, deadline):
        row = dict(day=env.day, reward=float(env.state+action))
        env.state += action
        env.day += 1
        counters['days'] += 1
        return row, env.day == 29
    monkeypatch.setattr(science, 'reset', reset)
    monkeypatch.setattr(science, 'advance', advance)
    monkeypatch.setattr(science, 'snapshot', lambda e: dict(day=e.day, state=e.state))
    monkeypatch.setattr(science, 'resample_future', lambda e, keys: copy.deepcopy(e))
    maximum = day-1+8*(29-day)
    budget = StepBudget(maximum, time.perf_counter()+10, time.perf_counter)
    result = science.paired_group(Env(), None, 10, day, 0,
                                 [dict(replica=r, keys=[r, r+32]) for r in range(4)], budget)
    assert budget.days == maximum <= 224
    assert len(result['prefix']) == day-1
    assert len(result['draws']) == 4
    for draw in result['draws']:
        for branch in draw['branches']:
            assert len(branch['trace']) == 29-day
            assert branch['initial'] == result['before']
    assert result['draws'][0]['branches'] == result['draws'][3]['branches']


def test_reject_incomplete_or_reordered_group_before_work():
    budget = StepBudget(224, time.perf_counter()+10, time.perf_counter)
    for replicas in ([0, 1, 2], [0, 2, 1, 3], [1, 2, 3, 4], [32, 33, 34, 35]):
        with pytest.raises(ValueError):
            science.paired_group(None, None, 10, 1, 0,
                                 [dict(replica=r, keys=[r, r+32]) for r in replicas], budget)
        assert budget.days == 0
