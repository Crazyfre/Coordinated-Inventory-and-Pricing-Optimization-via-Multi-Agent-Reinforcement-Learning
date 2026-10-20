import numpy as np
import pytest
from src.eval.parameter_review import episode_metrics, crossed, progress_threshold, SEEDS


def test_episode_economics_and_service_are_not_reward_proxies():
    trace = [dict(day=d, label='x', scenario=1, start=15, reward=2., info=dict(
        demand=[10.], sales=[9.], lost_sales=[1.], total_on_hand=4., orders_clipped=[2.],
        orders_delivered=[3.], reward_terms=dict(revenue=10., ordering_cost=3., holding_cost=1.,
        stockout_cost=4., price_change_cost=0.))) for d in range(1,29)]
    m = episode_metrics(trace)
    assert m['episodic_return'] == 56 and m['fill'] == .9 and m['success'] == 0
    assert m['clipped_units'] == 56 and m['mean_inventory'] == 4
    trace[0]['reward'] = 5.
    with pytest.raises(ValueError, match='reconstruct'):
        episode_metrics(trace)


def test_crossed_constant_effect_and_training_variability():
    assert crossed(np.full((5,32), 2.))['ci95'] == [2.,2.]
    result = crossed(np.repeat(np.arange(5)[:,None], 32, axis=1))
    assert result['ci95'][0] < 2 < result['ci95'][1]


def test_threshold_reuses_exact_start_without_duplicate_rollouts():
    rows = [dict(label='full7_5', scenario=k, reward_day=100.+k) for k in range(3)]
    rows += [dict(label=f'u{u}_{s}', scenario=k, reward_day=100.+k+(u/16))
             for s in SEEDS for u in [16,32,48,64] for k in range(3)]
    result = progress_threshold(rows)
    assert result['211']['threshold']['confirmed_step'] == 48*16*28
    assert result['211']['threshold']['onset_step'] == 16*16*28
