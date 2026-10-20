import pytest
from src.experiments import research


def test_chronological_windows_reject_crossing():
    assert research.window('validation', 1886, 28) == (1886, 1913)
    with pytest.raises(ValueError):
        research.window('train', 1800, 112)
    with pytest.raises(ValueError):
        research.window('test', 1914, 29)
    assert research.window('test', 1914, 28) == (1914, 1941)


def test_scenarios_and_ablations_have_real_knobs():
    import numpy as np
    assert set(research.SCENARIOS) == {f'S{i}' for i in range(1, 9)}
    assert set(research.ABLATIONS) == {f'A{i}' for i in range(1, 8)}
    env = research.make_research_env('S1', 'A2', days=7)
    obs, _ = env.reset(seed=123)
    assert np.all(obs['pricing'][10:150] == 0)
    assert np.all(obs['replenishment'][80:220] == 0)
    assert np.all(env.sim.shock_std == 0)
    fixed = research.make_research_env('S6', 'A5', days=7)
    fixed.reset(seed=2)
    assert fixed.config.lead_time.min_days == fixed.config.lead_time.max_days == 4
    fobs, _ = fixed.reset(seed=2)
    assert len(fobs['replenishment']) == len(obs['replenishment'])
    uncorr = research.make_research_env('S5', 'A6', days=7)
    assert np.array_equal(uncorr.sim.L, np.eye(10))
    loose = research.make_research_env('S8', 'A7', days=7)
    assert np.isinf(loose.capacity)


def test_seed_library_and_price_counterfactual_pairing():
    import numpy as np
    from src.eval.research import seed_library
    seeds = seed_library(5, 20)
    assert len(set(seeds)) == 5 and seeds == seed_library(5, 20)
    a = research.make_research_env(days=7)
    b = research.make_research_env(days=7)
    a.reset(seed=seeds[0]); b.reset(seed=seeds[0])
    # Earlier different prices must not shift tomorrow's exogenous variates.
    a.sim.set_current_price(a.base_price*.8)
    a.sim(1)
    b.sim.set_current_price(b.base_price*1.2)
    b.sim(1)
    a.sim.set_current_price(a.base_price)
    b.sim.set_current_price(b.base_price)
    assert np.array_equal(a.sim(2), b.sim(2))


def test_episode_metrics_and_paired_confidence_intervals():
    from src.eval.research import evaluate_episode, summarize, paired_difference
    from src.experiments.phase4 import heuristic
    env = research.make_research_env(days=7)
    row = evaluate_episode(env, lambda e, o: heuristic(e), lambda o: 2, 11)
    assert row['profit'] == pytest.approx(row['revenue']-row['ordering_cost']-row['holding_cost']-row['stockout_cost']-row['price_change_cost'])
    assert row['fill_rate'] == pytest.approx(row['sales']/row['demand'])
    assert row['capacity_violations'] == 0
    assert row['stockout_rate'] >= 0
    assert row['price_volatility'] == 0
    s = summarize([row, row])
    assert s['profit']['ci_low'] == s['profit']['ci_high'] == row['profit']
    assert paired_difference([1, 4, 7], [0, 3, 6])['mean'] == 1


def test_final_gate_refuses_legacy_calibration_and_smoke_models():
    from src.eval.research import validate_final
    with pytest.raises(ValueError, match='calibration'):
        validate_final({}, [{}]*10, 100)
    with pytest.raises(ValueError, match='checkpoint'):
        validate_final({'fit_end_day':1885, 'outcome_statistics_end_day':1885}, [{}]*10, 100)
    manifests = [dict(training_seed=i, training_end_day=1885, split='train',
                      updates=100, frozen=True, ablation='none') for i in range(10)]
    validate_final({'fit_end_day':1885,'outcome_statistics_end_day':1885}, manifests, 100)
    with pytest.raises(ValueError, match='50'):
        validate_final({'fit_end_day':1885,'outcome_statistics_end_day':1885}, manifests, 5)


def test_unbounded_capacity_has_finite_observations():
    import numpy as np
    env = research.make_research_env('S8', 'A7', days=7)
    obs, _ = env.reset(seed=1)
    assert all(np.isfinite(v).all() for v in obs.values())
    assert np.isfinite(env.get_state()).all()


def test_simulator_outcome_statistics_ignore_future_rows():
    from dataclasses import replace
    from src.demand.alignment import build_aligned_dataset
    data = build_aligned_dataset()
    clean = research.training_statistics_view(data)
    altered = data.df.copy()
    altered.loc[altered.day_index>1885, ['y','hist_7','hist_28']] = 1e6
    again = research.training_statistics_view(replace(data, df=altered))
    assert clean.df[['y','hist_7','hist_28']].equals(again.df[['y','hist_7','hist_28']])


def test_train_only_calibration_has_provenance_and_no_test_scores():
    payload = research.fit_training_calibration()
    assert payload['fit_end_day'] == payload['outcome_statistics_end_day'] == 1885
    assert len(payload['skus']) == 10
    assert all('validation' not in s for s in payload['skus'].values())
    assert all(s['phi']>0 for s in payload['skus'].values())


def test_first_training_warmup_never_wraps_into_test_calendar():
    import numpy as np
    env = research.make_research_env(split='train',start=15,days=7)
    env.reset(seed=1)
    sim = env.sim
    original = sim.start_day_index
    warm = sim.mu_for(-14,env.base_price)
    sim.start_day_index=1
    expected = sim.mu_for(1,env.base_price)
    sim.start_day_index=original
    assert np.array_equal(warm,expected)


def test_default_factory_redacts_future_anchors():
    from src.demand.alignment import build_aligned_dataset
    import numpy as np
    data=build_aligned_dataset()
    env=research.make_research_env(days=7)
    expected=data.df[data.df.day_index<=1885].groupby('series_id').y.mean()
    assert np.allclose(env.sim.level_anchor,[expected[s] for s in env.sim.series_ids])


def test_explicit_elasticity_sweep_override_keeps_controlled_negative_response():
    env=research.make_research_env('S4',days=7,elasticity=-.5)
    assert (env.sim.elasticity==-.5).all()
    with pytest.raises(ValueError,match='negative'):
        research.make_research_env('S4',days=7,elasticity=.5)
