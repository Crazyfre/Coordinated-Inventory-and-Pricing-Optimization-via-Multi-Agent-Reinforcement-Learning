import numpy as np
from types import SimpleNamespace


def test_forecast_controller_uses_staged_price_without_sampling():
    from src.baselines.operating import ForecastBaseStock
    class Sim:
        def mu_for(self, day, price):
            assert day == 1
            return 10 / price
    env = SimpleNamespace(day=1, n=1, sim=Sim(), _pending_price=np.array([2.]),
        episode_length_days=28, grid=np.arange(0,205,5),
        config=SimpleNamespace(replenishment=SimpleNamespace(order_step=5)),
        observation_components=lambda: dict(inventory=np.array([0.]), pipeline=np.zeros((1,7)),price=np.array([1.])))
    assert ForecastBaseStock(cover_days=6)(env).tolist() == [6]
    env._pending_price = None
    env.observation_components = lambda: dict(inventory=np.array([2.]), pipeline=np.array([[7.,0,0,0,0,0,0]]), price=np.array([2.]))
    # Target 30 minus position 9 = 21; ceiling sends 25 units, never grid indices as quantities.
    assert ForecastBaseStock(cover_days=6)(env).tolist() == [5]
    assert ForecastBaseStock(cover_days=1000)(env).tolist() == [40]


def test_controller_rejects_invalid_cover():
    import pytest
    from src.baselines.operating import ForecastBaseStock
    for cover in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            ForecastBaseStock(cover)


def test_replay_preserves_receiving_day_and_inventory_balance():
    from scripts.diagnose_operations import replay
    from src.experiments.research import make_research_env
    from src.baselines.operating import ForecastBaseStock
    env = make_research_env(days=8)
    rows = replay(env, ForecastBaseStock(), lambda obs: 4, 987, 'test')
    assert [r['day'] for r in rows] == list(range(1,9))
    for r in rows:
        np.testing.assert_allclose(np.array(r['inventory_start'])-r['sales'],r['inventory_end'])
        np.testing.assert_allclose(np.array(r['arrivals'])-r['accepted'],r['clipped'])
        assert len(r['order_lead_times']) == env.n
        assert all(2 <= lead <= 7 for qty,lead in zip(r['orders'],r['order_lead_times']) if qty > 0)
        terms = r['reward_terms']
        assert np.isclose(r['reward'], terms['revenue']-sum(v for k,v in terms.items() if k!='revenue'))
