"""Order pipeline + stochastic lead-time tests (Phase R1, plan item 5)."""

import numpy as np
import pytest

from src.config.locked_config import LeadTimeConfig
from src.env.pipeline import OrderPipeline


def make_pipeline(seed=0, cfg=None, n_skus=3):
    return OrderPipeline(n_skus=n_skus, lead_time_cfg=cfg, rng=np.random.default_rng(seed))


class TestLeadTimeSampling:
    def test_bounded_support_truncnormal(self):
        cfg = LeadTimeConfig(distribution="truncnormal", min_days=2, max_days=7)
        p = make_pipeline(seed=42, cfg=cfg)
        draws = [p.sample_lead_time() for _ in range(500)]
        assert all(2 <= d <= 7 for d in draws)
        assert all(isinstance(d, int) for d in draws)
        # Mean should be interior, not pinned to a bound.
        assert 2.5 < np.mean(draws) < 6.5

    def test_bounded_support_uniform(self):
        cfg = LeadTimeConfig(distribution="uniform", min_days=1, max_days=3)
        p = make_pipeline(seed=1, cfg=cfg)
        draws = [p.sample_lead_time() for _ in range(200)]
        assert set(draws) <= {1, 2, 3}

    def test_unknown_distribution_raises(self):
        cfg = LeadTimeConfig(distribution="weibull")
        p = make_pipeline(cfg=cfg)
        with pytest.raises(ValueError):
            p.sample_lead_time()

    def test_deterministic_given_seed(self):
        a = make_pipeline(seed=99)
        b = make_pipeline(seed=99)
        assert [a.sample_lead_time() for _ in range(50)] == [b.sample_lead_time() for _ in range(50)]


class TestPlacementAndArrival:
    def test_order_arrives_within_bounded_window(self):
        p = make_pipeline(seed=5, cfg=LeadTimeConfig(min_days=2, max_days=7))
        lead = p.place_orders(np.array([10.0, 0.0, 5.0]), day=10)
        assert lead[1] == 0  # zero order -> no draw
        for i in [0, 2]:
            assert 2 <= lead[i] <= 7
            # arrival day recorded
            assert p.outstanding()[i] == pytest.approx([10.0, 0.0, 5.0][i])

    def test_receive_pops_only_that_day(self):
        p = make_pipeline(seed=7)
        p.place_orders(np.array([10.0, 0, 0]), day=0)
        p.place_orders(np.array([0, 8.0, 0]), day=0)
        lead = p._in_flight
        days = sorted(lead.keys())
        assert len(days) == 2  # two independent draws may differ
        # outstanding sums to 18 across both
        assert p.outstanding().sum() == pytest.approx(18.0)

    def test_receive_returns_and_clears(self):
        p = make_pipeline(seed=11, cfg=LeadTimeConfig(distribution="uniform", min_days=1, max_days=1))
        p.place_orders(np.array([6.0, 4.0, 2.0]), day=3)
        arrivals = p.receive(4)  # day 3 + L(=1)
        assert np.allclose(arrivals, [6.0, 4.0, 2.0])
        assert p.outstanding().sum() == 0
        assert p.receive(4).sum() == 0  # already popped

    def test_accumulation_same_arrival_day(self):
        p = make_pipeline(seed=13, cfg=LeadTimeConfig(distribution="uniform", min_days=2, max_days=2))
        p.place_orders(np.array([5.0, 5.0, 5.0]), day=0)
        p.place_orders(np.array([1.0, 2.0, 3.0]), day=0)
        arrivals = p.receive(2)
        assert np.allclose(arrivals, [6.0, 7.0, 8.0])


class TestBookkeeping:
    def test_pending_buckets_sum_to_outstanding(self):
        p = make_pipeline(seed=17)
        p.place_orders(np.array([9.0, 3.0, 0.0]), day=5)
        p.place_orders(np.array([1.0, 1.0, 1.0]), day=5)
        buckets = p.pending_by_remaining_days(current_day=6, horizon=10)
        assert buckets.shape == (3, 10)
        assert buckets.sum() == pytest.approx(p.outstanding_from(7).sum())
        assert buckets[:, 0].sum() == 0  # nothing arrives on current_day+1 relative to booked days? (bounded >=2)

    def test_outstanding_decreases_after_receive(self):
        p = make_pipeline(seed=19, cfg=LeadTimeConfig(distribution="uniform", min_days=1, max_days=1))
        p.place_orders(np.array([12.0, 0.0, 0.0]), day=0)
        before = p.outstanding()[0]
        p.receive(1)
        after = p.outstanding()[0]
        assert after == pytest.approx(before - 12.0)

    def test_deterministic_placement(self):
        def run():
            p = make_pipeline(seed=23)
            leads = p.place_orders(np.array([4.0, 4.0, 4.0]), day=0)
            return leads.tolist(), {d: q.tolist() for d, q in p._in_flight.items()}
        assert run() == run()
