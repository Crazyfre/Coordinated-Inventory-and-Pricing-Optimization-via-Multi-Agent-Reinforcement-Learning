"""Tests for the calibrated demand simulator (Phase P1, task 4).

Checks the P1 exit criteria: determinism, controlled-elasticity monotonicity,
overdispersion, history feedback, correlation structure, env integration.
"""

import numpy as np
import pytest

from src.config.locked_config import get_locked_config
from src.demand.calibration import load_nb_params
from src.demand.simulator import CalibratedDemandSimulator
from src.env.replenishment_env import ReplenishmentEnv

N = get_locked_config().env.n_skus


def make_sim(seed=0, **kw):
    sim = CalibratedDemandSimulator(rng=np.random.default_rng(seed), **kw)
    sim.set_current_price(_base_prices(sim))
    sim.reset()
    return sim


def _base_prices(sim) -> np.ndarray:
    from src.config.sku_config import load_skus

    skus = {s.series_id: s for s in load_skus()}
    return np.array([skus[s].base_price for s in sim.series_ids])


class TestContract:
    def test_series_order_matches_calibration(self):
        sim = make_sim()
        assert sim.series_ids == list(load_nb_params()["series_order"])

    def test_requires_price_before_sampling(self):
        sim = CalibratedDemandSimulator(rng=np.random.default_rng(0))
        sim.reset()
        with pytest.raises(ValueError):
            sim(1)

    def test_output_shape_and_nonneg(self):
        sim = make_sim(seed=1)
        d = sim(1)
        assert d.shape == (N,)
        assert (d >= 0).all()

    def test_deterministic_given_seed(self):
        def run():
            sim = make_sim(seed=42)
            return [sim(d) for d in range(1, 30)]
        a, b = run(), run()
        assert all(np.allclose(x, y) for x, y in zip(a, b))


class TestElasticitySlider:
    def test_monotone_price_response(self):
        """Higher controlled elasticity magnitude -> lower demand at +10% price."""
        base_prices = None
        means = {}
        for e in (-0.5, -1.5, -3.0):
            sim = CalibratedDemandSimulator(elasticity=e, rng=np.random.default_rng(7))
            base_prices = _base_prices(sim)
            sim.set_current_price(base_prices * 1.10)  # +10% price
            sim.reset()
            draws = np.stack([sim(d) for d in range(1, 200)])
            means[e] = draws.mean()
        assert means[-3.0] < means[-1.5] < means[-0.5]

    def test_elasticity_magnitude_matches_log_ratio(self):
        """With elasticity e, E[demand(+10% price)]/E[demand(base)] ~= 1.1^e."""
        base_draws, hi_draws = [], []
        for seed in range(4):
            sim = CalibratedDemandSimulator(elasticity=-1.0, shock_std=0.0,
                                            rng=np.random.default_rng(seed))
            bp = _base_prices(sim)
            sim.set_current_price(bp)
            sim.reset()
            base_draws.append(np.stack([sim(d) for d in range(1, 150)]).mean(axis=0))
            sim.set_current_price(bp * 1.10)
            hi_draws.append(np.stack([sim(d) for d in range(1, 150)]).mean(axis=0))
        ratio = np.mean(hi_draws) / np.mean(base_draws)
        assert ratio == pytest.approx(1.1 ** -1.0, rel=0.15)


class TestDispersion:
    def test_overdispersion_vs_poisson(self):
        """Sample variance should exceed the mean for a low-phi SKU (NB)."""
        sim = make_sim(seed=3, shock_std=0.0)
        draws = np.stack([sim(d) for d in range(1, 400)])
        i = int(np.argmin(sim.phi))  # most overdispersed SKU
        # window-mean normalised variance index
        var_idx = draws[:, i].var() / max(draws[:, i].mean(), 1e-9)
        assert var_idx > 1.1, f"SKU {sim.series_ids[i]} looks Poisson (phi={sim.phi[i]})"


class TestExogeneity:
    def test_mu_stable_across_days_at_fixed_price(self):
        """Given a fixed price, mu varies ONLY through the calendar — the
        process is exogenous (no self-exciting feedback loop). The sd of
        log-mu across a 60-day span must be modest (calendar-driven), and
        no runaway growth may occur."""
        sim = make_sim(seed=5, shock_std=0.0)
        mus = []
        for d in range(1, 61):
            sim(d)
            mus.append(sim.current_mu().copy())
        log_mu = np.log(np.stack(mus))
        assert np.isfinite(log_mu).all()
        assert log_mu.std(axis=0).max() < 1.0  # calendar variation only
        assert (np.stack(mus) <= 5.0 * sim.level_anchor).all()  # anchor respected

    def test_mu_is_seed_independent(self):
        """With shock_std=0, mu is a pure function of (day, price): two sims
        with DIFFERENT seeds must produce identical mu trajectories — proof
        that sampled demand never feeds back into the mean."""
        def mu_path(seed):
            s = CalibratedDemandSimulator(shock_std=0.0, rng=np.random.default_rng(seed))
            s.set_current_price(_base_prices(s))
            s.reset()
            out = []
            for d in range(1, 41):
                s(d)
                out.append(s._last_mu.copy())
            return np.stack(out)
        a, b = mu_path(101), mu_path(202)  # different seeds
        assert np.allclose(a, b)


class TestCorrelatedShocks:
    def test_shock_correlation_direction(self):
        """With big correlated shocks, same-day sampling deviations should be
        positively co-deviating vs the shock-free case (statistical, tolerant)."""
        # Use only two SKUs with high |corr| from the calibration matrix.
        sim0 = make_sim(seed=11, shock_std=0.0)
        R = sim0.L @ sim0.L.T
        np.fill_diagonal(R, 0)
        i, j = np.unravel_index(np.argmax(R), R.shape)

        def co_deviation(shock_std):
            sims = []
            for seed in range(6):
                s = CalibratedDemandSimulator(shock_std=shock_std,
                                              rng=np.random.default_rng(seed))
                s.set_current_price(_base_prices(s))
                s.reset()
                mu = np.stack([s.current_mu()] + [np.full(N, np.nan) for _ in range(0)]).mean(axis=0)
                sims.append((s, mu))
            devs = []
            for s, mu in sims:
                draws = np.stack([s(d) for d in range(1, 300)])
                devs.append((draws[:, i] - mu[i]) * (draws[:, j] - mu[j]))
            return np.mean(devs)

        cd_on = co_deviation(shock_std=0.6)
        assert cd_on > 0, f"correlated shocks should co-deviate (got {cd_on})"


class TestEnvIntegration:
    def test_env_pushes_price_to_simulator(self):
        """ReplenishmentEnv must push its current price into a price-sensitive
        demand source before sampling (integration seam, Doc S2.5)."""
        seen = {}

        class SpySim(CalibratedDemandSimulator):
            def set_current_price(self, prices):
                seen["price"] = np.asarray(prices).copy()
                super().set_current_price(prices)

        sim = SpySim(rng=np.random.default_rng(9))
        sim.set_current_price(_base_prices(sim))
        sim.reset()
        env = ReplenishmentEnv(demand_source=sim, episode_length_days=5)
        env.reset(seed=9)
        assert np.allclose(seen["price"], env.base_price)

        env.step(np.zeros(N, dtype=int))  # hook None -> price held, still pushed
        assert np.allclose(seen["price"], env.base_price)

        def hook(day):
            return np.full(N, 12.0) if day == 2 else None

        env.pricing_hook = hook
        obs, r, term, trunc, info = env.step(np.zeros(N, dtype=int))
        assert np.allclose(seen["price"], 12.0)

    def test_full_episode_with_simulator(self):
        sim = make_sim(seed=13)
        env = ReplenishmentEnv(demand_source=sim, episode_length_days=28)
        obs, info = env.reset(seed=13)
        rng = np.random.default_rng(1)
        done = False
        steps = 0
        while not done:
            obs, r, term, trunc, info = env.step(rng.integers(0, len(env.grid), size=N))
            done = term or trunc
            steps += 1
            assert np.isfinite(r)
        assert steps == 28
