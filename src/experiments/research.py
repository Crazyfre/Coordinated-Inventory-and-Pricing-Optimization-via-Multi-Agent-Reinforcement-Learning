"""Research regime additions; never changes the locked configuration."""
SPLITS = {'train': (15, 1885), 'validation': (1886, 1913), 'test': (1914, 1941)}


def window(split, start, days):
    lo, hi = SPLITS[split]
    if days < 1 or start < lo or start + days - 1 > hi:
        raise ValueError('episode crosses chronological split')
    return start, start + days - 1


import numpy as np
from dataclasses import replace
from src.experiments.phase4 import CalibratedJointEnv, IndependentDemandStream
from src.demand.simulator import CalibratedDemandSimulator
from src.demand.forecaster import CalibratedMeanForecaster
from src.config.locked_config import get_locked_config
from src.config.sku_config import load_skus

SCENARIOS = {
    'S1': dict(shock=0., phi_factor=10., calendar=False),
    'S2': dict(shock=.15),
    'S3': dict(shock=.4, phi_factor=.35),
    'S4': dict(shock=.15, elasticity=-3.),
    'S5': dict(shock=.5),
    'S6': dict(shock=.15),
    'S7': dict(shock=.15, shift=1.5),
    'S8': dict(shock=.15, capacity=250.),
}
ABLATIONS = {'A1': 'independent critics (requires retraining)',
             'A2': 'zero demand history', 'A3': 'zero replenishment price',
             'A4': 'zero pricing inventory and headroom',
             'A5': 'fixed four-day supplier lead',
             'A6': 'identity residual correlation', 'A7': 'unbounded on-hand capacity'}


class ResearchEnv(CalibratedJointEnv):
    """Dimension-preserving observation masks; use identically during training."""
    ablation = None

    def _build_obs(self):
        obs = super()._build_obs()
        n, k, h = self.n, self.k, self.obs_pipe_horizon
        if self.ablation == 'A2':
            obs[n*(1+h):n*(1+h+k)] = 0
        if self.ablation == 'A3':
            obs[n*(1+h+k):n*(2+h+k)] = 0
        if self.ablation == 'A7':
            obs[n*(3+h+k):n*(4+h+k)] = 500.  # finite neutral capacity feature
        return obs

    def pricing_observation(self):
        obs = super().pricing_observation()
        if self.ablation == 'A2':
            obs[self.n:self.n*(1+self.k)] = 0
        if self.ablation == 'A4':
            obs[:self.n] = 0
            obs[-self.n:] = 0
        if self.ablation == 'A7':
            obs[-self.n:] = 500.
        return obs


class CounterfactualDemand(CalibratedDemandSimulator):
    """Day-indexed normal/uniform shocks, invariant to prior prices and orders.

    Actual counts differ with price: CRN pairs exogenous variates, not endogenous
    demand outcomes. NB inverse CDF avoids parameter-dependent RNG consumption.
    """
    def reset(self, rng=None, start_day_index=None):
        super().reset(rng, start_day_index)
        self.scenario_seed = int(self.rng.integers(0, 2**63-1))

    def _mu_mean(self, env_day, prices, *, with_shocks=False):
        # R2 labels history -k..-1, omitting zero; map it to contiguous calendar
        # dates start-k..start-1 rather than modulo-wrapping day zero into test.
        mapped = env_day+1 if env_day<0 else env_day
        actual = self.start_day_index+mapped-1
        if not 1 <= actual <= len(self.calendar_feats):
            raise ValueError('calendar wraparound forbidden in research regime')
        return super()._mu_mean(mapped,prices,with_shocks=with_shocks)

    def __call__(self, day):
        from scipy.stats import nbinom
        self.rng = np.random.default_rng(np.random.SeedSequence(
            [self.scenario_seed, int(day)+10000]))
        mu = self._mu_mean(day, self._price, with_shocks=True)
        self._last_mu = mu.copy()
        return nbinom.ppf(self.rng.uniform(1e-12, 1-1e-12, self.n),
                          self.phi, self.phi/(self.phi+mu)).astype(float)


def training_statistics_view(data):
    """Preserve future calendar but redact all future outcome-derived anchors."""
    df = data.df.copy()
    cols = ['y', 'hist_7', 'hist_28']
    means = df[df.day_index<=1885].groupby('series_id')[cols].mean()
    future = df.day_index>1885
    for col in cols:
        df[col] = df[col].astype(float)
        df.loc[future, col] = df.loc[future,'series_id'].map(means[col])
    return replace(data, df=df)


def fit_training_calibration(data=None):
    """Fit without touching the legacy nb_params artifact or reading test scores."""
    import pandas as pd
    from src.demand.alignment import build_aligned_dataset
    from src.demand.calibration import model_exog, _fit_series, predict_mu, pearson_residuals
    data = data or build_aligned_dataset()
    train = data.df[(data.df.day_index<=1885) & (data.df.price_observed==1)]
    skus, residuals = {}, {}
    for sid in data.series_ids:
        tr = train[train.series_id==sid]
        X = model_exog(tr); y = tr.y.to_numpy(float)
        beta, alpha = _fit_series(y, X)
        skus[sid] = dict(params=dict(zip(X.columns, map(float,beta))),
                         alpha=alpha, phi=1/alpha, n_train_days=len(tr))
        residuals[sid] = pd.Series(pearson_residuals(y,predict_mu(beta,X),alpha),index=tr.day_index)
    corr = pd.DataFrame(residuals).dropna().corr().to_numpy()
    return dict(status='research_train_only', fit_end_day=1885,
                outcome_statistics_end_day=1885, series_order=list(data.series_ids),
                skus=skus, residual_correlation=dict(matrix=corr.tolist()))


def make_research_env(scenario='S2', ablation=None, *, split='validation',
                      start=None, days=28, params=None, aligned=None, elasticity=None,
                      pricing_scope="shared"):
    start = SPLITS[split][0] if start is None else start
    window(split, start, days)
    if aligned is None:
        from src.demand.alignment import build_aligned_dataset
        aligned = training_statistics_view(build_aligned_dataset())
    knobs = SCENARIOS[scenario]
    elasticity = knobs.get('elasticity',-1.5) if elasticity is None else elasticity
    if elasticity >= 0:
        raise ValueError('controlled elasticity must be negative')
    sim = CounterfactualDemand(nb_params=params, aligned=aligned,
        start_day_index=start, shock_std=knobs['shock'], elasticity=elasticity)
    sim.phi *= knobs.get('phi_factor', 1.)
    if not knobs.get('calendar', True):
        sim.b0 += sim.b_cal @ sim.calendar_feats[start-1]
        sim.b_cal[:] = 0
    if 'shift' in knobs:
        sim.b0 += np.log(knobs['shift'])
    if ablation == 'A6':
        sim.L = np.eye(sim.n)
    lookup = {s.series_id:s.base_price for s in load_skus()}
    prices = np.array([lookup[s] for s in sim.series_ids])
    cfg = get_locked_config()
    if ablation == 'A5':
        cfg = replace(cfg, lead_time=replace(cfg.lead_time, min_days=4, max_days=4))
    capacity = knobs.get('capacity', 500.) if ablation != 'A7' else float('inf')
    env = ResearchEnv(config=cfg, pricing_scope=pricing_scope, obs_pipe_horizon=7, episode_length_days=days, capacity=capacity,
        demand_source=IndependentDemandStream(sim, prices),
        forecaster=CalibratedMeanForecaster(sim), base_price=prices,
        unit_cost=.6*prices, initial_inventory=np.minimum(sim.level_anchor*2, 50.))
    env.sim, env.ablation = sim, ablation
    return env
