"""Opt-in computation optimization; original supplier/environment code intact.

ScenarioPipeline seeds every positive SKU independently. The installed SciPy
truncnorm.rvs uses one uniform then its inverse CDF. Evaluate those inverse CDFs
as one array, retaining each seed, integer rounding and last RNG state. Frozen
studies never import this module. Uniform/fixed configurations use the original.
"""
import numpy as np
from scipy import stats
from src.experiments.phase4 import ScenarioPipeline
from src.env.pipeline import OrderPipeline


class VectorizedScenarioPipeline(ScenarioPipeline):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if stats.truncnorm._rvs.__func__ is not stats.rv_continuous._rvs:
            raise RuntimeError('Installed SciPy must use the inspected inverse-CDF sampler')

    def place_orders(self,qty,day):
        cfg=self.cfg;lo,hi=cfg.min_days,cfg.max_days
        if cfg.distribution!='truncnormal' or hi==lo:return super().place_orders(qty,day)
        qty=np.asarray(qty,dtype=float);indices=np.nonzero(qty>0)[0]
        leads=np.zeros(self.n_skus,dtype=int)
        if not len(indices):return leads
        mean=cfg.mean_days if cfg.mean_days is not None else (lo+hi)/2.
        std=cfg.std_days if cfg.std_days is not None else max((hi-lo)/3.,.5)
        a,b=(lo-mean)/std,(hi-mean)/std
        uniforms=[]
        for i in indices:
            self.rng=np.random.default_rng(np.random.SeedSequence([self.scenario_seed,int(day),int(i)]))
            uniforms.append(self.rng.uniform())
        draws=stats.truncnorm.ppf(np.asarray(uniforms),a,b,loc=mean,scale=std)
        rounded=np.clip(np.rint(draws),lo,hi).astype(int)
        # Protect integer boundaries from any scalar/vector floating-point
        # discrepancy. The original sampler is authoritative near a half-day.
        close=np.abs(draws-(np.floor(draws)+.5))<1e-10
        for j in np.nonzero(close)[0]:
            i=indices[j]
            rng=np.random.default_rng(np.random.SeedSequence([self.scenario_seed,int(day),int(i)]))
            original=OrderPipeline(self.n_skus,cfg,rng)
            rounded[j]=original.sample_lead_time()
            if j==len(indices)-1:self.rng=rng
        leads[indices]=rounded
        for i,lead in zip(indices,rounded):
            arrival=int(day)+int(lead)
            if arrival not in self._in_flight:self._in_flight[arrival]=np.zeros(self.n_skus)
            self._in_flight[arrival][i]+=qty[i]
        return leads


class VectorizedSupplierEnv:
    """Delegate the original env; replace only a just-reset empty pipeline."""
    def __init__(self,env):self.env=env

    def __getattr__(self,name):return getattr(self.env,name)

    def reset(self,*,seed=None,options=None):
        result=self.env.reset(seed=seed,options=options);original=self.env.pipeline
        if type(original) is not ScenarioPipeline or original._in_flight:
            raise ValueError('fresh empty canonical ScenarioPipeline required')
        pipeline=VectorizedScenarioPipeline(original.n_skus,original.cfg,original.scenario_seed)
        pipeline.rng=original.rng;pipeline._in_flight=original._in_flight
        self.env.pipeline=pipeline
        return result
