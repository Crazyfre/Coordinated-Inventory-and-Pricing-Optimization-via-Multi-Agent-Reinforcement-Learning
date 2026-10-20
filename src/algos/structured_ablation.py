"""Declared parameter-sharing and noise-coupling controls for structured RL."""
import numpy as np
from src.algos.parameter_policy_gradient import antithetic_gradient


class StructuredControl:
    def __init__(self, condition, seed):
        if condition not in ('main','pooled','independent'):
            raise ValueError('unknown structured control')
        self.condition=condition
        dimension=1 if condition=='pooled' else 10
        self.mean=np.full(dimension,7.5);self.sigma=np.full(dimension,.75)
        self.m=np.zeros(dimension);self.v=np.zeros(dimension);self.steps=0
        self.rng=np.random.default_rng(seed)

    def expand(self, parameters):
        p=np.asarray(parameters,float)
        if p.shape!=self.mean.shape or not np.isfinite(p).all():
            raise ValueError('matching finite parameter vector required')
        return np.repeat(p,10) if self.condition=='pooled' else p.copy()

    def update(self, epsilon, differences):
        gradient=antithetic_gradient(epsilon,differences,self.sigma)
        self.steps+=1;self.m=.9*self.m+.1*gradient;self.v=.999*self.v+.001*gradient**2
        self.mean+=.1*(self.m/(1-.9**self.steps))/(np.sqrt(self.v/(1-.999**self.steps))+1e-8)
        self.mean=self.mean.clip(0,16)
        return gradient

    def state(self):
        return dict(condition=self.condition,mean=self.mean.tolist(),sigma=self.sigma.tolist(),
            m=self.m.tolist(),v=self.v.tolist(),steps=self.steps,rng=self.rng.bit_generator.state)


def pair_keys(condition, seed_index, pair_index):
    if condition not in ('main','pooled','independent') or not 0<=seed_index<10 or not 0<=pair_index<512:
        raise ValueError('valid condition,seed index and pair index required')
    base={'main':1800000,'pooled':1900000,'independent':2000000}[condition]+seed_index*2000
    plus=base+(2*pair_index if condition=='independent' else pair_index)
    return plus,plus+1 if condition=='independent' else plus
