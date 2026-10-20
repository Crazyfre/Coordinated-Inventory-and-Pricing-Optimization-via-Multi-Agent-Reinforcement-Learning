"""Structured mixed-cadence direct policy search; separate from MAPPO."""
import numpy as np
from src.algos.parameter_policy_gradient import order_action, antithetic_gradient


def initial_parameters(covers):
    covers=np.asarray(covers,float)
    if covers.shape!=(10,) or not np.isfinite(covers).all():
        raise ValueError('ten finite cover coefficients required')
    return np.r_[covers,np.full(10,4.),np.zeros(11)]


def price_action(obs,day,parameters):
    x=np.asarray(obs,np.float32);p=np.asarray(parameters,np.float32)
    if x.ndim!=1 or len(x)<40 or p.shape!=(31,) or not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError('finite pricing observation and31 parameters required')
    if int(day)!=day or not 1<=day<=28:
        raise ValueError('episode day1..28 required')
    z=np.clip(x[:10]/(1+7*x[-20:-10]),0,2)-1
    score=p[10:20]+p[20:30]*z+p[30]*np.float32((29-day)/28-.5)
    return np.floor(np.clip(score,0,4)+.5).astype(np.int64)


def ordering_action(obs,day,parameters):
    return order_action(obs,day,np.asarray(parameters)[:10])


class JointParameterGradient:
    def __init__(self,covers,arm,seed):
        if arm not in ('joint','ordering','pricing'):
            raise ValueError('joint,ordering or pricing arm required')
        self.arm=arm;self.mean=initial_parameters(covers)
        self.sigma=np.r_[np.full(20,.75),np.full(11,.5)]
        self.active=np.ones(31,bool)
        if arm=='ordering':self.active[10:]=False
        if arm=='pricing':self.active[:10]=False
        self.m=np.zeros(31);self.v=np.zeros(31);self.steps=0;self.rng=np.random.default_rng(seed)

    def perturbations(self):
        return self.rng.normal(size=(8,31))*self.active

    def update(self,epsilon,differences):
        gradient=antithetic_gradient(epsilon,differences,self.sigma)*self.active
        self.steps+=1;self.m=.9*self.m+.1*gradient;self.v=.999*self.v+.001*gradient**2
        self.mean+=.1*(self.m/(1-.9**self.steps))/(np.sqrt(self.v/(1-.999**self.steps))+1e-8)
        self.mean[:10]=self.mean[:10].clip(0,16)
        self.mean[10:20]=self.mean[10:20].clip(0,4)
        self.mean[20:]=self.mean[20:].clip(-4,4)
        return gradient

    def state(self):
        return dict(arm=self.arm,mean=self.mean.tolist(),sigma=self.sigma.tolist(),active=self.active.tolist(),
            m=self.m.tolist(),v=self.v.tolist(),steps=self.steps,rng=self.rng.bit_generator.state)
