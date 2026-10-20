"""Episodic Gaussian parameter-space policy gradient for structured inventory policies."""
import numpy as np


def initial_mean(arm):
    if arm not in ('sku', 'arrival'):
        raise ValueError('sku or arrival arm required')
    return np.r_[np.full(10, 7.5), np.zeros(11 if arm == 'arrival' else 0)]


def sigmas(arm):
    return np.r_[np.full(10, .75), np.full(11 if arm == 'arrival' else 0, .5)]


def order_action(obs, day, parameters):
    x = np.asarray(obs, np.float32); p = np.asarray(parameters, np.float32)
    if x.shape != (260,) or p.shape not in [(10,), (21,)] or not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError('finite canonical observation and10/21 parameters required')
    if int(day) != day or not 1 <= day <= 28:
        raise ValueError('episode day1..28 required')
    pipeline = x[10:80].reshape(10, 7)
    cover = p[:10].copy()
    if len(p) == 21:
        share = pipeline[:, 3:].sum(1)/(1+pipeline.sum(1))
        cover += p[10:20]*share+p[20]*np.float32((29-day)/28-.5)
    cover = cover.clip(0, 20)
    position = x[:10]+pipeline.sum(1)
    return np.ceil(np.maximum(cover*x[-10:]-position, 0)/5).clip(0, 40).astype(np.int64)


def antithetic_gradient(epsilon, differences, sigma):
    e, d, s = np.asarray(epsilon, float), np.asarray(differences, float), np.asarray(sigma, float)
    if e.ndim != 2 or d.shape != (len(e),) or s.shape != (e.shape[1],) or np.any(s <= 0):
        raise ValueError('matching paired differences, perturbations and positive scales required')
    if not all(np.isfinite(v).all() for v in [e, d, s]):
        raise ValueError('finite gradient inputs required')
    return (d[:, None]*e/(2*s)).mean(0)


class ParameterPolicyGradient:
    def __init__(self, arm, seed):
        self.arm = arm; self.mean = initial_mean(arm); self.sigma = sigmas(arm)
        self.rng = np.random.default_rng(seed)
        self.m = np.zeros_like(self.mean); self.v = np.zeros_like(self.mean); self.steps = 0

    def update(self, epsilon, differences):
        gradient = antithetic_gradient(epsilon, differences, self.sigma)
        self.steps += 1
        self.m = .9*self.m+.1*gradient; self.v = .999*self.v+.001*gradient**2
        self.mean += .1*(self.m/(1-.9**self.steps))/(np.sqrt(self.v/(1-.999**self.steps))+1e-8)
        self.mean[:10] = self.mean[:10].clip(0, 16); self.mean[10:] = self.mean[10:].clip(-8, 8)
        return gradient

    def state(self):
        return dict(arm=self.arm, mean=self.mean.tolist(), sigma=self.sigma.tolist(),
                    m=self.m.tolist(), v=self.v.tolist(), steps=self.steps, rng=self.rng.bit_generator.state)
