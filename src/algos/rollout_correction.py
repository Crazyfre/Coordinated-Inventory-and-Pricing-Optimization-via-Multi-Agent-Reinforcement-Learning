"""Explicit-rule initialization with a reward-fitted deterministic correction."""
import numpy as np
from scripts.diagnose_residual_support import project_target


def features(obs, n, horizon, sku, delta, day):
    obs = np.asarray(obs)
    forecast = obs[-n:].astype(float)
    position = obs[:n] + obs[n:n+n*horizon].reshape(n,horizon).sum(axis=1)
    # Observation layout ends with price, last lead, headroom, forecast.
    headroom = float(obs[-2*n+sku])
    log = lambda x: np.sign(x)*np.log1p(abs(x))
    base = np.array([1.,log(forecast[sku]/5),log(position[sku]/5),
                     headroom/500,(29-day)/28,forecast[sku]/max(float(forecast.sum()),1e-12)])
    return np.concatenate([delta*base,delta*delta*base])


class RolloutCorrection:
    def __init__(self, center, target, n, horizon, weights=None, scale=None):
        self.center,self.target,self.n,self.horizon=center,target,n,horizon
        self.weights=np.zeros(12) if weights is None else np.asarray(weights,dtype=float)
        self.scale=np.ones(12) if scale is None else np.asarray(scale,dtype=float)

    def action(self, obs, day):
        c=self.center.predict(obs)
        action=project_target(c,self.target.act(obs)[0])
        best=1.
        choice=None
        for sku in range(self.n):
            for a in range(max(0,int(c[sku])-2),min(40,int(c[sku])+2)+1):
                x=features(obs,self.n,self.horizon,sku,a-int(action[sku]),day)
                gain=float((x/self.scale)@self.weights)
                if gain>best:
                    best=gain;choice=(sku,a)
        if choice is not None:
            action[choice[0]]=choice[1]
        return action


def fit_ridge(x,y):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-8)
    z=x/scale
    weights=np.linalg.solve(z.T@z+10*np.eye(x.shape[1]),z.T@y)
    return weights,scale
