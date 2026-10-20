"""Train-label-fitted inventory-position policy; deterministic imitation, not PPO."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linprog


def position_features(obs, n, pipe_horizon):
    """Actor-local forecast and inventory position, expressed in grid steps.

    Preserve float32 pipeline aggregation used by the environment's observations.
    No teacher coefficient, future outcomes or realized future demand are inputs.
    """
    x=torch.as_tensor(obs,dtype=torch.float32)
    position=x[..., :n]+x[..., n:n*(1+pipe_horizon)].reshape(*x.shape[:-1],n,pipe_horizon).sum(-1)
    return np.stack([x[..., -n:].numpy().astype(float)/5,
                     position.numpy().astype(float)/5,np.ones(position.shape)],axis=-1)


def fit_intervals(features, labels):
    """Maximum-margin fit to ceil/clipped action-bin constraints, train data only."""
    x=np.asarray(features,float).reshape(-1,3)
    y=np.asarray(labels).reshape(-1)
    if len(x)!=len(y) or not len(y) or not np.isfinite(x).all() or not np.isin(y,np.arange(41)).all():
        raise ValueError('finite matching features and integer labels 0..40 required')
    # For y=0: z<=0; for y=40: z>39; otherwise y-1<z<=y.
    upper=y<40; lower=y>0
    a=np.concatenate([np.column_stack([x[upper],np.ones(upper.sum())]),
                      np.column_stack([-x[lower],np.ones(lower.sum())])])
    b=np.concatenate([y[upper],-(y[lower]-1)])
    result=linprog([0.,0.,0.,-1.],A_ub=a,b_ub=b,
                   bounds=[(-20,20),(-20,20),(-2,2),(0,.49)],method='highs',
                   options={'time_limit':60.})
    if not result.success:
        raise RuntimeError('bounded interval fit failed: '+result.message)
    violation=float(np.max(a@result.x-b))
    if violation>1e-6:
        raise RuntimeError('fitted constraints violate numerical tolerance')
    return dict(coefficients=result.x[:3].tolist(),margin=float(result.x[3]),
                max_constraint_violation=violation,labels=len(y),solver_status=result.message)


class InventoryPositionLinearPolicy:
    variant='inventory_position_interval_imitation_v1'

    def __init__(self,n,pipe_horizon,coefficients):
        self.n,self.pipe_horizon=int(n),int(pipe_horizon)
        self.coefficients=np.asarray(coefficients,float)
        if self.coefficients.shape!=(3,) or not np.isfinite(self.coefficients).all():
            raise ValueError('three finite coefficients required')

    def predict(self,obs):
        score=position_features(obs,self.n,self.pipe_horizon)@self.coefficients
        return np.ceil(np.maximum(score,0)).clip(0,40).astype(np.int64)

    def act(self,obs,greedy=True):
        if not greedy:
            raise ValueError('deterministic diagnostic; no stochastic PPO distribution')
        return self.predict(obs),0.

    def save(self,path):
        Path(path).write_text(json.dumps(dict(variant=self.variant,n=self.n,pipe_horizon=self.pipe_horizon,
            coefficients=self.coefficients.tolist(),grid_step=5,grid_max=200,fixed_price_multiplier=1.2),indent=2))

    @classmethod
    def load(cls,path,n,pipe_horizon):
        data=json.loads(Path(path).read_text())
        if (data.get('variant')!=cls.variant or data.get('n')!=n or data.get('pipe_horizon')!=pipe_horizon
            or data.get('grid_step')!=5 or data.get('grid_max')!=200 or data.get('fixed_price_multiplier')!=1.2):
            raise ValueError('matching structured imitation schema required')
        return cls(n,pipe_horizon,data['coefficients'])
