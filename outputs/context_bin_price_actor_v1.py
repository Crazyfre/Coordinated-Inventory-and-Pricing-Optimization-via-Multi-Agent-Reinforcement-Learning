"""Uninstalled local context-table categorical actor; no training on import."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from outputs.full_price_actor_v1 import FullPriceActor
from outputs.economic_proxy_rule_v2 import scores

COVER_EDGES=(.25,.5,1.,2.)
SCORE_EDGES=(-128.,-16.,-2.,-.25,0.,.25,2.,16.)


def context_indices(day,cover,delta):
    cover=np.asarray(cover,float);delta=np.asarray(delta,float)
    if type(day) is not int or not 1<=day<=28 or cover.shape!=(10,) or delta.shape!=(10,4):
        raise ValueError('Episode day1..28/ten cover values/four score differences required')
    if not np.isfinite(cover).all() or not np.isfinite(delta).all() or np.any((cover<0)|(cover>3)):
        raise ValueError('Finite clipped cover[0,3] and score differences required')
    cb=np.searchsorted(COVER_EDGES,cover,side='right')
    sb=np.searchsorted(SCORE_EDGES,delta,side='right')
    result=(((day-1)//7)*5+cb[:,None])*9+sb
    assert result.shape==(10,4) and np.all((result>=0)&(result<180))
    return result.astype(np.float32)


class ContextBinPriceActor(FullPriceActor):
    schema='local_week_cover_proxy_bin_shared_price_logits_v1'
    context_shape=(4,5,9)

    def __init__(self,base_prices,phi):
        super().__init__(base_prices,phi)
        self.weight=nn.Parameter(torch.zeros(180,4))
        self.register_buffer('log_prior',torch.log(torch.tensor(self.prior)))
        self.register_buffer('logit_gain',torch.tensor(8.))

    def features(self,observation,day):
        x=np.asarray(observation,np.float32)
        if x.ndim!=1 or x.size<180 or not np.isfinite(x).all():
            raise ValueError('Complete finite actor-local pricing observation required')
        value=scores(x,day,self.base_prices.detach().cpu().numpy(),self.phi.detach().cpu().numpy())
        cover=np.clip(x[:10]/(1+7*x[-20:-10]),0,3)
        return context_indices(day,cover,value[:,:4]-value[:,4:5])

    def distribution(self,features):
        if not isinstance(features,torch.Tensor) or features.device!=self.weight.device:
            raise ValueError('Feature tensor on actor device required')
        if features.shape[-2:]!=(10,4) or not torch.isfinite(features).all():
            raise ValueError('Finite ten-by-four context indices required')
        if (features!=features.floor()).any() or (features<0).any() or (features>=180).any():
            raise ValueError('Integral context indices0..179 required')
        index=features.long();action=torch.arange(4,device=self.weight.device)
        residual=(self.logit_gain*self.weight[index,action]).clamp(-6,6)
        logits=torch.cat((residual,torch.zeros(residual.shape[:-1]+(1,),dtype=residual.dtype,device=residual.device)),dim=-1)
        return Categorical(logits=(logits+self.log_prior)/self.temperature)
