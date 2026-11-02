"""Finite Fisher pricing step; replaces pricing Adam, no automatic training."""
import copy,math
import torch
from outputs.shared_price_fisher_components_v2 import fisher_blocks,damped_direction,mean_categorical_kl

CONFIG=dict(schema='shared720_price_fisher_armijo_prebatch_kl_v1',damping=.001,kl_limit=.002,backtracks=12,armijo=.1,gradient_clip=.5)

def _step(actor,features,old,loss_fn):
    # Validate full input before mutating any weight/state.
    blocks=fisher_blocks(actor,features)
    if not isinstance(old,torch.Tensor) or old.requires_grad:
        raise ValueError('Detached complete prebatch probabilities required')
    current=actor.distribution(features).probs.detach()
    initial_kl=float(mean_categorical_kl(old,current))
    if initial_kl>CONFIG['kl_limit']+1e-7:raise ValueError('Current policy must be within its prebatch trust region')
    baseline=actor.weight.detach().clone()
    loss=loss_fn()
    if not isinstance(loss,torch.Tensor) or loss.ndim!=0 or not torch.isfinite(loss):
        raise ValueError('Finite scalar differentiable pricing loss required')
    raw,=torch.autograd.grad(loss,actor.weight)
    if not torch.isfinite(raw).all():raise ValueError('Finite pricing gradient required')
    norm=float(torch.linalg.vector_norm(raw));before=float(loss.detach())
    common=dict(loss_before=before,loss_after=before,kl=initial_kl,gradient_norm=norm,scale=0.,trials=0,accepted=False,reason=0)
    if norm==0:return dict(common,reason=1)
    clipped=raw*min(1.,CONFIG['gradient_clip']/(norm+1e-6))
    direction=damped_direction(blocks,-clipped,damping=CONFIG['damping'])
    pred=float((-raw.to(torch.float64)*direction).sum())
    flat=direction.reshape(20,36)
    quadratic=float(.5*torch.einsum('bi,bij,bj->',flat,blocks,flat))
    if not math.isfinite(pred) or pred<=0 or not math.isfinite(quadratic) or quadratic< -1e-8:
        raise FloatingPointError('Finite descent direction and nonnegative Fisher quadratic required')
    scale=min(1.,math.sqrt(CONFIG['kl_limit']/max(quadratic,1e-30)))
    try:
        for trial in range(CONFIG['backtracks']):
            alpha=scale*(.5**trial)
            with torch.no_grad():
                actor.weight.copy_(baseline+alpha*direction.to(baseline.dtype))
                candidate=loss_fn()
                kl=float(mean_categorical_kl(old,actor.distribution(features).probs.detach()))
                value=float(candidate)
            if math.isfinite(value) and math.isfinite(kl) and kl<=CONFIG['kl_limit'] and value<before and not torch.equal(actor.weight,baseline) and value<=before-CONFIG['armijo']*alpha*pred:
                return dict(common,loss_after=value,kl=kl,scale=alpha,trials=trial+1,accepted=True,reason=0)
        with torch.no_grad():actor.weight.copy_(baseline)
        return dict(common,trials=CONFIG['backtracks'],reason=2)
    except BaseException:
        with torch.no_grad():actor.weight.copy_(baseline)
        raise

class PriceFisherState:
    """Explicit checkpointable counts for a stateless finite natural step."""
    def __init__(self):self.attempts=0;self.accepted=0;self.rejected=0;self.trials=0
    def state_dict(self):
        return dict(config=copy.deepcopy(CONFIG),attempts=self.attempts,accepted=self.accepted,rejected=self.rejected,trials=self.trials)
    def load_state_dict(self,state):
        if not isinstance(state,dict) or set(state)!=set(self.state_dict()) or state.get('config')!=CONFIG:
            raise ValueError('Exact pricing Fisher state schema/constants required')
        counts=[state[k] for k in ('attempts','accepted','rejected','trials')]
        if any(type(v) is not int or v<0 for v in counts):raise ValueError('Nonnegative integer natural-step counters required')
        attempts,accepted,rejected,trials=counts
        if attempts>128 or accepted+rejected!=attempts or not accepted<=trials<=12*attempts:
            raise ValueError('Complete bounded natural-step count arithmetic required')
        self.attempts,self.accepted,self.rejected,self.trials=counts
    def step(self,actor,features,old,loss_fn):
        if self.attempts>=128:raise ValueError('At most128 natural attempts per registered fit')
        result=_step(actor,features,old,loss_fn)
        self.attempts+=1;self.accepted+=int(result['accepted']);self.rejected+=int(not result['accepted']);self.trials+=result['trials']
        return result
