import numpy as np
import torch
from src.algos.temporal_cover_ppo import plan_log_prob,paired_advantages,TemporalCoverPPO,evaluation_plan
from src.algos.parameter_policy_gradient import order_action
from scripts.verify_temporal_cover_ppo import independent_update
from scripts.verify_structured_ordering_ppo import equal_state


def test_repeated_latent_is_counted_only_once():
    mean=torch.zeros(10);z=torch.ones(2,1,10)
    once=plan_log_prob(z,mean);daily=plan_log_prob(z.repeat(1,28,1),mean)
    torch.testing.assert_close(daily,28*once)
    expected=torch.distributions.Normal(mean,.75).log_prob(z[:,0]).sum(-1)
    torch.testing.assert_close(once,expected)


def test_arms_exactly_preserve_greedy_start():
    covers=np.linspace(1,15,10);rng=np.random.default_rng(11)
    for arm in ['episode','daily']:
        plan=evaluation_plan(covers,arm,1,True)
        for day in range(1,29):
            obs=rng.uniform(0,20,260).astype(np.float32)
            np.testing.assert_array_equal(order_action(obs,day,plan[0 if arm=='episode' else day-1]),order_action(obs,day,covers))


def test_antithetic_draws_and_pair_baseline():
    learner=TemporalCoverPPO(np.full(10,7.5),'daily')
    epsilon,plans=learner.plans()
    np.testing.assert_allclose(plans[::2]+plans[1::2],15.,atol=2e-6)
    r=np.arange(16,dtype=float).reshape(8,2)
    a=np.array([r[i,j]-np.mean([r[k,l] for k in range(8) if k!=i for l in range(2)]) for i in range(8) for j in range(2)])
    np.testing.assert_allclose(paired_advantages(r),(a-a.mean())/(a.std()+1e-8),rtol=1e-6)
    metrics=learner.update(plans,r)
    assert metrics['steps']>0 and torch.isfinite(learner.mean).all()


def test_independent_optimizer_reconstruction_both_cadences():
    torch.set_num_threads(1)
    for arm in ['episode','daily']:
        model=TemporalCoverPPO(np.full(10,7.5),arm);_,plans=model.plans()
        mean=torch.nn.Parameter(model.mean.detach().clone());opt=torch.optim.Adam([mean],lr=.03)
        returns=np.arange(16,dtype=float).reshape(8,2)**2
        result=model.update(plans,returns);old,adv,steps=independent_update(mean,opt,plans,returns)
        assert result['old_logp']==old and result['advantage']==adv and result['steps']==steps
        equal_state(model.mean.detach(),mean.detach());equal_state(model.optimizer.state_dict(),opt.state_dict())
