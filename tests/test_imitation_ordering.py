"""Physical support and checkpoint safeguards for explicitly assisted ordering."""
import numpy as np
import pytest
import torch
from src.algos.mappo import Actor
from src.algos.imitation_ordering import teacher_indices,TeacherResidualActor,ImitationAssistedMAPPO
from src.env.joint_env import JointRetailEnv


def test_teacher_uses_pipeline_and_inventory_without_future_information():
    # One SKU, two pipeline buckets, final element is deterministic forecast.
    obs=torch.tensor([[10.,15.,5.,0.,10.],[60.,0.,0.,0.,10.]])
    assert teacher_indices(obs,1,2).tolist()==[[6],[0]]


def test_residual_boundary_duplicates_have_one_physical_probability_and_gradient():
    actor=Actor(5,[41],8)
    actor.__class__=TeacherResidualActor
    actor.n,actor.pipe_horizon,actor.order_step=1,2,5
    with torch.no_grad():
        actor.heads[0].weight.zero_();actor.heads[0].bias.zero_()
    obs=torch.tensor([[0.,0.,0.,0.,0.],[0.,0.,0.,0.,1000.]])
    dist=actor(obs)[0]
    np.testing.assert_allclose(dist.probs[0,:3].detach(),[1/3]*3,rtol=1e-6)
    assert torch.count_nonzero(dist.probs[0,3:])==0
    np.testing.assert_allclose(dist.probs[1,38:].detach(),[1/3]*3,rtol=1e-6)
    assert torch.count_nonzero(dist.probs[1,:38])==0
    loss=-dist.log_prob(torch.tensor([1,39])).mean();loss.backward()
    assert actor.heads[0].bias.grad.abs().sum()>0
    action=dist.sample()
    assert all(int(a) in allowed for a,allowed in zip(action,[(0,1,2),(38,39,40)]))


def test_explicit_checkpoint_rejects_residual_schema_mismatch(tmp_path):
    torch.set_num_threads(1)
    env=JointRetailEnv(episode_length_days=2)
    original=ImitationAssistedMAPPO(env,normalize_inputs=True)
    path=tmp_path/'checkpoint.pt';original.save(path)
    restored=ImitationAssistedMAPPO(env,normalize_inputs=True);restored.load(path)
    obs,_=env.reset(seed=5)
    np.testing.assert_array_equal(original.replenishment.act(obs['replenishment'],greedy=True)[0],
                                  restored.replenishment.act(obs['replenishment'],greedy=True)[0])
    with pytest.raises(ValueError,match='matching explicit'):
        ImitationAssistedMAPPO(env,residual=True,normalize_inputs=True).load(path)


def test_teacher_runner_rejects_unbounded_budget_before_output(tmp_path):
    from scripts.diagnose_teacher_ordering import run
    with pytest.raises(ValueError,match='bounded'):
        run(tmp_path/'not_created','unused.json',teacher_episodes=65)
    assert not (tmp_path/'not_created').exists()


def test_paired_fill_uncertainty_and_seed_alignment():
    from scripts.diagnose_teacher_ordering import compare
    base=[dict(seed=1,reward_day=10,fill=.6),dict(seed=2,reward_day=10,fill=.6)]
    rows=[dict(seed=1,reward_day=10,fill=.6),dict(seed=2,reward_day=10,fill=.596)]
    result=compare(rows,base)
    assert result['fill_delta']>-.005
    assert result['fill_ci95'][0]<-.005
    with pytest.raises(ValueError,match='paired seeds'):
        compare(rows[::-1],base)


def test_expired_measurement_preserves_no_unbounded_episode():
    from scripts.diagnose_teacher_ordering import measure
    import time
    env=JointRetailEnv(episode_length_days=2)
    with pytest.raises(TimeoutError,match='deadline'):
        measure(env,None,[1],'expired',deadline=time.perf_counter()-1)


def test_vanilla_loader_rejects_labeled_diagnostic_checkpoint_before_mutation(tmp_path):
    from src.algos.mappo import HeterogeneousMAPPO
    env=JointRetailEnv(episode_length_days=2)
    path=tmp_path/'assisted.pt'
    ImitationAssistedMAPPO(env,normalize_inputs=True).save(path)
    vanilla=HeterogeneousMAPPO(env)
    original=[p.detach().clone() for p in vanilla.pricing.parameters()]
    with pytest.raises(ValueError,match='variant'):
        vanilla.load(path)
    assert vanilla.normalize_inputs is False
    assert all(torch.equal(p,b) for p,b in zip(vanilla.pricing.parameters(),original))
