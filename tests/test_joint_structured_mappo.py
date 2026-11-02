import copy
import numpy as np
import torch
from src.algos.structured_mappo import StructuredMAPPO
from src.algos.parameter_policy_gradient import order_action
from src.env.joint_env import JointRetailEnv


def setup(mode='central'):
    torch.set_num_threads(1)
    env=JointRetailEnv(episode_length_days=28,pricing_scope='per_sku')
    covers=np.linspace(4,10,10,dtype=np.float32)
    return env,covers,StructuredMAPPO(env,covers,71,critic_mode=mode)


def test_exact_greedy_anchor_without_central_state_access():
    env,covers,model=setup()
    def forbidden():raise AssertionError('actor leaked central state')
    env.get_state=forbidden
    episode=model.collect(env,8,greedy=True,training=False)
    for row in episode['trace']:
        assert np.array_equal(row['actions'],order_action(row['obs'],row['day'],covers))
        if row['price_actions'] is not None:assert np.all(row['price_actions']==4)


def test_joint_update_changes_both_actors_and_critic_and_roundtrips():
    env,covers,model=setup()
    before=copy.deepcopy(model.state_dict())
    episodes=[model.collect(env,k) for k in (10,11,12,13)]
    assert len(episodes[0]['rows']['pricing'])==4 and len(episodes[0]['rows']['ordering'])==28
    metrics=model.update(episodes)
    assert all(np.isfinite(v) for v in metrics.values())
    for prefix in ('pricing.','ordering.','critics.'):
        assert any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith(prefix))
    state=copy.deepcopy(model.checkpoint())
    expected=model.collect(env,42,greedy=True,training=False)['rewards']
    other=StructuredMAPPO(env,covers,90);other.restore(state)
    assert expected==other.collect(env,42,greedy=True,training=False)['rewards']


def test_local_control_never_reads_global_state_and_freezes_other_actor():
    env,_,model=setup('local')
    def forbidden():raise AssertionError('local critic leaked central state')
    env.get_state=forbidden
    before=copy.deepcopy(model.pricing.state_dict())
    batch=[model.collect(env,k,frozen=('pricing',)) for k in (18,19)]
    metrics=model.update(batch)
    assert metrics['pricing_actor_steps']==0
    assert all(torch.equal(v,model.pricing.state_dict()[k]) for k,v in before.items())
