import copy
import numpy as np
import pytest
import torch
from src.algos.pipeline_structured_mappo import PipelineMAPPO,PipelineOrderingActor,pipeline_features,MODES
from src.algos.action_marginal_mappo import ActionMarginalMAPPO
from src.env.joint_env import JointRetailEnv


def test_timing_alias_resolution_and_exact_mass_at_nonzero_weights():
    torch.set_num_threads(1)
    x=np.zeros(260,np.float32);x[-10:]=3;x[10]=40;y=x.copy();y[10]=0;y[12]=40
    assert np.array_equal(pipeline_features(x,8,'current'),pipeline_features(y,8,'current'))
    assert np.array_equal(pipeline_features(x,8,'collapsed'),pipeline_features(y,8,'collapsed'))
    assert not np.array_equal(pipeline_features(x,8,'timing'),pipeline_features(y,8,'timing'))
    a=PipelineOrderingActor(np.full(10,8.),'timing')
    with torch.no_grad():a.weight[:,7]=1
    def density(obs):
        p=obs[:10]+obs[10:80].reshape(10,7).sum(1)
        return torch.tensor(np.c_[pipeline_features(obs,8,'timing'),p,obs[-10:]])
    f,g=density(x),density(y)
    assert not torch.equal(a.distribution(f).mean,a.distribution(g).mean)
    masses=a.grid_log_masses(f)
    assert torch.allclose(masses.logsumexp(-1),torch.zeros(10),atol=1e-6,rtol=0)
    # Independent Normal CDF calculation at the zero-order threshold.
    expected=a.distribution(f).cdf(f[:,-2]/f[:,-1])
    assert torch.allclose(masses[:,0].exp(),expected,atol=1e-6,rtol=0)
    a.stats(f,torch.zeros(10))[0].backward()
    assert a.weight.grad[0,7]!=0


def test_warmstart_current_equivalence_training_checkpoint_and_local_execution():
    torch.set_num_threads(1);env=JointRetailEnv(episode_length_days=28,pricing_scope='per_sku')
    covers=np.linspace(4,10,10);old=ActionMarginalMAPPO(env,covers,98)
    models={m:PipelineMAPPO(env,covers,98,representation=m) for m in MODES}
    assert sum(p.numel() for p in models['timing'].ordering.parameters())==sum(p.numel() for p in models['collapsed'].ordering.parameters())
    torch.manual_seed(71235);reference=old.collect(env,84)
    episodes={}
    for mode,model in models.items():
        torch.manual_seed(71235);ep=model.collect(env,84);episodes[mode]=ep
        assert ep['rewards']==reference['rewards']
        for a,b in zip(ep['trace'],reference['trace']):
            assert np.array_equal(a['actions'],b['actions']) and np.array_equal(a['price_actions'],b['price_actions'])
        assert all(a['logp']==b['logp'] for a,b in zip(ep['rows']['ordering'],reference['rows']['ordering']))
    old_metrics=old.update([reference]);current_metrics=models['current'].update([episodes['current']])
    assert old_metrics==current_metrics
    assert all(torch.equal(v,models['current'].state_dict()[k]) for k,v in old.state_dict().items())
    model=models['timing'];before={r:getattr(model,r).weight.detach().clone() for r in ('pricing','ordering')}
    model.update([episodes['timing']]);assert all(not torch.equal(before[r],getattr(model,r).weight) for r in before)
    assert torch.count_nonzero(model.ordering.weight[:,7:])>0
    checkpoint=copy.deepcopy(model.checkpoint());restored=PipelineMAPPO(env,covers,99);restored.restore(checkpoint)
    with pytest.raises(ValueError,match='representation'):models['collapsed'].restore(checkpoint)
    with pytest.raises(ValueError,match='pipeline'):restored.restore(old.checkpoint())
    env.get_state=lambda:(_ for _ in ()).throw(AssertionError('global execution access'))
    assert model.collect(env,86,greedy=True,training=False)['rewards']==restored.collect(env,86,greedy=True,training=False)['rewards']
