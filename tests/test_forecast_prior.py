import importlib
import numpy as np
import torch
from src.algos.mappo import Actor


def test_opt_in_checkpoint_records_prior_and_roundtrips(tmp_path):
    import src.algos.forecast_prior as module
    assert hasattr(module, 'ForecastPriorMAPPO'), 'opt-in MAPPO adapter is missing'
    from src.experiments.phase4 import make_env
    env = make_env(7,500,joint=True)
    model = module.ForecastPriorMAPPO(env,hidden=8,normalize_inputs=True)
    obs,_ = env.reset(seed=3)
    env.select_price(4)
    staged = env.replenishment_observation()
    assert not np.array_equal(staged[-env.n:],obs['replenishment'][-env.n:])
    b = env.observation_components()
    np.testing.assert_allclose(staged[:env.n],b['inventory'])
    np.testing.assert_allclose(staged[env.n:env.n*(1+env.obs_pipe_horizon)],b['pipeline'].ravel())
    action = model.replenishment.act(staged,True)[0]
    assert env.action_space.contains(action)
    path = tmp_path/'prior.pt'
    model.save(path)
    data = torch.load(path,weights_only=True)
    assert data['variant'] == 'forecast_prior_mappo'
    restored = module.ForecastPriorMAPPO(env,hidden=8,normalize_inputs=True)
    restored.load(path)
    np.testing.assert_array_equal(action,restored.replenishment.act(staged,True)[0])
    vanilla = module.HeterogeneousMAPPO(env,hidden=8,normalize_inputs=True)
    assert isinstance(vanilla.replenishment,Actor)
    assert not isinstance(vanilla.replenishment,module.ForecastPriorActor)



def test_clean_correction_runner_exists():
    assert importlib.util.find_spec('scripts.evaluate_learning_correction') is not None, 'paired clean runner missing'


def test_prior_uses_inventory_position_and_preserves_all_41_actions():
    spec = importlib.util.find_spec('src.algos.forecast_prior')
    assert spec is not None, 'opt-in forecast prior is missing'
    from src.algos.forecast_prior import ForecastPriorActor
    torch.set_num_threads(1)
    actor = ForecastPriorActor(12, [41], 8, n=1, pipe_horizon=2,
                               order_step=5, normalize_inputs=True)
    for p in actor.parameters():
        p.data.zero_()
    obs = torch.zeros(1,12)
    obs[0,0] = 10
    obs[0,1:3] = torch.tensor([10.,10.])
    obs[0,-1] = 10
    dist = actor(obs)[0]
    assert dist.probs.shape == (1,41)
    assert torch.all(dist.probs > 0)
    assert int(dist.probs.argmax()) == 6  # ceil((6*10 - (10+20))/5)
    obs[0,1:3] = 100
    assert int(actor(obs)[0].probs.argmax()) == 0
    obs[0,1:3] = 10
    obs[0,-1] = 5  # staged-price forecast, not stale named components
    assert int(actor(obs)[0].probs.argmax()) == 0
    actions = torch.tensor([[6]])
    lp, _ = actor.stats(obs,actions)
    assert torch.allclose(lp,actor(obs)[0].log_prob(actions[:,0]))
    (-lp.mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in actor.parameters())
