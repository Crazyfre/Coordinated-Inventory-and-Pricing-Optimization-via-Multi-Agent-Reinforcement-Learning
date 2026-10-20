import numpy as np
import torch


def test_fixed_fixture_and_known_policy():
    from scripts.diagnose_learning import make_fixture, evaluate
    env = make_fixture()
    a = evaluate(env, lambda obs: np.array([1]))
    b = evaluate(env, lambda obs: np.array([0]))
    assert abs(a['daily_reward'] - (100. - .1 + 26 * 20.) / 28) < 1e-6
    assert a['fill'] == 1.
    assert a['daily_reward'] > b['daily_reward']


def test_diagnostic_seed_can_start_from_bad_policy():
    from scripts.diagnose_learning import run
    result = run(updates=0, seed=1)
    assert result['initial']['fill'] < .2


def test_reduced_fixture_learns_from_bad_initial_policy():
    from scripts.diagnose_learning import run
    result = run(updates=128, seed=1)
    assert result['initial']['daily_reward'] < 0
    assert result['passed'], result['final']


def test_receiving_clip_makes_large_orders_equivalent():
    from scripts.diagnose_learning import make_fixture, evaluate
    env = make_fixture(q_max=200)
    outcomes = [evaluate(env, lambda obs, a=a: np.array([a])) for a in range(6, 41)]
    assert all(x['daily_reward'] == outcomes[0]['daily_reward'] for x in outcomes)
    assert outcomes[0]['daily_reward'] < evaluate(env, lambda obs: np.array([1]))['daily_reward']


def test_gae_bootstraps_cuts_but_not_episode_boundaries():
    from src.algos.ppo import compute_gae
    _, cut = compute_gae(np.array([1.]), np.array([0.]), np.array([0.]), 10., gamma=.5)
    _, terminal = compute_gae(np.array([1.]), np.array([0.]), np.array([1.]), 10., gamma=.5)
    assert cut[0] == 6.
    assert terminal[0] == 1.


def test_action_logprob_roundtrip_and_gradient():
    from src.algos.ppo import MultiDiscreteActorCritic, PPO
    torch.set_num_threads(1)
    torch.manual_seed(11)
    ppo = PPO(MultiDiscreteActorCritic(3, [3, 4], hidden=8))
    obs = np.array([1, 2, 3], dtype=np.float32)
    action, lp, _ = ppo.act(obs)
    newlp, _, _ = ppo.evaluate(torch.tensor(obs[None]), torch.tensor(action[None]))
    assert abs(newlp.item() - lp) < 1e-6
    (-newlp).backward()
    assert sum(p.grad.abs().sum().item() for p in ppo.net.logits_heads.parameters()) > 0
