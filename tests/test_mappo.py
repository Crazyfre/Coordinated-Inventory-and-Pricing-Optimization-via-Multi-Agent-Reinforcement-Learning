import numpy as np


def test_mappo_checkpoint_roundtrip():
    from src.algos.mappo import HeterogeneousMAPPO
    from src.env.joint_env import JointRetailEnv
    from pathlib import Path
    env = JointRetailEnv(episode_length_days=9)
    agent = HeterogeneousMAPPO(env, hidden=16)
    obs, _ = env.reset(seed=4)
    expected = agent.pricing.act(obs['pricing'],greedy=True)[0]
    path = Path('outputs/tdd_mappo.pt'); path.parent.mkdir(exist_ok=True)
    agent.save(str(path))
    other = HeterogeneousMAPPO(env,hidden=16)
    other.load(str(path))
    assert np.array_equal(expected,other.pricing.act(obs['pricing'],greedy=True)[0])
    assert agent.value(env.get_state()) == other.value(env.get_state())

def test_p4_cli_runs_calibrated_joint_training():
    import subprocess, sys, json
    from pathlib import Path
    result = subprocess.run([sys.executable,'-m','scripts.train_p4','--updates','1',
                             '--days','9','--episodes','1','--output','outputs/tdd_p4'],
                            capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads(Path('outputs/tdd_p4/summary.json').read_text())
    assert 'mappo' in summary
    assert len(summary) == 6
    assert Path('outputs/tdd_p4/checkpoint.pt').exists()


def test_weekly_gae_credits_entire_held_price_interval():
    from src.algos.mappo import interval_gae
    adv, returns = interval_gae(np.ones(9),np.zeros(2),[0,7],gamma=.5,lam=1.)
    expected = [sum(.5**k for k in range(9)),sum(.5**k for k in range(2))]
    assert np.allclose(returns,expected)
    assert np.allclose(adv,expected)


import torch


def test_mappo_updates_separate_actors_and_shared_critic():
    from src.algos.mappo import HeterogeneousMAPPO
    from src.env.joint_env import JointRetailEnv
    torch.set_num_threads(1)
    env = JointRetailEnv(episode_length_days=9)
    agent = HeterogeneousMAPPO(env, hidden=16)
    groups = [agent.pricing, agent.replenishment, agent.critic]
    before = [[p.detach().clone() for p in g.parameters()] for g in groups]
    metrics = agent.train(2, seed=7)
    assert metrics[-1]['pricing_decisions'] == 2
    assert metrics[-1]['daily_steps'] == 9
    assert all(np.isfinite(v) for m in metrics for v in m.values())
    for old, group in zip(before, groups):
        assert any(not torch.equal(a,b) for a,b in zip(old,group.parameters()))
    assert not ({id(p) for p in agent.pricing.parameters()} &
                {id(p) for p in agent.replenishment.parameters()})
