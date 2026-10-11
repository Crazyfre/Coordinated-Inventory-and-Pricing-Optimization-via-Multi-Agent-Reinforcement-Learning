import numpy as np


def test_shared_multidiscrete_action_has_vector_shape():
    from src.algos.ppo import MultiDiscreteActorCritic, PPO
    action, _, _ = PPO(MultiDiscreteActorCritic(3, [4, 5])).act(np.zeros(3))
    assert action.shape == (2,)

def test_r3_cli_creates_training_and_evaluation_artifacts():
    import subprocess, sys, json
    from pathlib import Path
    result = subprocess.run([sys.executable, '-m', 'scripts.train_r3', '--updates','1',
                             '--days','9','--episodes','1','--output','outputs/tdd_r3'],
                            capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads(Path('outputs/tdd_r3/summary.json').read_text())
    assert set(summary) == {'ppo','zero','constant','ss'}
    assert Path('outputs/tdd_r3/checkpoint.pt').exists()


import torch
from src.env.replenishment_env import ReplenishmentEnv


def test_shared_value_clipping_uses_rollout_value_not_target():
    from src.algos.ppo import DiscreteActorCritic, PPO, RolloutBuffer
    net = DiscreteActorCritic(1,2,hidden=4)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.v.bias.fill_(1.)
    buffer = RolloutBuffer(1,1)
    buffer.add([0],0,-np.log(2),0,2,True)
    metrics = PPO(net,epochs_per_update=1).update(buffer,0)
    assert np.isclose(metrics['v_loss'],1.62,atol=1e-5)



def test_daily_ppo_trains_vector_actions():
    from src.agents.replenishment_agent import ReplenishmentPPOAgent
    torch.set_num_threads(1)
    env = ReplenishmentEnv(episode_length_days=9)
    agent = ReplenishmentPPOAgent(env, rollout_steps=12, epochs_per_update=1)
    before = [p.detach().clone() for p in agent.net.parameters()]
    history = agent.train(2, seed=7)
    obs, _ = env.reset(seed=7)
    action, _, _ = agent.act(obs, greedy=True)
    assert action.shape == (env.n,)
    assert env.action_space.contains(action)
    assert all(np.isfinite(h['v_loss']) for h in history)
    assert any(not torch.equal(a,b) for a,b in zip(before,agent.net.parameters()))
