"""Daily replenishment PPO using the co-owned PPO implementation."""
import numpy as np
import torch
from src.algos.ppo import MultiDiscreteActorCritic, PPO, RolloutBuffer


class ReplenishmentPPOAgent:
    def __init__(self, env, hidden=64, rollout_steps=256, epochs_per_update=4,
                 minibatch_size=64, device='cpu', lr=3e-4):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.device = device
        self.rollout_steps = rollout_steps
        self.net = MultiDiscreteActorCritic(self.obs_dim, env.action_space.nvec, hidden)
        self.ppo = PPO(self.net, lr=lr, epochs_per_update=epochs_per_update,
                       minibatch_size=minibatch_size, device=device)

    def act(self, obs, greedy=False):
        action, logp, value = self.ppo.act(obs, greedy)
        # The shared core returns (n,1); the environment requires (n,).
        return np.asarray(action, dtype=np.int64).reshape(-1), logp, value

    def train(self, total_updates, seed=0):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        obs, _ = self.env.reset(seed=seed)
        buffer = RolloutBuffer(self.rollout_steps, self.obs_dim, self.env.n, self.device)
        history = []
        for update in range(total_updates):
            buffer.reset()
            while not buffer.full:
                action, logp, value = self.act(obs)
                next_obs, reward, term, trunc, _ = self.env.step(action)
                buffer.add(obs, action, logp, value, reward, term or trunc)
                obs = next_obs
                if term or trunc:
                    obs, _ = self.env.reset(seed=int(rng.integers(0, 2**31)))
            _, _, value = self.act(obs, greedy=True)
            metrics = self.ppo.update(buffer, value)
            metrics['update'] = update
            history.append(metrics)
        return history

    def save(self, path):
        self.ppo.save(path)

    def load(self, path):
        self.ppo.load(path)
