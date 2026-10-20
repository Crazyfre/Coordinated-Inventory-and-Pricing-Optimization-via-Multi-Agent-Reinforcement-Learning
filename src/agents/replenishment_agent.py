"""Daily replenishment PPO using the co-owned PPO implementation."""
import numpy as np
import torch
from src.algos.ppo import MultiDiscreteActorCritic, PPO, RolloutBuffer
from src.algos.stability import SignedLog


class ReplenishmentPPOAgent:
    def __init__(self, env, hidden=64, rollout_steps=256, epochs_per_update=4,
                 minibatch_size=64, device='cpu', lr=3e-4,
                 normalize_inputs=False, reward_scale=1.):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.device = device
        self.rollout_steps = rollout_steps
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError('reward_scale must be positive')
        self.reward_scale = reward_scale
        self.normalize_inputs = normalize_inputs
        self.net = MultiDiscreteActorCritic(self.obs_dim, env.action_space.nvec, hidden)
        if normalize_inputs:
            self.net = torch.nn.Sequential(SignedLog(), self.net)
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
                buffer.add(obs, action, logp, value, reward/self.reward_scale, term or trunc)
                obs = next_obs
                if term or trunc:
                    obs, _ = self.env.reset(seed=int(rng.integers(0, 2**31)))
            _, _, value = self.act(obs, greedy=True)
            metrics = self.ppo.update(buffer, value)
            metrics['update'] = update
            history.append(metrics)
        return history

    def save(self, path):
        torch.save({'net':self.net.state_dict(),'optimizer':self.ppo.optimizer.state_dict(),
                    'normalize_inputs':self.normalize_inputs,'reward_scale':self.reward_scale},path)

    def load(self, path):
        data = torch.load(path,map_location=self.device,weights_only=True)
        normalized = data.get('normalize_inputs',any(k.startswith('1.body.') for k in data['net']))
        if normalized != self.normalize_inputs:
            self.net = torch.nn.Sequential(SignedLog(),self.net) if normalized else self.net[1]
            self.ppo.net = self.net
            self.normalize_inputs = normalized
        self.net.load_state_dict(data['net'])
        self.ppo.optimizer.load_state_dict(data['optimizer'])
        # For metadata-free checkpoints, caller's scale remains authoritative.
        self.reward_scale = data.get('reward_scale',self.reward_scale)
