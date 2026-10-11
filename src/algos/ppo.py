"""Shared PPO core (Phase 0 co-owned module, first exercised by P3).

Clipped-objective PPO (Schulman et al. 2017) with GAE(lambda) advantages —
deliberately minimal, torch-only, no external RL dependency. Both tracks
build their actors on THIS module (agents.md decision 1):

    - replenishment (R3/MAPPO): MultiDiscrete action head, one Categorical
      over the order grid per SKU.
    - pricing (P3):            plain Discrete action head over the 5 locked
      multipliers.

Signatures agreed for co-ownership (do not change without both tracks):

    class ActorCritic(nn.Module): forward(obs) -> (action_logits | dists, value)
    PPO(net, ...).act(obs, greedy=False) -> (action, logp, value)
    PPO.update(rollouts, next_value) -> dict of loss metrics
    RolloutBuffer(obs_dim, act_dim, size)  # .add/.finish/.iter_minibatches

Design rules baked in:
    - Advantage normalization per update (stability);
    - value clipping + value-loss coefficient (MAPPO-compatible defaults);
    - entropy bonus (explores the 5-price grid early);
    - gradient clipping by global norm;
    - NO parameter-sharing assumptions — the actor is whatever the track
      supplies; the Phase-4 centralized critic is fed GLOBAL state by its
      training loop (separate concern, NOT inside this file).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Networks: joint actor-critic heads for Discrete / MultiDiscrete actions
# ---------------------------------------------------------------------------


def mlp(sizes: list[int], activation: str = "tanh") -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh() if activation == "tanh" else nn.ReLU())
    return nn.Sequential(*layers)


def orthogonal_init(module: nn.Module, gain: float = np.sqrt(2.0)) -> None:
    """Orthogonal init for Linear layers (PPO best practice)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.zeros_(module.bias)


class DiscreteActorCritic(nn.Module):
    """Actor + value head for a Discrete action space (PricingEnv: 5 multipliers)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.body = mlp([obs_dim, hidden, hidden])
        self.logits = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        self.apply(orthogonal_init)
        # Small policy-logits gain: near-uniform start over the price grid.
        nn.init.orthogonal_(self.logits.weight, gain=0.01)
        nn.init.zeros_(self.logits.bias)

    def forward(self, obs: torch.Tensor):
        h = torch.tanh(self.body(obs))
        return self.logits(h), self.v(h).squeeze(-1)


class MultiDiscreteActorCritic(nn.Module):
    """Actor + value head for MultiDiscrete (one head per SKU's order grid)."""

    def __init__(self, obs_dim: int, nvec: list[int] | np.ndarray, hidden: int = 64):
        super().__init__()
        self.nvec = list(map(int, nvec))
        self.body = mlp([obs_dim, hidden, hidden])
        self.logits_heads = nn.ModuleList(
            [nn.Linear(hidden, nv) for nv in self.nvec]
        )
        self.v = nn.Linear(hidden, 1)
        self.apply(orthogonal_init)
        for head in self.logits_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, obs: torch.Tensor):
        h = torch.tanh(self.body(obs))
        dists = [torch.distributions.Categorical(logits=head(h))
                 for head in self.logits_heads]
        return dists, self.v(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """On-policy storage for one PPO update window."""

    def __init__(self, size: int, obs_dim: int, act_dim: int = 1,
                 device: str = "cpu"):
        self.size = size
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.int64)
        self.logp = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.device = device

    def add(self, obs, action, logp, value, reward, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = np.atleast_1d(np.asarray(action))
        self.logp[i] = logp
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.ptr += 1

    @property
    def full(self) -> bool:
        return self.ptr >= self.size

    def compute_advantages(
        self, next_value: float, gamma: float = 0.99, lam: float = 0.95,
        normalize: bool = True,
    ) -> None:
        """GAE(lambda) over the buffer, treating it as ONE contiguous traj."""
        adv, returns = compute_gae(
            self.rewards[:self.ptr], self.values[:self.ptr],
            self.dones[:self.ptr], next_value, gamma=gamma, lam=lam,
        )
        self.advantages[:self.ptr] = adv
        self.returns[:self.ptr] = returns
        if normalize:
            a = self.advantages[:self.ptr]
            self.advantages[:self.ptr] = (a - a.mean()) / (a.std() + 1e-8)

    def iter_minibatches(self, minibatch_size: int, seed: int = 0, include_values: bool = False):
        n = self.ptr
        idx = np.random.default_rng(seed).permutation(n)
        for start in range(0, n, minibatch_size):
            mb = idx[start:start + minibatch_size]
            to = lambda arr, dtype: torch.as_tensor(
                arr[mb], dtype=dtype, device=self.device
            )  # noqa: E731
            batch = (
                to(self.obs, torch.float32),
                to(self.actions, torch.int64),
                to(self.logp, torch.float32),
                to(self.advantages, torch.float32),
                to(self.returns, torch.float32),
            )
            yield batch + (to(self.values, torch.float32),) if include_values else batch

    def reset(self) -> None:
        self.ptr = 0


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
    next_value: float, gamma: float = 0.99, lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE(lambda) for a (possibly episode-segmented) trajectory."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        non_terminal = 1.0 - dones[t]
        next_v = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        adv[t] = last_gae
    return adv, adv + np.asarray(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------------


class PPO:
    """Clipped-objective PPO update machinery around a joint actor-critic."""

    def __init__(
        self,
        net: nn.Module,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        grad_norm: float = 0.5,
        epochs_per_update: int = 4,
        minibatch_size: int = 64,
        device: str = "cpu",
    ):
        self.net = net
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_norm = grad_norm
        self.epochs_per_update = epochs_per_update
        self.minibatch_size = minibatch_size
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        self.device = device
        self.net.to(device)

    # ------------------------------------------------------------------ acting

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False):
        """Returns (action, logp, value) as numpy / python floats.

        Discrete net: action int. MultiDiscrete net: action array (n_heads,).
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        out, value = self.net(obs_t)

        if isinstance(out, list):  # MultiDiscrete: list of Categoricals
            if greedy:
                action = torch.stack([p.probs.argmax(dim=-1) for p in out])
            else:
                action = torch.stack([p.sample() for p in out])
            logp = sum(float(p.log_prob(action[i]))
                       for i, p in enumerate(out))
            return action.squeeze(-1).cpu().numpy(), float(logp), float(value)

        # Discrete: out = logits
        dist = torch.distributions.Categorical(logits=out)
        if greedy:
            action = out.argmax(dim=-1)
        else:
            action = dist.sample()
        return int(action.item()), float(dist.log_prob(action)), float(value)

    # ------------------------------------------------------------------ update

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Recompute logp/value/entropy for minibatch updates."""
        out, values = self.net(obs)
        if isinstance(out, list):  # MultiDiscrete
            logps, entropies = [], []
            for i, dist in enumerate(out):
                logps.append(dist.log_prob(actions[:, i]))
                entropies.append(dist.entropy())
            logp = torch.stack(logps, dim=1).sum(dim=1)
            entropy = torch.stack(entropies, dim=1).sum(dim=1)
        else:  # Discrete
            dist = torch.distributions.Categorical(logits=out)
            logp = dist.log_prob(actions.squeeze(-1))
            entropy = dist.entropy()
        return logp, values, entropy

    def update(self, buffer: RolloutBuffer, next_value: float) -> dict:
        """One PPO update over the buffer. Returns loss metrics."""
        cfg = self
        buf = buffer
        if buf.ptr == 0:
            raise ValueError("buffer is empty")
        buf.compute_advantages(
            next_value, normalize=False
        )  # normalization inside minibatches breaks nothing, do it here:
        a = buf.advantages[:buf.ptr]
        buf.advantages[:buf.ptr] = (a - a.mean()) / (a.std() + 1e-8)

        metrics = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "n_updates": 0}
        for epoch in range(cfg.epochs_per_update):
            for (obs_b, act_b, logp_b, adv_b, ret_b, old_v_b) in buf.iter_minibatches(
                cfg.minibatch_size, seed=epoch, include_values=True
            ):
                logp, values, entropy = self.evaluate(obs_b, act_b)

                ratio = torch.exp(logp - logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
                pi_loss = -torch.min(surr1, surr2).mean()

                v_clipped = old_v_b + torch.clamp(
                    values - old_v_b, -cfg.clip_eps, cfg.clip_eps
                )
                v_loss = 0.5 * torch.max(
                    (values - ret_b) ** 2, (v_clipped - ret_b) ** 2
                ).mean()

                loss = pi_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy.mean()

                cfg.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_norm)
                cfg.optimizer.step()

                m = metrics  # local alias
                m["pi_loss"] += pi_loss.item()
                m["v_loss"] += v_loss.item()
                m["entropy"] += entropy.mean().item()
                m["n_updates"] += 1

        n = max(metrics["n_updates"], 1)
        metrics["pi_loss"] /= n
        metrics["v_loss"] /= n
        metrics["entropy"] /= n
        return metrics

    # ------------------------------------------------------------------ io

    def save(self, path: str) -> None:
        torch.save({
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str, map_location: str = "cpu") -> None:
        ckpt = torch.load(path, map_location=map_location)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
