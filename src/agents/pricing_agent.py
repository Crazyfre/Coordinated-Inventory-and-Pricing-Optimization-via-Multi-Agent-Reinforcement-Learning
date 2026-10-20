"""PPO pricing agent (Phase P3, pricing track).

A thin glue layer: weekly PricingEnv (shared Discrete or per-SKU MultiDiscrete)
+ the shared PPO core
(src/algos/ppo.py). No RL logic lives here beyond wiring — the objective,
GAE, clipping and optimization are all in the co-owned algos module so the
replenishment track reuses identical machinery (agents.md decision 1).

Training loop design:
    - rollouts of `rollout_steps` weekly steps across episodes (episodes are
      reset automatically when terminated/truncated);
    - GAE bootstrap from the value of the state after the last rollout step;
    - reward scale: the canonical weekly reward can be large; the agent
      divides it by `reward_scale` (default 1e2 ONLY if user passes it) —
      the DEFAULT is unscaled, matching the canonical reward exactly;
    - greedy eval: argmax actions over full episodes with CRN seeds so
      policy-vs-fixed comparisons share demand realizations (Doc S5.6).

CRN supply seam: the env's demand source is RESEEDABLE via reset(seed);
fixed pricing baselines run under the same episode seeds, giving paired
comparisons (half the variance of unpaired ones).
"""
from __future__ import annotations

import numpy as np
import torch

from src.algos.ppo import DiscreteActorCritic, MultiDiscreteActorCritic, PPO, RolloutBuffer
from src.algos.pricing_schema import pricing_schema, validate_checkpoint_schema


class PricingPPOAgent:
    """Weekly PPO over the environment's explicit shared or per-SKU price grid."""

    def __init__(
        self,
        env,
        obs_dim: int | None = None,
        act_dim: int | None = None,
        hidden: int = 64,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        gamma: float = 0.99,
        lam: float = 0.95,
        rollout_steps: int = 512,
        epochs_per_update: int = 4,
        minibatch_size: int = 64,
        device: str = "cpu",
    ):
        self.env = env
        self.action_schema = pricing_schema(env)
        self.n_price_heads = len(self.action_schema["pricing_nvec"])
        cfg = env.config
        self.n_actions = (
            act_dim if act_dim is not None else len(cfg.pricing.multipliers)
        )
        obs_dim = obs_dim if obs_dim is not None else env.observation_space.shape[0]
        if self.n_actions != len(cfg.pricing.multipliers):
            raise ValueError("pricing act_dim must match locked multiplier grid")
        self.net = (MultiDiscreteActorCritic(
            obs_dim, self.action_schema["pricing_nvec"], hidden=hidden)
            if self.action_schema["pricing_scope"] == "per_sku"
            else DiscreteActorCritic(obs_dim, self.n_actions, hidden=hidden))
        self.ppo = PPO(
            self.net, lr=lr, clip_eps=clip_eps, entropy_coef=entropy_coef,
            epochs_per_update=epochs_per_update,
            minibatch_size=minibatch_size, device=device,
        )
        self.gamma, self.lam = gamma, lam
        self.rollout_steps = rollout_steps
        self.obs_dim, self.device = obs_dim, device

        # Training state.
        self._obs = None
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ acting

    def act(self, obs: np.ndarray, greedy: bool = False) -> tuple[int | np.ndarray, float, float]:
        """(action, logp, value) — greedy picks the argmax multiplier."""
        return self.ppo.act(obs, greedy=greedy)

    # ------------------------------------------------------------------ training

    def train(self, total_updates: int, seed: int = 0, verbose: bool = False) -> list[dict]:
        """Synchronous on-policy training by rollout-update cycles.

        Returns per-update metrics; each update uses `rollout_steps` env steps.
        """
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)  # sampling in act() uses torch's global RNG
        obs, _ = self.env.reset(seed=int(seed))
        buffer = RolloutBuffer(self.rollout_steps, self.obs_dim,
                               act_dim=self.n_price_heads, device=self.device)
        history: list[dict] = []

        for update in range(total_updates):
            buffer.reset()
            while not buffer.full:
                action, logp, value = self.act(obs, greedy=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                buffer.add(obs, action, logp, value, reward, terminated or truncated)
                obs = next_obs
                if terminated or truncated:
                    obs, _ = self.env.reset()

            # Bootstrap from the value of the CURRENT (fresh episode) obs.
            _, _, last_value = self.act(obs, greedy=False)
            metrics = self.ppo.update(buffer, last_value)
            metrics["update"] = update
            history.append(metrics)
            if verbose and (update % 10 == 0 or update == total_updates - 1):
                print(f"[update {update:4d}] pi_loss {metrics['pi_loss']:+.4f} "
                      f"v_loss {metrics['v_loss']:.4f} entropy {metrics['entropy']:.4f}")
        return history

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, episodes: int = 8, seed: int = 123, greedy: bool = True) -> dict:
        """CRN-greedy evaluation: same episode seeds used by baselines.

        Returns mean weekly reward, per-episode rewards and chosen actions.
        """
        ep_rewards: list[float] = []
        actions_taken: list[list] = []
        for ep in range(episodes):
            obs, _ = self.env.reset(seed=seed + ep)
            total, actions = 0.0, []
            while True:
                a, _, _ = self.act(obs, greedy=greedy)
                obs, reward, term, trunc, info = self.env.step(a)
                total += reward
                actions.append(np.asarray(a).tolist())
                if term or trunc:
                    break
            ep_rewards.append(total)
            actions_taken.append(actions)
        return {
            "mean_weekly_reward": float(np.mean(ep_rewards)),
            "episode_rewards": ep_rewards,
            "actions": actions_taken,
        }

    # ------------------------------------------------------------------ io

    def save(self, path: str) -> None:
        torch.save({"net": self.net.state_dict(),
                    "optimizer": self.ppo.optimizer.state_dict(),
                    "action_schema": self.action_schema}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device, weights_only=True)
        validate_checkpoint_schema(data, self.action_schema)
        self.net.load_state_dict(data["net"])
        self.ppo.optimizer.load_state_dict(data["optimizer"])
