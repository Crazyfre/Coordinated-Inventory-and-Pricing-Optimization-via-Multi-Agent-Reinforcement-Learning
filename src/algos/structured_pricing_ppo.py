"""Compact weekly PPO pricing actor with frozen structured replenishment.

This is a stage-one component, not joint MAPPO. Actor features contain only the
pricing observation and known episode clock. The critic uses centralized state.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from src.algos.mappo import interval_gae


def pricing_features(obs, day, base_prices):
    x = np.asarray(obs, np.float32)
    base = np.asarray(base_prices, np.float32)
    if x.ndim != 1 or x.size < 180 or base.shape != (10,) or np.any(base <= 0):
        raise ValueError("pricing observation and ten positive base prices required")
    if not 1 <= day <= 28:
        raise ValueError("day outside episode")
    forecast = x[-20:-10]
    history = x[10:150].reshape(10, 14)
    return np.stack([
        np.ones(10),
        np.clip(x[:10] / (1 + 7 * forecast), 0, 3),
        np.log1p(np.maximum(forecast, 0)) / 5,
        np.clip(history[:, -7:].mean(1) / (1 + forecast), 0, 3),
        x[150:160] / base - 1,
        np.full(10, (29 - day) / 28),
        np.clip(x[-10:] / 500, 0, 1),
    ], axis=-1).astype(np.float32)


class StructuredPricingActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(10, 7, 5))
        with torch.no_grad():
            self.weight[:, 0, :] = torch.log(torch.tensor([.05, .05, .05, .05, .8]))

    def distribution(self, features):
        return Categorical(logits=torch.einsum("...if,ifk->...ik", features, self.weight))

    def stats(self, features, actions):
        dist = self.distribution(features)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)

    @torch.no_grad()
    def act(self, features, greedy=False):
        dist = self.distribution(torch.as_tensor(features))
        action = dist.probs.argmax(-1) if greedy else dist.sample()
        return action.numpy(), float(dist.log_prob(action).sum())


def critic_features(state, day):
    x = np.asarray(state, np.float32)
    return np.r_[np.sign(x) * np.log1p(np.abs(x)) / 5, (29 - day) / 28].astype(np.float32)


class StructuredPricingPPO:
    def __init__(self, state_dim, seed):
        torch.manual_seed(seed)
        self.actor = StructuredPricingActor()
        self.critic = nn.Sequential(nn.Linear(state_dim + 1, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=.01)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=.001)

    @torch.no_grad()
    def value(self, state):
        return float(self.critic(torch.as_tensor(state)).squeeze())

    def state(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    actor_optimizer=self.actor_optimizer.state_dict(),
                    critic_optimizer=self.critic_optimizer.state_dict(),
                    rng=torch.get_rng_state())

    def update(self, episodes):
        features, actions, old_logs, states, advantages, returns = [], [], [], [], [], []
        for episode in episodes:
            rows = episode["decisions"]
            adv, ret = interval_gae(np.asarray(episode["rewards"]) / 100,
                                    [r["value"] for r in rows], [0, 7, 14, 21], .99, .95)
            features.extend([r["features"] for r in rows])
            actions.extend([r["action"] for r in rows])
            old_logs.extend([r["logp"] for r in rows])
            states.extend([r["state"] for r in rows])
            advantages.extend(adv)
            returns.extend(ret)
        f = torch.tensor(np.array(features), dtype=torch.float32)
        a = torch.tensor(np.array(actions), dtype=torch.long)
        old = torch.tensor(old_logs, dtype=torch.float32)
        adv = torch.tensor(advantages, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        steps = 0
        for _ in range(4):
            lp, _ = self.actor.stats(f, a)
            ratio = (lp - old).exp()
            kl = ((ratio - 1) - (lp - old)).mean()
            if float(kl.detach()) > .02:
                break
            loss = -torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv).mean()
            self.actor_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
            self.actor_optimizer.step()
            steps += 1
        state = torch.tensor(np.array(states), dtype=torch.float32)
        target = torch.tensor(returns, dtype=torch.float32)
        for _ in range(4):
            prediction = self.critic(state).squeeze(-1)
            value_loss = .5 * (prediction - target).square().mean()
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), .5)
            self.critic_optimizer.step()
        with torch.no_grad():
            lp, ent = self.actor.stats(f, a)
            ratio = (lp - old).exp()
        return dict(actor_steps=steps, actor_loss=float(loss.detach()),
                    critic_loss=float(value_loss.detach()),
                    kl=float(((ratio - 1) - (lp - old)).mean()), entropy=float(ent.mean()))
