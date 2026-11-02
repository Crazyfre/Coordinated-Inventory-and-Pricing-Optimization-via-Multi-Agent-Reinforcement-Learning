"""Daily latent Gaussian cover policy; no changes to the successful SPG actor."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from src.algos.structured_pricing_ppo import StructuredPricingPPO
from src.algos.mappo import interval_gae


def ordering_features(obs, day):
    x = np.asarray(obs, np.float32)
    if x.shape != (260,) or not np.isfinite(x).all() or int(day) != day or not 1 <= day <= 28:
        raise ValueError("finite canonical observation and day1..28 required")
    pipe = x[10:80].reshape(10, 7)
    forecast = x[-10:]
    return np.stack([np.ones(10), np.clip(x[:10] / (1 + forecast) / 20, 0, 2),
        np.clip(pipe.sum(1) / (1 + forecast) / 20, 0, 2),
        pipe[:, 3:].sum(1) / (1 + pipe.sum(1)), np.log1p(np.maximum(forecast, 0)) / 5,
        np.full(10, (29 - day) / 28), np.clip(x[-20:-10] / 500, 0, 1)], -1).astype(np.float32)


def execute_cover(obs, latent):
    x, c = np.asarray(obs, np.float32), np.asarray(latent, np.float32)
    if x.shape != (260,) or c.shape != (10,) or not np.isfinite(c).all():
        raise ValueError("canonical observation and finite ten-cover action required")
    position = x[:10] + x[10:80].reshape(10, 7).sum(1)
    return np.ceil(np.maximum(c.clip(0, 20) * x[-10:] - position, 0) / 5).clip(0, 40).astype(np.int64)


class StructuredOrderingActor(nn.Module):
    def __init__(self, covers):
        super().__init__()
        c = torch.as_tensor(covers, dtype=torch.float32)
        if c.shape != (10,) or not torch.isfinite(c).all():
            raise ValueError("ten finite initial covers required")
        self.register_buffer("covers", c.clone())
        self.weight = nn.Parameter(torch.zeros(10, 7))

    def distribution(self, features):
        return Normal(self.covers + (features * self.weight).sum(-1), .75)

    def stats(self, features, latent):
        d = self.distribution(features)
        return d.log_prob(latent).sum(-1), d.entropy().sum(-1)

    @torch.no_grad()
    def act(self, features, greedy=False):
        d = self.distribution(torch.as_tensor(features))
        z = d.mean if greedy else d.sample()
        return z.numpy(), float(d.log_prob(z).sum())


class StructuredOrderingPPO(StructuredPricingPPO):
    def __init__(self, state_dim, covers, seed):
        super().__init__(state_dim, seed)
        self.actor = StructuredOrderingActor(covers)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=.03)

    def update(self, episodes):
        features, actions, logs, states, advantages, returns = [], [], [], [], [], []
        for episode in episodes:
            rows = episode["decisions"]
            assert len(rows) == len(episode["rewards"]) == 28
            adv, ret = interval_gae(np.asarray(episode["rewards"]) / 100,
                [r["value"] for r in rows], list(range(28)), .99, .95)
            features.extend([r["features"] for r in rows])
            actions.extend([r["action"] for r in rows])
            logs.extend([r["logp"] for r in rows])
            states.extend([r["state"] for r in rows])
            advantages.extend(adv); returns.extend(ret)
        x = torch.tensor(np.array(features), dtype=torch.float32)
        a = torch.tensor(np.array(actions), dtype=torch.float32)
        old = torch.tensor(logs, dtype=torch.float32)
        adv = torch.tensor(advantages, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        steps = 0
        for _ in range(4):
            lp, _ = self.actor.stats(x, a)
            ratio = (lp - old).exp()
            if float((((ratio - 1) - (lp - old)).mean()).detach()) > .02:
                break
            loss = -torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv).mean()
            self.actor_optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
            self.actor_optimizer.step(); steps += 1
        state = torch.tensor(np.array(states), dtype=torch.float32)
        target = torch.tensor(returns, dtype=torch.float32)
        for _ in range(4):
            value_loss = .5 * (self.critic(state).squeeze(-1) - target).square().mean()
            self.critic_optimizer.zero_grad(); value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), .5)
            self.critic_optimizer.step()
        with torch.no_grad():
            lp, _ = self.actor.stats(x, a); ratio = (lp - old).exp()
        return dict(actor_steps=steps, actor_loss=float(loss.detach()), critic_loss=float(value_loss.detach()),
                    kl=float(((ratio - 1) - (lp - old)).mean()))
