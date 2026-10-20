"""Separately labeled, fixed-price compact PPO; no historical class changes."""
import copy
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.algos.imitation_ordering import teacher_indices
from src.algos.ppo import mlp, orthogonal_init, compute_gae
from src.algos.stability import SignedLog


FEATURES = (['intercept', 'own_stock'] + [f'own_arrival_{k}' for k in range(1, 8)]
            + ['forecast', 'price', 'last_lead', 'headroom', 'remaining', 'other_stock']
            + [f'other_arrival_{k}' for k in range(1, 8)] + ['other_forecast']
            + [f'identity_{i}' for i in range(10)])
TRANSFORMS = 'signed_log_own_stock_pipeline_forecast_over5_price_raw;lead/7;capacity/500;remaining/28;identity_v1'


def checked_inputs(obs, days):
    x = torch.as_tensor(obs, dtype=torch.float32)
    d = torch.as_tensor(days)
    if x.ndim != 2 or x.shape[1] != 260 or not torch.isfinite(x).all():
        raise ValueError('finite batch of canonical 10-SKU/7-pipeline/14-history observations required')
    if d.shape != (len(x),) or not torch.isfinite(d).all() or not ((d >= 1) & (d <= 28) & (d == d.long())).all():
        raise ValueError('one integer episode day1..28 per observation required')
    return x, d.long()


def compact_features(obs, days):
    x, d = checked_inputs(obs, days)
    stock, pipe, forecast = x[:, :10], x[:, 10:80].reshape(-1, 10, 7), x[:, -10:]
    log = lambda a: torch.sign(a)*torch.log1p(torch.abs(a))
    scalar = lambda a: a.unsqueeze(-1)
    return torch.cat([
        torch.ones(len(x), 10, 1), scalar(log(stock/5)), log(pipe/5),
        scalar(log(forecast/5)), scalar(log(x[:, -40:-30])), scalar(x[:, -30:-20]/7),
        scalar(x[:, -20:-10]/500), ((29-d.float())/28)[:, None, None].expand(-1, 10, 1),
        scalar((stock.sum(1, keepdim=True)-stock)/500),
        (pipe.sum(1, keepdim=True)-pipe)/500,
        scalar((forecast.sum(1, keepdim=True)-forecast)/500),
        torch.eye(10)[None].expand(len(x), -1, -1)], dim=-1)


class CompactResidualActor(nn.Module):
    def __init__(self, coefficients):
        super().__init__()
        self.register_buffer('coefficients', torch.as_tensor(coefficients, dtype=torch.float64).clone())
        InventoryPositionLinearPolicy(10, 7, coefficients)  # validate
        self.weight = nn.Parameter(torch.zeros(5, 33))

    def support(self, obs):
        # Preserve the established NumPy double dot product after float32 aggregation.
        center = torch.as_tensor(InventoryPositionLinearPolicy(10, 7, self.coefficients.cpu().numpy()).predict(obs))
        target = teacher_indices(obs, 10, 7, cover=7.5)
        default = torch.minimum(torch.maximum(target, (center-2).clamp(min=0)), (center+2).clamp(max=40))
        indices = (center[..., None]+torch.arange(-2, 3)).clamp(0, 40)
        valid = torch.ones_like(indices, dtype=torch.bool)
        valid[..., 1:] = indices[..., 1:] != indices[..., :-1]
        return default, indices, valid

    def forward(self, obs, days):
        x, d = checked_inputs(obs, days)
        default, indices, valid = self.support(x)
        count = valid.sum(-1, keepdim=True)
        prior = torch.where(indices == default[..., None], .98, .02/(count-1))
        logits = (compact_features(x, d) @ self.weight.T + prior.log()).masked_fill(~valid, -torch.inf)
        physical = torch.full((len(x), 10, 41), -torch.inf).scatter_reduce(-1, indices, logits, reduce='amax')
        terminal = torch.full_like(physical, -torch.inf).scatter(-1, default[..., None], 0.)
        return Categorical(logits=torch.where((d >= 27)[:, None, None], terminal, physical))

    def stats(self, obs, days, actions):
        dist = self(obs, days)
        a = torch.as_tensor(actions)
        if a.shape != dist.batch_shape or not ((a == a.long()) & (a >= 0) & (a <= 40)).all():
            raise ValueError('integer physical actions of shape(batch,10) required')
        a = a.long()
        if not (dist.probs.gather(-1, a[..., None]) > 0).all():
            raise ValueError('out-of-support physical action')
        return dist.log_prob(a).sum(-1), dist.entropy().sum(-1)

    @torch.no_grad()
    def act(self, obs, day, greedy=False, generator=None):
        x = torch.as_tensor(obs, dtype=torch.float32)[None]
        dist = self(x, [day])
        action = (dist.probs.argmax(-1) if greedy else
                  torch.multinomial(dist.probs[0], 1, generator=generator).T)
        return action[0].numpy(), float(dist.log_prob(action).sum())


def episode_targets(rows):
    """Compute each finite episode separately; returns align with flattened rows."""
    adv, ret = [], []
    for episode in rows:
        if len(episode) != 28 or [r['day'] for r in episode] != list(range(1, 29)):
            raise ValueError('complete ordered28-day episodes required')
        done = np.zeros(28); done[-1] = 1
        a, v = compute_gae(np.array([r['reward']/100 for r in episode]),
                           np.array([r['value'] for r in episode]), done, 0., .99, .95)
        adv.extend(a); ret.extend(v)
    return np.asarray(adv), np.asarray(ret)


class CompactResidualPPO:
    variant = 'compact_projected_residual_ppo_v1'

    def __init__(self, state_dim, coefficients, identity, provenance, seed):
        if len(identity) != 10 or len(set(identity)) != 10:
            raise ValueError('ten unique ordered SKU identifiers required')
        torch.manual_seed(seed)
        self.actor = CompactResidualActor(coefficients)
        self.critic = nn.Sequential(SignedLog(), mlp([state_dim+1, 64, 64, 1]))
        self.critic.apply(orthogonal_init)
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=.0003) for m in [self.actor, self.critic]]
        self.generator = torch.Generator().manual_seed(seed)
        self.spec = dict(features=FEATURES, transforms=TRANSFORMS, identity=list(identity),
                         provenance=copy.deepcopy(provenance), state_dim=state_dim,
                         coefficients=list(map(float, coefficients)), initial_default_probability=.98,
                         gamma=.99, lam=.95, reward_scale=100., epochs=4, seed=int(seed))
        self.updates = 0

    @torch.no_grad()
    def value(self, state):
        return float(self.critic(torch.as_tensor(state, dtype=torch.float32)).squeeze())

    def update(self, episodes):
        if len(episodes) != 8:
            raise ValueError('exactly8 complete episodes per update required')
        adv, ret = episode_targets(episodes)
        rows = [r for ep in episodes for r in ep]
        x = torch.tensor(np.array([r['obs'] for r in rows]), dtype=torch.float32)
        days = torch.tensor([r['day'] for r in rows])
        actions = torch.tensor(np.array([r['action'] for r in rows]))
        old = torch.tensor([r['log_prob'] for r in rows])
        active = days <= 26
        advantage = torch.tensor(adv[active.numpy()])
        advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
        actor_norms, actor_losses = [], []
        for _ in range(4):
            lp, entropy = self.actor.stats(x[active], days[active], actions[active])
            logratio = lp-old[active]; ratio = logratio.exp()
            kl = ((ratio-1)-logratio).mean()
            if float(kl.detach()) > .02:
                break
            loss = -torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage).mean()-.01*entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite actor loss')
            self.optimizers[0].zero_grad(); loss.backward()
            norm = nn.utils.clip_grad_norm_(self.actor.parameters(), .5, error_if_nonfinite=True)
            self.optimizers[0].step()
            actor_norms.append(float(norm)); actor_losses.append(float(loss.detach()))
        states = torch.tensor(np.array([r['state'] for r in rows]), dtype=torch.float32)
        oldv = torch.tensor([r['value'] for r in rows]); targets = torch.tensor(ret)
        critic_norms = []
        for _ in range(4):
            value = self.critic(states).squeeze(-1)
            clipped = oldv+(value-oldv).clamp(-.2, .2)
            loss = .5*torch.maximum((value-targets).square(), (clipped-targets).square()).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite critic loss')
            self.optimizers[1].zero_grad(); loss.backward()
            norm = nn.utils.clip_grad_norm_(self.critic.parameters(), .5, error_if_nonfinite=True)
            self.optimizers[1].step(); critic_norms.append(float(norm))
        with torch.no_grad():
            lp, entropy = self.actor.stats(x[active], days[active], actions[active])
            diff = lp-old[active]
            variance = float(targets.var(unbiased=False))
            explained = None if variance == 0 else float(1-(targets-self.critic(states).squeeze(-1)).var(unbiased=False)/variance)
        self.updates += 1
        return dict(update=self.updates, daily_steps=len(rows), active_actor_rows=int(active.sum()),
                    actor_epochs=len(actor_losses), actor_losses=actor_losses, actor_grad_norms=actor_norms,
                    critic_epochs=4, critic_loss=float(loss.detach()), critic_grad_norms=critic_norms,
                    post_kl=float((diff.exp()-1-diff).mean()), entropy=float(entropy.mean()),
                    explained_variance=explained, actor_clipped_steps=sum(v > .5 for v in actor_norms),
                    critic_clipped_steps=sum(v > .5 for v in critic_norms))

    def save(self, path):
        torch.save(dict(variant=self.variant, spec=self.spec, actor=self.actor.state_dict(),
                        critic=self.critic.state_dict(), optimizers=[o.state_dict() for o in self.optimizers],
                        action_rng=self.generator.get_state(), torch_rng=torch.get_rng_state(), updates=self.updates), path)

    def load(self, path):
        data = torch.load(path, weights_only=True, map_location='cpu')
        if data.get('variant') != self.variant or data.get('spec') != self.spec:
            raise ValueError('matching compact variant and specification required')
        self.actor.load_state_dict(data['actor']); self.critic.load_state_dict(data['critic'])
        for optimizer, state in zip(self.optimizers, data['optimizers']):
            optimizer.load_state_dict(state)
        self.generator.set_state(data['action_rng']); torch.set_rng_state(data['torch_rng'])
        self.updates = data['updates']
