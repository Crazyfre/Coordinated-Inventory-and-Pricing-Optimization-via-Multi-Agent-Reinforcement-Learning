"""Uninstalled PPO-CTDE: economic support and frozen-prebatch coordinate-Q CV."""
import copy
import math
import numpy as np
import torch
from torch import nn
from outputs.coordinate_reference_mappo import CoordinateReferenceMAPPO
from outputs.temperature_integrated_mappo import TemperatureOrderingActor
from outputs.full_price_actor_v1 import FullPriceActor
from src.algos.value_normalized_mappo import RunningValueNorm


class RootwiseTemperatureOrderingActor(TemperatureOrderingActor):
    def grid_log_masses(self,features):
        flat=features.reshape(-1,10,22)
        masses=torch.stack([TemperatureOrderingActor.grid_log_masses(self,row) for row in flat])
        return masses.reshape(features.shape[:-1]+(41,))


class FullPriceCoordinateQMAPPO(CoordinateReferenceMAPPO):
    variant = 'full_physical_price_coordinate_q_ppo_ctde_v1'
    design = 'full_price_prebatch_all_coordinate_q_mc_correction_v1'
    def __init__(self, env, covers, phi, seed=0, *, critic_mode='central'):
        super().__init__(env, covers, seed, critic_mode=critic_mode)
        self.pricing = FullPriceActor(env.base_price, phi)
        self.ordering = RootwiseTemperatureOrderingActor(covers)
        self.temperature_arm = 'annealed'
        with torch.no_grad():
            self.ordering.weight.zero_()
            self.ordering.weight[:, 0, :] = torch.log(torch.tensor([.32, .36, .32]))
        for network in self.critics.values():
            network[-1] = nn.Linear(64, 41)
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)
        self.value_norm = RunningValueNorm()
        self.actor_optimizers = {r: torch.optim.Adam(getattr(self, r).parameters(), lr=.003) for r in ('pricing', 'ordering')}
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=.001)

    @torch.no_grad()
    def value(self, role, state):
        # No action-independent state-value baseline is used in this variant.
        return 0.

    @torch.no_grad()
    def q_table(self, state, role, features):
        inputs = np.repeat(np.asarray(state, dtype=np.float32)[None, :], 10, axis=0)
        inputs[:, -10:] = np.eye(10)
        width = 5 if role == 'pricing' else 41
        values = self.value_norm.denormalize(self.critic(role)(torch.tensor(inputs)))[:, :width]
        anchor = torch.full((10,), 4, dtype=torch.long) if role == 'pricing' else torch.tensor(features[:, -1], dtype=torch.long)
        return (values - values[torch.arange(10), anchor][:, None]).numpy()

    def collect(self, env, key, *, budget=None, deadline=float('inf'), training=True, greedy=False):
        if not training:
            return CoordinateReferenceMAPPO.collect(self, env, key, greedy=greedy, training=False, deadline=deadline)
        if greedy or budget is None:
            raise ValueError('explicit joint stochastic on-policy collection/accounting required')
        from outputs.full_price_q_collector_v1 import integrated_collect
        return integrated_collect(self, env, key, budget=budget, deadline=deadline)

    def update(self, episodes, *, epochs=4):
        if not episodes or any(ep['frozen'] for ep in episodes) or epochs != 4:
            raise ValueError('complete unfrozen joint batch/four epochs required')
        tensors, metrics = {}, {}
        tensor = lambda values: torch.as_tensor(np.asarray(values), dtype=torch.float32)
        # Freeze/validate BOTH roles before any actor/critic mutation.
        for role in ('pricing', 'ordering'):
            rows, roots = [], []
            for ep in episodes:
                current = ep['rows'][role]
                if [r['start'] for r in current] != ([0, 7, 14, 21] if role == 'pricing' else list(range(28))):
                    raise ValueError('declared role cadence required')
                selected = [r for r in ep['roots'] if r['role'] == role]
                if len(selected) != len(current):
                    raise ValueError('complete root supports required')
                rows += current
                roots += selected
            f = tensor([r['features'] for r in rows])
            actions = torch.tensor(np.asarray([r['action'] for r in rows]), dtype=torch.long)
            old = getattr(self, role).distribution(f).probs.detach()
            all_q = tensor([r['q_pre'] for r in rows])
            indices, coordinates, alternatives, masses, targets = [], [], [], [], []
            for k, (row, root) in enumerate(zip(rows, roots)):
                i = row['coordinate']
                enabled = np.arange(10)
                if row['coordinate_multiplier'] != len(enabled) or (i not in enabled if len(enabled) else i != 0):
                    raise ValueError('state-only coordinate multiplier/support required')
                if not np.array_equal(row['q_pre'], self.q_table(row['state'], role, row['features'])):
                    raise ValueError('Q must remain frozen before both actor updates')
                support = torch.nonzero(old[k, i] > 0).flatten().tolist()
                if [e['action'] for e in root['entries']] != support:
                    raise ValueError('complete positive physical action support required')
                for entry in root['entries']:
                    if entry['probability'] != float(old[k, i, entry['action']]):
                        raise ValueError('exact old physical masses required')
                    indices.append(k); coordinates.append(i); alternatives.append(entry['action'])
                    masses.append(entry['probability']); targets.append(entry['residual_return'])
            ri, ci, ai = [torch.tensor(v, dtype=torch.long) for v in (indices, coordinates, alternatives)]
            mass, target = tensor(masses), tensor(targets)
            mean = (mass * target).sum() / len(rows)
            scale = ((mass * (target - mean).square()).sum() / len(rows)).sqrt() + 1e-8
            q_selected = all_q[ri, ci, ai]
            tensors[role] = dict(rows=rows, features=f, actions=actions, old=old, q=all_q,
                ri=ri, ci=ci, ai=ai, mass=mass, target=target, mean=mean.detach(), scale=scale.detach(),
                q_selected=q_selected, state=tensor([r['state'] for r in rows]),
                old_logp=tensor([r['joint_logp'] for r in rows]),
                gamma=tensor([self.gamma ** r['start'] for r in rows]),
                factors=tensor([r['coordinate_multiplier'] for r in rows]))
        for role, item in tensors.items():
            actor = getattr(self, role)
            steps = 0
            for _ in range(epochs):
                distribution = actor.distribution(item['features'])
                delta = distribution.log_prob(item['actions']).sum(-1) - item['old_logp']
                if float((delta.exp() - 1 - delta).mean().detach()) > .02:
                    break
                ratio = distribution.probs / item['old'].clamp_min(1e-20)
                model_adv = (item['q'] - item['mean']) / item['scale']
                full = item['old'] * torch.minimum(ratio * model_adv, ratio.clamp(.9, 1.1) * model_adv)
                selected_ratio = ratio[item['ri'], item['ci'], item['ai']]
                residual = (item['target'] - item['q_selected']) / item['scale']
                correction = item['mass'] * torch.minimum(selected_ratio * residual,
                                                         selected_ratio.clamp(.9, 1.1) * residual)
                loss = -(item['gamma'] * full.sum((1, 2))).mean()
                loss -= (item['gamma'][item['ri']] * item['factors'][item['ri']] * correction).sum() / len(item['rows'])
                optimizer = self.actor_optimizers[role]
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), .5); optimizer.step(); steps += 1
            with torch.no_grad():
                delta = actor.distribution(item['features']).log_prob(item['actions']).sum(-1) - item['old_logp']
                metrics[role + '_kl'] = float((delta.exp() - 1 - delta).mean())
            metrics[role + '_actor_steps'] = steps
            metrics[role + '_mc_target_mean'] = float(item['target'].mean())
            metrics[role + '_pre_q_abs_mean'] = float(item['q'].abs().mean())
        self.value_norm.update([tensors[r]['target'] for r in ('pricing', 'ordering')])
        for _ in range(epochs):
            losses = []
            for role, item in tensors.items():
                predicted = self.critic(role)(item['state'])[item['ri'], item['ai']]
                losses.append((item['mass'] * (predicted - self.value_norm.normalize(item['target'])).square()).sum() / len(item['rows']))
            loss = .25 * sum(losses)
            self.critic_optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.critics.parameters(), .5); self.critic_optimizer.step()
        metrics.update(critic_loss=float(loss.detach()), value_norm_updates=int(self.value_norm.updates))
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError('finite coordinate-Q update metrics required')
        return metrics

    def checkpoint(self):
        return dict(variant=self.variant, design=self.design, mode=self.mode, gamma=self.gamma,
            temperature_arm='annealed', ordering_temperature=float(self.ordering.temperature), pricing_temperature=float(self.pricing.temperature),
            pricing_schema=self.pricing.schema, prior=(.1875, .1875, .1875, .1875, .25), q_fit_timing='after_both_actor_updates',
            model=self.state_dict(), actors={r: o.state_dict() for r, o in self.actor_optimizers.items()},
            critic_optimizer=self.critic_optimizer.state_dict(), rng=torch.get_rng_state())

    def restore(self, data):
        expected = self.checkpoint()
        for field in ('variant', 'design', 'mode', 'gamma', 'temperature_arm', 'pricing_schema', 'prior', 'q_fit_timing'):
            if data.get(field) != expected[field]:
                raise ValueError('matching economic-candidate/Q schema required')
        tau = data.get('ordering_temperature')
        state = data.get('model', {})
        if not isinstance(tau, (float, int)) or not math.isfinite(tau) or not .25 <= tau <= 1:
            raise ValueError('valid saved ordering temperature required')
        if state.keys() != expected['model'].keys() or any(not isinstance(v, torch.Tensor) or
            v.shape != expected['model'][k].shape or not torch.isfinite(v).all() for k, v in state.items()):
            raise ValueError('complete finite matching tensors required')
        if data.get('pricing_temperature') != tau or float(state['pricing.temperature']) != tau or float(state['ordering.temperature']) != tau:
            raise ValueError('matching temperature tensor required')
        for name in ('pricing.base_prices', 'pricing.phi', 'ordering.covers'):
            if not torch.equal(state[name], expected['model'][name]):
                raise ValueError('exact static actor buffers required')
        # Validate optimizer/RNG restoration on isolated copies before mutation.
        for role in ('pricing', 'ordering'):
            copy.deepcopy(self.actor_optimizers[role]).load_state_dict(data['actors'][role])
        copy.deepcopy(self.critic_optimizer).load_state_dict(data['critic_optimizer'])
        if not isinstance(data['rng'], torch.Tensor) or data['rng'].dtype != torch.uint8 or data['rng'].shape != torch.get_rng_state().shape:
            raise ValueError('matching CPU policy RNG required')
        self.load_state_dict(state)
        for role, optimizer in self.actor_optimizers.items():
            optimizer.load_state_dict(data['actors'][role])
        self.critic_optimizer.load_state_dict(data['critic_optimizer'])
        torch.set_rng_state(data['rng'])
