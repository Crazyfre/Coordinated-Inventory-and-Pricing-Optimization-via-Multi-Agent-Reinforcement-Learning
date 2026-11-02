"""Uninstalled two-role PPO-CTDE with economic features and calibrated relative Q."""
import math
import numpy as np
import torch
from torch import nn

from outputs.full_price_coordinate_q_mappo_v1 import FullPriceCoordinateQMAPPO
from outputs.role_calibrated_q_components_v1 import (EconomicFeaturePriceActor, LockedRoleRMS,
    LaggedQWeight, SharedRoleActionQ, LocalRoleActionQ, anchor_differences,
    anchored_weighted_loss)


class RoleCalibratedQMAPPO(FullPriceCoordinateQMAPPO):
    variant = 'role_calibrated_economic_feature_q_mappo_v1'
    design = 'anchored_role_scaled_lagged_q_separate_ppo_clipping_v1'
    schemas = dict(critic_scale_schema='first_train_batch_locked_role_rms_v1',
        q_coefficient_schema='lagged_prebatch_q_ema_clamped_v1',
        critic_head_schema='shared_trunk_role_heads_anchor_difference_v1', critic_epochs=8)

    def __init__(self, env, covers, phi, seed=0, *, critic_mode='central'):
        super().__init__(env, covers, phi, seed, critic_mode=critic_mode)
        widths = {name: network[0].in_features for name, network in self.critics.items()}
        self.pricing = EconomicFeaturePriceActor(env.base_price, phi)
        if self.mode == 'central':
            self.critics = nn.ModuleDict({'shared': SharedRoleActionQ(widths['shared'])})
        else:
            self.critics = nn.ModuleDict({'local': LocalRoleActionQ(widths)})
        del self.value_norm
        self.role_scales = LockedRoleRMS()
        self.q_weight = LaggedQWeight()
        self.actor_optimizers = {r: torch.optim.Adam(getattr(self, r).parameters(), lr=.003)
                                 for r in ('pricing', 'ordering')}
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=.001)

    def predict_raw(self, states, role):
        return self.critics['shared' if self.mode == 'central' else 'local'](states, role)

    @torch.no_grad()
    def raw_q_table(self, state, role, features):
        expanded = np.repeat(np.asarray(state, dtype=np.float32)[None, :], 10, axis=0)
        expanded[:, -10:] = np.eye(10)
        anchors = (torch.full((10,), 4, dtype=torch.long) if role == 'pricing'
                   else torch.as_tensor(features[:, -1], dtype=torch.long))
        prediction = self.predict_raw(torch.tensor(expanded), role)
        return (anchor_differences(prediction, anchors) * self.role_scales.scale(role)).numpy()

    @torch.no_grad()
    def q_table(self, state, role, features):
        raw = torch.as_tensor(self.raw_q_table(state, role, features))
        return (self.q_weight.coefficient(role) * raw).numpy()

    def collect(self, env, key, *, budget=None, deadline=float('inf'), training=True, greedy=False):
        if not training:
            return super().collect(env, key, budget=budget, deadline=deadline, training=False, greedy=greedy)
        from outputs.full_price_q_collector_v1 import integrated_collect
        if greedy or budget is None:
            raise ValueError('Complete stochastic joint collection with explicit accounting required')
        result = integrated_collect(self, env, key, budget=budget, deadline=deadline)
        for role, rows in result['rows'].items():
            beta = float(self.q_weight.coefficient(role))
            for row in rows:
                row['q_raw_pre'] = self.raw_q_table(row['state'], role, row['features'])
                row['q_beta_pre'] = beta
                assert np.array_equal(row['q_pre'], row['q_raw_pre'] * np.float32(beta))
        return result

    def update(self, episodes, *, epochs=4):
        if not episodes or epochs != 4 or any(ep['frozen'] for ep in episodes):
            raise ValueError('Complete unfrozen joint batch and four actor epochs required')
        tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
        batches, metrics = {}, {}
        for role in ('pricing', 'ordering'):
            rows, roots = [], []
            for ep in episodes:
                current = ep['rows'][role]
                assert [r['start'] for r in current] == ([0, 7, 14, 21] if role == 'pricing' else list(range(28)))
                selected = [r for r in ep['roots'] if r['role'] == role]
                assert len(current) == len(selected)
                rows.extend(current)
                roots.extend(selected)
            features = tensor([r['features'] for r in rows])
            actions = torch.as_tensor(np.asarray([r['action'] for r in rows]), dtype=torch.long)
            old = getattr(self, role).distribution(features).probs.detach()
            q, raw = tensor([r['q_pre'] for r in rows]), tensor([r['q_raw_pre'] for r in rows])
            beta = float(self.q_weight.coefficient(role))
            indices, coordinates, alternatives, masses, targets = [], [], [], [], []
            for k, (row, root) in enumerate(zip(rows, roots)):
                i = row['coordinate']
                assert row['coordinate_multiplier'] == 10 and 0 <= i < 10 and row['q_beta_pre'] == beta
                assert np.array_equal(row['q_pre'], self.q_table(row['state'], role, row['features']))
                assert np.array_equal(row['q_raw_pre'], self.raw_q_table(row['state'], role, row['features']))
                support = torch.nonzero(old[k, i] > 0).flatten().tolist()
                assert [entry['action'] for entry in root['entries']] == support
                for entry in root['entries']:
                    assert entry['probability'] == float(old[k, i, entry['action']])
                    indices.append(k); coordinates.append(i); alternatives.append(entry['action'])
                    masses.append(entry['probability']); targets.append(entry['residual_return'])
            ri, ci, ai = [torch.tensor(x, dtype=torch.long) for x in (indices, coordinates, alternatives)]
            mass, target = tensor(masses), tensor(targets)
            mean = (mass * target).sum() / len(rows)
            scale = ((mass * (target - mean).square()).sum() / len(rows)).sqrt() + 1e-8
            width = 5 if role == 'pricing' else 41
            dense_mass, dense_target = torch.zeros(len(rows), width), torch.zeros(len(rows), width)
            dense_mass[ri, ai] = mass
            dense_target[ri, ai] = target
            row_coordinate = torch.tensor([r['coordinate'] for r in rows], dtype=torch.long)
            anchors = torch.tensor([r['reference_action'][r['coordinate']] for r in rows], dtype=torch.long)
            batches[role] = dict(rows=rows, features=features, actions=actions, old=old, q=q, raw=raw,
                beta=beta, ri=ri, ci=ci, ai=ai, mass=mass, target=target, mean=mean.detach(), scale=scale.detach(),
                q_selected=q[ri, ci, ai], state=tensor([r['state'] for r in rows]),
                old_logp=tensor([r['joint_logp'] for r in rows]),
                gamma=tensor([self.gamma ** r['start'] for r in rows]),
                dense_mass=dense_mass, dense_target=dense_target, anchors=anchors,
                selected_raw=raw[torch.arange(len(rows)), row_coordinate])
        # No critic, coefficient or scale mutation until BOTH actors finish.
        for role, item in batches.items():
            actor = getattr(self, role)
            steps = 0
            for _ in range(4):
                distribution = actor.distribution(item['features'])
                delta = distribution.log_prob(item['actions']).sum(-1) - item['old_logp']
                if float((delta.exp() - 1 - delta).mean().detach()) > .02:
                    break
                ratio = distribution.probs / item['old'].clamp_min(1e-20)
                model_adv = (item['q'] - item['mean']) / item['scale']
                model = item['old'] * torch.minimum(ratio * model_adv, ratio.clamp(.9, 1.1) * model_adv)
                selected_ratio = ratio[item['ri'], item['ci'], item['ai']]
                residual = (item['target'] - item['q_selected']) / item['scale']
                correction = item['mass'] * torch.minimum(selected_ratio * residual,
                    selected_ratio.clamp(.9, 1.1) * residual)
                loss = -(item['gamma'] * model.sum((1, 2))).mean()
                loss -= (10 * item['gamma'][item['ri']] * correction).sum() / len(item['rows'])
                optimizer = self.actor_optimizers[role]
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), .5)
                optimizer.step(); steps += 1
            with torch.no_grad():
                delta = actor.distribution(item['features']).log_prob(item['actions']).sum(-1) - item['old_logp']
                metrics[role + '_kl'] = float((delta.exp() - 1 - delta).mean())
            metrics.update({role + '_actor_steps': steps,
                role + '_mc_target_mean': float(item['target'].mean()),
                role + '_pre_q_abs_mean': float(item['q'].abs().mean()),
                role + '_raw_pre_q_abs_mean': float(item['raw'].abs().mean()),
                role + '_q_beta_pre': item['beta']})
        for role, item in batches.items():
            if not self.role_scales.locked[('pricing', 'ordering').index(role)]:
                self.role_scales.lock(role, item['dense_target'], item['dense_mass'])
            self.q_weight.observe(role, item['selected_raw'], item['dense_target'], item['dense_mass'])
        for _ in range(8):
            losses = [anchored_weighted_loss(self.predict_raw(item['state'], role), item['anchors'],
                item['dense_target'], item['dense_mass'], self.role_scales.scale(role))
                for role, item in batches.items()]
            loss = .25 * sum(losses)
            self.critic_optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.critics.parameters(), .5)
            self.critic_optimizer.step()
        for role in batches:
            metrics[role + '_q_beta_post'] = float(self.q_weight.coefficient(role))
            metrics[role + '_locked_rms'] = float(self.role_scales.scale(role))
        assert torch.equal(self.q_weight.observations, self.q_weight.observations[:1].expand(2))
        metrics.update(critic_loss=float(loss.detach()), critic_epochs=8,
            calibration_observations=int(self.q_weight.observations[0]), role_scale_locks=int(self.role_scales.locked.sum()))
        assert all(np.isfinite(value) for value in metrics.values())
        return metrics

    def checkpoint(self):
        result = super().checkpoint()
        result.update(self.schemas)
        return result

    def restore(self, data):
        if any(data.get(k) != v for k, v in self.schemas.items()):
            raise ValueError('Matching anchored-role/calibrator schema required')
        expected = self.checkpoint()['model']
        state = data.get('model', {})
        if state.keys() != expected.keys() or any(not isinstance(v, torch.Tensor) or
            v.dtype != expected[k].dtype or v.shape != expected[k].shape for k, v in state.items()):
            raise ValueError('Matching complete checkpoint tensors and dtypes required')
        observed = state['q_weight.observations']
        locked, scales = state['role_scales.locked'], state['role_scales.scales']
        if (observed < 0).any() or (observed > 32).any() or observed[0] != observed[1]:
            raise ValueError('Complete paired calibration history0..32 required')
        if not torch.isfinite(scales).all() or (scales < .01).any() or locked[0] != locked[1]:
            raise ValueError('Finite locked role scales required')
        if bool(locked.all()) != bool(observed[0] > 0):
            raise ValueError('Scale lock and completed-batch history must agree')
        if not locked.any() and not torch.equal(scales, torch.ones_like(scales)):
            raise ValueError('Exact uninitialized role scales required')
        if (state['q_weight.qq'] < 0).any() or not torch.isfinite(state['q_weight.qy']).all() or not torch.isfinite(state['q_weight.qq']).all():
            raise ValueError('Finite lagged moment state required')
        if observed[0] == 0 and (torch.count_nonzero(state['q_weight.qq']) or torch.count_nonzero(state['q_weight.qy'])):
            raise ValueError('Exact initial lagged moments required')
        return super().restore(data)
