"""Uninstalled shared-Fisher pricing / unchanged Adam ordering joint role-Q PPO."""
import numpy as np
import torch
from torch import nn
from outputs.context_curriculum_q_mappo_v1 import ContextCurriculumQMAPPO
from outputs.role_calibrated_q_components_v1 import anchored_weighted_loss
from outputs.shared_price_fisher_step_v1 import PriceFisherState,CONFIG


def role_surrogate(actor,item):
    distribution=actor.distribution(item['features'])
    ratio=distribution.probs/item['old'].clamp_min(1e-20)
    model_adv=(item['q']-item['mean'])/item['scale']
    model=item['old']*torch.minimum(ratio*model_adv,ratio.clamp(.9,1.1)*model_adv)
    selected_ratio=ratio[item['ri'],item['ci'],item['ai']]
    residual=(item['target']-item['q_selected'])/item['scale']
    correction=item['mass']*torch.minimum(selected_ratio*residual,selected_ratio.clamp(.9,1.1)*residual)
    loss=-(item['gamma']*model.sum((1,2))).mean()
    return loss-(10*item['gamma'][item['ri']]*correction).sum()/len(item['rows'])


class SharedPriceFisherQMAPPO(ContextCurriculumQMAPPO):
    variant='shared_price_fisher_context_curriculum_q_mappo_v1'
    design='shared720_damped_fisher_kl002_armijo_price_order_adam_role_q_v1'
    schemas=dict(ContextCurriculumQMAPPO.schemas,pricing_optimizer_schema=CONFIG['schema'],pricing_fisher_damping=.001,pricing_exact_kl_limit=.002,pricing_backtracks=12)

    def __init__(self,env,covers,phi,seed=0,*,critic_mode='central'):
        super().__init__(env,covers,phi,seed,critic_mode=critic_mode)
        self.actor_optimizers['pricing']=PriceFisherState()

    def restore(self,data):
        if not isinstance(data,dict):raise ValueError('Complete new Fisher joint checkpoint required')
        saved=data.get('actors',{}).get('pricing',{})
        check=PriceFisherState();check.load_state_dict(saved)
        observed=data.get('model',{}).get('q_weight.observations')
        if not isinstance(observed,torch.Tensor) or observed.shape!=(2,) or not 0<=int(observed[0])<=32:
            raise ValueError('Complete bounded role calibration observations required')
        if not int(observed[0])<=check.attempts<=4*int(observed[0]):
            raise ValueError('One to four price attempts per complete calibration batch required')
        return super().restore(data)

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
        # Both actors finish before any calibration or critic mutation.
        fields=('loss_before','loss_after','kl','gradient_norm','scale','trials','accepted','reason')
        for epoch in range(4):
            metrics['pricing_natural_'+str(epoch)+'_attempted']=0
            for field in fields:metrics['pricing_natural_'+str(epoch)+'_'+field]=0
        for role,item in batches.items():
            actor=getattr(self,role);steps=0
            for epoch in range(4):
                distribution=actor.distribution(item['features'])
                delta=distribution.log_prob(item['actions']).sum(-1)-item['old_logp']
                if float((delta.exp()-1-delta).mean().detach())>.02:break
                if role=='pricing':
                    result=self.actor_optimizers[role].step(actor,item['features'],item['old'],lambda:role_surrogate(actor,item))
                    metrics['pricing_natural_'+str(epoch)+'_attempted']=1
                    for field in fields:metrics['pricing_natural_'+str(epoch)+'_'+field]=int(result[field]) if isinstance(result[field],bool) else result[field]
                    steps+=int(result['accepted'])
                    if not result['accepted']:break
                else:
                    loss=role_surrogate(actor,item);optimizer=self.actor_optimizers[role]
                    optimizer.zero_grad();loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(),.5);optimizer.step();steps+=1
            with torch.no_grad():
                delta=actor.distribution(item['features']).log_prob(item['actions']).sum(-1)-item['old_logp']
                metrics[role+'_kl']=float((delta.exp()-1-delta).mean())
            metrics.update({role+'_actor_steps':steps,role+'_mc_target_mean':float(item['target'].mean()),
                role+'_pre_q_abs_mean':float(item['q'].abs().mean()),role+'_raw_pre_q_abs_mean':float(item['raw'].abs().mean()),role+'_q_beta_pre':item['beta']})
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

