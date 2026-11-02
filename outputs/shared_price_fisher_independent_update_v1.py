"""Independent complete numerical transition; no new producer update/Fisher/step call."""
import math
import numpy as np
import torch
from torch import nn
from outputs.role_calibrated_q_components_v1 import anchored_weighted_loss
from outputs.context_curriculum_q_independent_replay_v1 import independent_price_distribution


def independent_distribution(actor,features):
    return independent_price_distribution(actor,features.detach().numpy())


def independent_surrogate(actor,item):
    distribution=independent_distribution(actor,item['features']) if actor.weight.shape==(180,4) else actor.distribution(item['features'])
    ratio=distribution.probs/item['old'].clamp_min(1e-20)
    model_adv=(item['q']-item['mean'])/item['scale']
    model=item['old']*torch.minimum(ratio*model_adv,ratio.clamp(.9,1.1)*model_adv)
    selected_ratio=ratio[item['ri'],item['ci'],item['ai']]
    residual=(item['target']-item['q_selected'])/item['scale']
    correction=item['mass']*torch.minimum(selected_ratio*residual,selected_ratio.clamp(.9,1.1)*residual)
    loss=-(item['gamma']*model.sum((1,2))).mean()
    loss-=(10*item['gamma'][item['ri']]*correction).sum()/len(item['rows'])
    return loss


def independent_kl(old,new):
    a,b=old.double().detach().numpy(),new.double().detach().numpy()
    return math.fsum(float(x)*(math.log(float(x))-math.log(float(y))) for x,y in zip(a.flat,b.flat))/(a.shape[0]*10)


def independent_price_step(state,actor,features,old,loss_fn):
    if state.attempts>=128:raise ValueError('Complete128step budget')
    loss=loss_fn();gradient,=torch.autograd.grad(loss,actor.weight)
    if not torch.isfinite(loss) or not torch.isfinite(gradient).all():raise ValueError('Finite independently reconstructed loss/gradient')
    before=float(loss.detach());norm=float(torch.linalg.vector_norm(gradient));baseline=actor.weight.detach().clone()
    current=independent_distribution(actor,features).probs.detach();initial_kl=independent_kl(old,current)
    result=dict(loss_before=before,loss_after=before,kl=initial_kl,gradient_norm=norm,scale=0.,trials=0,accepted=False,reason=1)
    if norm:
        index=features.long();columns=torch.arange(4)
        raw=(8.*actor.weight[index,columns]).detach().requires_grad_()
        derivative,=torch.autograd.grad(raw.clamp(-6,6).sum(),raw)
        h=derivative.double().numpy().reshape(-1,4)*(8/float(actor.temperature))
        p=current.double().numpy().reshape(-1,5);ids=(index.numpy()*4+np.arange(4)).reshape(-1,4)
        matrix=np.zeros((720,720),dtype=np.float64)
        for j in range(len(p)):
            covariance=(np.diag(p[j,:4])-np.outer(p[j,:4],p[j,:4]))*np.outer(h[j],h[j])/len(p)
            matrix[np.ix_(ids[j],ids[j])]+=covariance
        clipped=(gradient*min(1.,.5/(norm+1e-6))).double().numpy().reshape(-1)
        direction=np.linalg.solve(matrix+.001*np.eye(720),-clipped).reshape(180,4)
        raw_gradient=gradient.double().numpy().reshape(-1)
        predicted=float(-raw_gradient@direction.reshape(-1));quadratic=float(.5*direction.reshape(-1)@matrix@direction.reshape(-1))
        alpha=min(1.,math.sqrt(.002/max(quadratic,1e-30)))
        result['reason']=2
        try:
            for trial in range(12):
                scale=alpha*2.**(-trial)
                # Preserve actual float32 multiplication/addition precision.
                proposal=baseline.numpy()+np.float32(scale)*direction.astype(np.float32)
                with torch.no_grad():actor.weight.copy_(torch.from_numpy(proposal));value=float(loss_fn())
                new=independent_distribution(actor,features).probs.detach();kl=independent_kl(old,new)
                result['trials']=trial+1
                if math.isfinite(value) and math.isfinite(kl) and kl<=.002 and value<before and not torch.equal(actor.weight,baseline) and value<=before-.1*scale*predicted:
                    result.update(loss_after=value,kl=kl,scale=scale,accepted=True,reason=0);break
            if not result['accepted']:
                with torch.no_grad():actor.weight.copy_(baseline)
        except BaseException:
            with torch.no_grad():actor.weight.copy_(baseline)
            raise
    state.attempts+=1;state.accepted+=int(result['accepted']);state.rejected+=int(not result['accepted']);state.trials+=result['trials']
    return result


def independent_update(self, episodes, *, epochs=4):
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
                result=independent_price_step(self.actor_optimizers[role],actor,item['features'],item['old'],lambda:independent_surrogate(actor,item))
                metrics['pricing_natural_'+str(epoch)+'_attempted']=1
                for field in fields:metrics['pricing_natural_'+str(epoch)+'_'+field]=int(result[field]) if isinstance(result[field],bool) else result[field]
                steps+=int(result['accepted'])
                if not result['accepted']:break
            else:
                loss=independent_surrogate(actor,item);optimizer=self.actor_optimizers[role]
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

