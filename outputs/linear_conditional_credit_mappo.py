"""Uninstalled joint PPO/CTDE prototype with two declared credit modes.

Same linear actors and SKU-conditioned shared critic in both modes. Conditional
mode uses selected-coordinate ratios/returns; whole mode uses full-role ratios
and same-key whole-reference returns. A credit PACKAGE comparison, not vanilla
MAPPO equivalence or an isolated single-component ablation. No research fit yet.
"""
import numpy as np
import torch
from torch import nn
from outputs.linear_coordinate_diagnostic import LinearCoordinateDiagnostic
from outputs.coordinate_credit_prototype import selected_log_prob,clipped_terms


class LinearConditionalCreditMAPPO(LinearCoordinateDiagnostic):
    def __init__(self,env,covers,seed=0,*,credit_mode='coordinate'):
        if credit_mode not in ('whole','coordinate'):raise ValueError('whole or coordinate credit required')
        super().__init__(env,covers,seed);self.credit_mode=credit_mode
        self.actor_optimizers={role:torch.optim.Adam(getattr(self,role).parameters(),lr=.003) for role in ('pricing','ordering')}

    def update(self,episodes,*,epochs=4):
        if not episodes or any(ep['frozen'] for ep in episodes):raise ValueError('joint on-policy batch required')
        batches={};metrics={}
        for role in ('pricing','ordering'):
            records=[];targets=[]
            for ep in episodes:
                rows=ep['rows'][role]
                if [r['start'] for r in rows]!=([0,7,14,21] if role=='pricing' else list(range(28))):
                    raise ValueError('role cadence mismatch')
                if self.credit_mode=='whole':
                    actual=np.asarray(ep['rewards'],dtype=float);reference=np.asarray(ep['whole_reference_rewards'],dtype=float)
                    if actual.shape!=(28,) or reference.shape!=(28,):raise ValueError('whole reference must cover28days')
                    values=[sum(self.gamma**j*(a-b)/100 for j,(a,b) in enumerate(zip(actual[r['start']:],reference[r['start']:]))) for r in rows]
                else:values=[r['target'] for r in rows]
                records.extend(rows);targets.extend(values)
            tensor=lambda values:torch.as_tensor(np.asarray(values),dtype=torch.float32)
            features=tensor([r['features'] for r in records]);actions=torch.as_tensor(np.asarray([r['action'] for r in records]),dtype=torch.long)
            coordinates=torch.tensor([r['coordinate'] for r in records],dtype=torch.long)
            old=tensor([r['logp'] for r in records]);joint_old=tensor([r['joint_logp'] for r in records])
            target=tensor(targets);adv=target-tensor([r['value'] for r in records])
            if not torch.isfinite(adv).all():raise ValueError('finite critic targets required')
            adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            time_weight=tensor([self.gamma**r['start'] for r in records]);actor=getattr(self,role);steps=0
            for _ in range(epochs):
                distribution=actor.distribution(features);component=distribution.log_prob(actions)
                full=component.sum(-1)-joint_old
                if float((full.exp()-1-full).mean().detach())>.02:break
                if self.credit_mode=='coordinate':
                    lp=selected_log_prob(distribution,actions,coordinates)
                    surrogate=clipped_terms(lp,old,adv)
                else:
                    ratio=full.exp();surrogate=torch.minimum(ratio*adv,ratio.clamp(.9,1.1)*adv)
                loss=-(time_weight*surrogate).mean()
                optimizer=self.actor_optimizers[role];optimizer.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(),.5);optimizer.step();steps+=1
            with torch.no_grad():
                full=actor.distribution(features).log_prob(actions).sum(-1)-joint_old
                metrics[role+'_kl']=float((full.exp()-1-full).mean())
            metrics[role+'_actor_steps']=steps;metrics[role+'_critic_target_mean']=float(target.mean())
            batches[role]=(tensor([r['state'] for r in records]),target)
        self.value_norm.update([target for _,target in batches.values()])
        for _ in range(epochs):
            loss=.25*sum((self.critic(role)(state).squeeze(-1)-self.value_norm.normalize(target)).square().mean()
                for role,(state,target) in batches.items())
            self.critic_optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.critics.parameters(),.5)
            self.critic_optimizer.step()
        mean,std=self.value_norm.moments()
        metrics.update(critic_loss=float(loss.detach()),value_norm_updates=int(self.value_norm.updates),value_mean=float(mean),value_std=float(std))
        if not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError('nonfinite joint update')
        return metrics

    def checkpoint(self):
        return dict(variant='linear_conditional_credit_mappo_v1',credit_mode=self.credit_mode,mode=self.mode,
            gamma=self.gamma,critic_target='selected_coordinate_mc' if self.credit_mode=='coordinate' else 'whole_reference_mc',
            model=self.state_dict(),actors={role:optimizer.state_dict() for role,optimizer in self.actor_optimizers.items()},
            critic_optimizer=self.critic_optimizer.state_dict(),rng=torch.get_rng_state())

    def restore(self,data):
        expected='selected_coordinate_mc' if self.credit_mode=='coordinate' else 'whole_reference_mc'
        if (data.get('variant')!='linear_conditional_credit_mappo_v1' or data.get('credit_mode')!=self.credit_mode
            or data.get('mode')!=self.mode or data.get('gamma')!=self.gamma or data.get('critic_target')!=expected):
            raise ValueError('matching conditional-credit mode/schema required')
        self.load_state_dict(data['model'])
        for role,optimizer in self.actor_optimizers.items():optimizer.load_state_dict(data['actors'][role])
        self.critic_optimizer.load_state_dict(data['critic_optimizer']);torch.set_rng_state(data['rng'])
