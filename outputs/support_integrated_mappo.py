"""UNINSTALLED learner prototype; no calibrated fit or automatic experiment.

PPO surrogate integrates selected-SKU physical action support with OLD masses.
Both local linear actors learn; an indexed shared global critic fits the old-
policy expected residual return and supplies pre-update baseline predictions.
"""
import numpy as np
import torch
from torch import nn
from outputs.enumerated_credit_diagnostic import EnumeratedCreditDiagnostic
from outputs.coordinate_reference_mappo import CoordinateReferenceMAPPO


class SupportIntegratedMAPPO(EnumeratedCreditDiagnostic):
    def __init__(self,env,covers,seed=0):
        super().__init__(env,covers,seed)
        self.actor_optimizers={r:torch.optim.Adam(getattr(self,r).parameters(),lr=.003) for r in ('pricing','ordering')}

    def collect(self,env,key,*,budget=None,deadline=float('inf'),training=True,greedy=False):
        if not training:
            return CoordinateReferenceMAPPO.collect(self,env,key,greedy=greedy,training=False,deadline=deadline)
        if greedy or budget is None:raise ValueError('stochastic on-policy collection and explicit accounting required')
        return super().collect(env,key,budget=budget,deadline=deadline)

    def update(self,episodes,*,epochs=4):
        if not episodes or any(ep['frozen'] for ep in episodes):raise ValueError('unfrozen joint on-policy batch required')
        batches={};metrics={}
        for role in ('pricing','ordering'):
            records=[];roots=[]
            for ep in episodes:
                rows=ep['rows'][role]
                if [r['start'] for r in rows]!=([0,7,14,21] if role=='pricing' else list(range(28))):raise ValueError('cadence')
                selected=[r for r in ep['roots'] if r['role']==role]
                if len(rows)!=len(selected):raise ValueError('complete root support required')
                records+=rows;roots+=selected
            tensor=lambda a:torch.as_tensor(np.asarray(a),dtype=torch.float32)
            features=tensor([r['features'] for r in records]);actions=torch.tensor(np.asarray([r['action'] for r in records]),dtype=torch.long)
            joint_old=tensor([r['joint_logp'] for r in records])
            row_indices=[];skus=[];candidates=[];old_mass=[];returns=[];root_targets=[]
            for k,(row,root) in enumerate(zip(records,roots)):
                entries=root['entries'];total=sum(e['probability'] for e in entries)
                if abs(total-1)>1e-6 or any(e['probability']<=0 for e in entries):raise ValueError('full positive physical support')
                root_targets.append(sum(e['probability']*e['residual_return'] for e in entries))
                for entry in entries:
                    row_indices.append(k);skus.append(row['coordinate']);candidates.append(entry['action'])
                    old_mass.append(entry['probability']);returns.append(entry['residual_return'])
            row_indices=torch.tensor(row_indices,dtype=torch.long);skus=torch.tensor(skus,dtype=torch.long)
            candidates=torch.tensor(candidates,dtype=torch.long);old_mass=tensor(old_mass)
            values=tensor([r['value'] for r in records]);adv=tensor(returns)-values[row_indices]
            # Per-role old-action-mass weighted moments; each root has mass1.
            mean=(old_mass*adv).sum()/len(records)
            variance=(old_mass*(adv-mean).square()).sum()/len(records)
            adv=(adv-mean)/(variance.sqrt()+1e-8)
            weights=tensor([self.gamma**r['start'] for r in records])[row_indices]
            actor=getattr(self,role);steps=0
            for _ in range(epochs):
                distribution=actor.distribution(features)
                delta=distribution.log_prob(actions).sum(-1)-joint_old
                if float((delta.exp()-1-delta).mean().detach())>.02:break
                mass=distribution.probs[row_indices,skus,candidates]
                ratio=mass/old_mass
                surrogate=torch.minimum(ratio*adv,ratio.clamp(.9,1.1)*adv)
                loss=-10*(old_mass*weights*surrogate).sum()/len(records)
                optimizer=self.actor_optimizers[role];optimizer.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(),.5);optimizer.step();steps+=1
            with torch.no_grad():
                delta=actor.distribution(features).log_prob(actions).sum(-1)-joint_old
                metrics[role+'_kl']=float((delta.exp()-1-delta).mean())
            target=tensor(root_targets);batches[role]=(tensor([r['state'] for r in records]),target)
            metrics[role+'_actor_steps']=steps;metrics[role+'_critic_target_mean']=float(target.mean())
        self.value_norm.update([target for _,target in batches.values()])
        for _ in range(epochs):
            loss=.25*sum((self.critic(role)(state).squeeze(-1)-self.value_norm.normalize(target)).square().mean()
                for role,(state,target) in batches.items())
            self.critic_optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.critics.parameters(),.5);self.critic_optimizer.step()
        mean,std=self.value_norm.moments()
        metrics.update(critic_loss=float(loss.detach()),value_norm_updates=int(self.value_norm.updates),value_mean=float(mean),value_std=float(std))
        if not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError('finite update metrics required')
        return metrics

    def checkpoint(self):
        return dict(variant='support_integrated_mappo_prototype_v1',mode=self.mode,gamma=self.gamma,
            critic_target='old_policy_expected_coordinate_residual_mc',model=self.state_dict(),
            actors={r:o.state_dict() for r,o in self.actor_optimizers.items()},critic_optimizer=self.critic_optimizer.state_dict(),rng=torch.get_rng_state())

    def restore(self,data):
        if (data.get('variant')!='support_integrated_mappo_prototype_v1' or data.get('mode')!=self.mode or data.get('gamma')!=self.gamma
            or data.get('critic_target')!='old_policy_expected_coordinate_residual_mc'):raise ValueError('support-integrated schema required')
        self.load_state_dict(data['model'])
        for role,optimizer in self.actor_optimizers.items():optimizer.load_state_dict(data['actors'][role])
        self.critic_optimizer.load_state_dict(data['critic_optimizer']);torch.set_rng_state(data['rng'])
