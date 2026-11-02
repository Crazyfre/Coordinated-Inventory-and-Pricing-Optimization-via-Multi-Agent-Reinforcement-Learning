"""Synthetic-proof prototype: state-matched coordinate-reference PPO/CTDE.

NOT installed in src, NOT a calibrated experiment, NOT vanilla MAPPO. Both local
neural actors retain their exact prior; one uniformly chosen SKU score is used
at each role decision. A shared SKU-conditioned residual critic enters credit.
All branch days are charged; no opened assessment supplies training examples.
"""
import copy
import time
import numpy as np
import torch
from torch import nn
from src.algos.neural_residual_mappo import NeuralResidualMAPPO
from src.algos.parameter_policy_gradient import order_action
from src.env.interface import JointAction
from outputs.coordinate_credit_prototype import selected_log_prob, clipped_terms


class StepBudget:
    def __init__(self, maximum):
        self.maximum=int(maximum);self.steps=0;self.actual=0;self.branch=0

    def claim(self, kind, deadline):
        if self.steps>=self.maximum or time.perf_counter()>deadline:
            raise RuntimeError('synthetic proof step/time budget')
        if kind not in ('actual','branch'):raise ValueError('unknown step kind')
        self.steps+=1;setattr(self,kind,getattr(self,kind)+1)


def clone_local_env(env):
    result=copy.deepcopy(env)
    assert result is not env and result.pipeline is not env.pipeline
    assert result.inventory is not env.inventory and result.forecaster is not env.forecaster
    assert result.np_random is not env.np_random
    assert not np.shares_memory(result.inventory._on_hand,env.inventory._on_hand)
    # JointRetailEnv.step rebinds its pricing hook before using it. Calibrated
    # ResearchEnv-specific clone/supplier proof is still required before research.
    return result


class CoordinateReferenceMAPPO(NeuralResidualMAPPO):
    def __init__(self,env,covers,seed=0,*,critic_mode='central'):
        super().__init__(env,covers,seed,critic_mode=critic_mode)
        rebuilt={}
        for name,old in self.critics.items():
            value=nn.Sequential(nn.Linear(old[0].in_features+10,64),nn.Tanh(),nn.Linear(64,1))
            nn.init.zeros_(value[-1].weight);nn.init.zeros_(value[-1].bias);rebuilt[name]=value
        self.critics=nn.ModuleDict(rebuilt)
        self.critic_optimizer=torch.optim.Adam(self.critics.parameters(),lr=.001)

    def indexed_state(self,env,obs,day,role,coordinate):
        return np.r_[super().critic_input(env,obs,day,role),np.eye(10)[coordinate]].astype(np.float32)

    @torch.no_grad()
    def _scores(self,role,features,actions):
        return getattr(self,role).distribution(torch.as_tensor(features)).log_prob(torch.as_tensor(actions)).numpy()

    def continuation(self,env,role,root_action,*,budget,deadline):
        """Consume an isolated snapshot; no reset, gradients or critic access."""
        first=int(env.day);trace=[];rewards=[]
        for day in range(first,29):
            price=None
            if env.pricing_due:
                if day==first and role=='ordering':
                    # Today's weekly price already precedes this order root;
                    # retain it and consume NO new pricing policy draw.
                    assert env._pending_price is not None
                    ratios=env._pending_price/env.base_price
                    price=np.abs(ratios[:,None]-np.asarray(env.config.pricing.multipliers)).argmin(1)
                else:
                    if day==first and role=='pricing':price=np.asarray(root_action).copy()
                    else:price,_=self.pricing.act(self.pricing.features(env.pricing_observation(),day))
                    env.select_price(price)
            obs=env.replenishment_observation().copy()
            if day==first and role=='ordering':action=np.asarray(root_action).copy()
            else:action,_=self.ordering.act(self.ordering.features(obs,day))
            budget.claim('branch',deadline)
            _,reward,done,info=env.step(JointAction(None,action));assert done==(day==28)
            rewards.append(float(reward));trace.append(dict(day=day,obs=obs,price_actions=price,
                actions=action,reward=float(reward),info=info))
        return rewards,trace

    def collect(self,env,key,*,greedy=False,frozen=(),training=True,deadline=float('inf'),
                budget=None,keep_proof_snapshots=False):
        if not training:
            return super().collect(env,key,greedy=greedy,frozen=frozen,training=False,deadline=deadline)
        if greedy or frozen:raise ValueError('prototype requires both stochastic learning actors')
        if budget is None:raise ValueError('explicit synthetic proof step budget required')
        env.reset(seed=key);coordinate_rng=np.random.default_rng(np.random.SeedSequence([key,6179]))
        rows={'pricing':[],'ordering':[]};snapshots=[];rewards=[];trace=[]
        def record(role,obs,features,actions,coordinate,reference_action):
            state=self.indexed_state(env,obs,env.day,role,coordinate)
            lp=self._scores(role,features,actions)
            row=dict(features=features,action=actions.copy(),coordinate=coordinate,
                component_logps=lp.copy(),logp=float(lp[coordinate]),joint_logp=float(lp.sum()),
                state=state,value=self.value(role,state),start=len(rewards),
                reference_action=reference_action.copy())
            rows[role].append(row)
            # After root action sampling, before applying that action. This RNG
            # state belongs to future choices and is restored independently.
            snapshots.append((role,row,clone_local_env(env),torch.get_rng_state().clone()))
        for day in range(1,29):
            price=None
            if env.pricing_due:
                coordinate=int(coordinate_rng.integers(10))
                obs=env.pricing_observation().copy();f=self.pricing.features(obs,day)
                price,_=self.pricing.act(f);reference=price.copy();reference[coordinate]=4
                record('pricing',obs,f,price,coordinate,reference);env.select_price(price)
            coordinate=int(coordinate_rng.integers(10))
            obs=env.replenishment_observation().copy();f=self.ordering.features(obs,day)
            action,_=self.ordering.act(f)
            anchor=order_action(obs,day,self.ordering.covers.detach().numpy())
            reference=action.copy();reference[coordinate]=anchor[coordinate]
            record('ordering',obs,f,action,coordinate,reference)
            budget.claim('actual',deadline)
            _,reward,done,info=env.step(JointAction(None,action));assert done==(day==28)
            rewards.append(float(reward));trace.append(dict(day=day,obs=obs,price_actions=price,
                actions=action,reward=float(reward),info=info))
        actual_rng=torch.get_rng_state().clone();reference_days=0;skipped=0
        try:
            for role,row,snapshot,rng in snapshots:
                if np.array_equal(row['action'],row['reference_action']):
                    row['reference_rewards']=rewards[row['start']:].copy();row['branch_days']=0;skipped+=1
                else:
                    torch.set_rng_state(rng)
                    rr,rt=self.continuation(clone_local_env(snapshot),role,row['reference_action'],budget=budget,deadline=deadline)
                    row['reference_rewards']=rr;row['branch_days']=len(rr);reference_days+=len(rr)
                actual=sum(self.gamma**j*r for j,r in enumerate(rewards[row['start']:]))/100
                baseline=sum(self.gamma**j*r for j,r in enumerate(row['reference_rewards']))/100
                row['target']=float(actual-baseline)
        finally:torch.set_rng_state(actual_rng)
        result=dict(rows=rows,rewards=rewards,trace=trace,frozen=[],reference_days=reference_days,
                    skipped_identical=skipped,actual_policy_rng=actual_rng)
        if keep_proof_snapshots:result['proof_snapshots']=snapshots
        return result

    def update(self,episodes,*,epochs=4):
        if not episodes or any(ep['frozen'] for ep in episodes):raise ValueError('joint stochastic batch required')
        batches={};metrics={}
        for role in ('pricing','ordering'):
            records=[]
            for ep in episodes:
                rr=ep['rows'][role]
                if [r['start'] for r in rr]!=([0,7,14,21] if role=='pricing' else list(range(28))):
                    raise ValueError('decision cadence mismatch')
                records.extend(rr)
            tensor=lambda value:torch.as_tensor(np.asarray(value),dtype=torch.float32)
            features=tensor([r['features'] for r in records])
            actions=torch.as_tensor(np.asarray([r['action'] for r in records]),dtype=torch.long)
            indices=torch.tensor([r['coordinate'] for r in records],dtype=torch.long)
            old=tensor([r['logp'] for r in records]);joint_old=tensor([r['joint_logp'] for r in records])
            target=tensor([r['target'] for r in records]);adv=target-tensor([r['value'] for r in records])
            adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8);actor=getattr(self,role);steps=0
            for _ in range(epochs):
                distribution=actor.distribution(features)
                logp=selected_log_prob(distribution,actions,indices)
                full=distribution.log_prob(actions).sum(-1)-joint_old
                # Full ROLE KL guard, even though the loss samples one score.
                if float((full.exp()-1-full).mean().detach())>.02:break
                loss=-clipped_terms(logp,old,adv).mean()
                optimizer=self.actor_optimizers[role];optimizer.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(),.5);optimizer.step();steps+=1
            with torch.no_grad():
                full=actor.distribution(features).log_prob(actions).sum(-1)-joint_old
                metrics[role+'_kl']=float((full.exp()-1-full).mean())
            metrics[role+'_actor_steps']=steps
            batches[role]=(tensor([r['state'] for r in records]),target)
        self.value_norm.update([target for _,target in batches.values()])
        for _ in range(epochs):
            loss=.25*sum((self.critic(role)(states).squeeze(-1)-self.value_norm.normalize(target)).square().mean()
                        for role,(states,target) in batches.items())
            self.critic_optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.critics.parameters(),.5)
            self.critic_optimizer.step()
        metrics['critic_loss']=float(loss.detach());metrics['value_norm_updates']=int(self.value_norm.updates)
        if not all(np.isfinite(value) for value in metrics.values()):raise FloatingPointError('nonfinite coordinate update')
        return metrics

    def checkpoint(self):
        result=super().checkpoint();result.update(variant='coordinate_reference_mappo_prototype_v1',
            coordinate_sampling='uniform_sku',critic_target='coordinate_reference_mc');return result

    def restore(self,data):
        if data['variant']!='coordinate_reference_mappo_prototype_v1' or data['coordinate_sampling']!='uniform_sku':
            raise ValueError('coordinate reference checkpoint required')
        if data['critic_target']!='coordinate_reference_mc':raise ValueError('coordinate MC target required')
        super().restore(dict(data,variant='neural_residual_mc_mappo_v1',critic_target='mc'))
