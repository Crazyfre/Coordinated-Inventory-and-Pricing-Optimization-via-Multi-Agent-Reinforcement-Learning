"""Joint structured PPO actors with a phase-aware centralized critic.

Actors see only role observations plus the known day. Ordering PPO operates on
Gaussian latent covers; the deterministic grid mapping is part of the policy.
The greedy initialization exactly matches the structured ordering anchor and
price index4. Stochastic exploration is deliberately not the frozen baseline.
"""
import time
import numpy as np
import torch
from torch import nn
from src.algos.mappo import interval_gae
from src.algos.structured_ordering_ppo import StructuredOrderingActor,ordering_features,execute_cover
from src.algos.structured_pricing_ppo import StructuredPricingActor,pricing_features,critic_features
from src.env.interface import JointAction


class StructuredMAPPO(nn.Module):
    def __init__(self, env, covers, seed=0, *, critic_mode='central', actor_lr=.003,critic_target='gae'):
        super().__init__()
        if critic_mode not in ('central','local'):raise ValueError('central or local critic required')
        if critic_target not in ('gae','mc'):raise ValueError('gae or mc critic target required')
        self.critic_target=critic_target
        torch.manual_seed(seed)
        self.mode=critic_mode
        self.pricing=StructuredPricingActor()
        with torch.no_grad():
            self.pricing.weight[:,0,:]=torch.log(torch.tensor([.005,.005,.005,.005,.98]))
        self.ordering=StructuredOrderingActor(covers)
        obs,_=env.reset(seed=0)
        widths=({'shared':len(env.get_state())+3} if self.mode=='central' else
                {'pricing':len(obs['pricing'])+1,'ordering':len(obs['replenishment'])+1})
        self.critics=nn.ModuleDict({k:nn.Sequential(nn.Linear(w,64),nn.Tanh(),nn.Linear(64,1)) for k,w in widths.items()})
        for critic in self.critics.values():
            nn.init.zeros_(critic[-1].weight);nn.init.zeros_(critic[-1].bias)
        self.actor_optimizers={k:torch.optim.Adam(getattr(self,k).parameters(),lr=actor_lr) for k in ('pricing','ordering')}
        self.critic_optimizer=torch.optim.Adam(self.critics.parameters(),lr=.001)
        self.gamma=float(env.config.rl.gamma)

    def critic_input(self,env,obs,day,role):
        if self.mode=='central':
            return np.r_[critic_features(env.get_state(),day),
                         [1.,0.] if role=='pricing' else [0.,1.]].astype(np.float32)
        return critic_features(obs,day)

    def critic(self,role):return self.critics['shared' if self.mode=='central' else role]

    @torch.no_grad()
    def value(self,role,state):return float(self.critic(role)(torch.as_tensor(state)).squeeze())

    def collect(self,env,key,*,greedy=False,frozen=(),training=True,deadline=float('inf')):
        if not set(frozen)<= {'pricing','ordering'}:raise ValueError('unknown frozen actor')
        env.reset(seed=key)
        rows={'pricing':[],'ordering':[]};rewards=[];trace=[]
        for day in range(1,29):
            if time.perf_counter()>deadline:raise TimeoutError('structured MAPPO deadline')
            price=None
            if env.pricing_due:
                obs=env.pricing_observation().copy()
                f=pricing_features(obs,day,env.base_price)
                price,lp=self.pricing.act(f,greedy or 'pricing' in frozen)
                if training:
                    state=self.critic_input(env,obs,day,'pricing')
                    rows['pricing'].append(dict(features=f,action=price.copy(),logp=lp,
                        state=state,value=self.value('pricing',state),start=len(rewards)))
                env.select_price(price)
            obs=env.replenishment_observation().copy()
            f=ordering_features(obs,day)
            latent,lp=self.ordering.act(f,greedy or 'ordering' in frozen)
            action=execute_cover(obs,latent)
            if training:
                state=self.critic_input(env,obs,day,'ordering')
                rows['ordering'].append(dict(features=f,action=latent.copy(),logp=lp,
                    state=state,value=self.value('ordering',state),start=len(rewards)))
            _,reward,done,info=env.step(JointAction(None,action))
            if done != (day==28):raise ValueError('complete28-day episodes required')
            rewards.append(float(reward))
            trace.append(dict(day=day,obs=obs,price_actions=price,actions=action,
                              reward=float(reward),info=info))
        return dict(rows=rows,rewards=rewards,trace=trace,frozen=list(frozen))

    def update(self,episodes,*,epochs=4):
        if not episodes:raise ValueError('nonempty on-policy batch required')
        frozen=set(episodes[0]['frozen'])
        if any(set(e['frozen'])!=frozen for e in episodes):raise ValueError('mixed frozen roles')
        batches={};metrics={}
        for role in ('pricing','ordering'):
            records=[];advantages=[];returns=[]
            for ep in episodes:
                r=ep['rows'][role]
                starts=[v['start'] for v in r]
                if starts != ([0,7,14,21] if role=='pricing' else list(range(28))):
                    raise ValueError('invalid decision cadence')
                a,t=interval_gae(np.asarray(ep['rewards'])/100.,[v['value'] for v in r],starts,self.gamma,.95)
                if self.critic_target=='mc':
                    t=np.asarray([sum(self.gamma**j*float(v)/100. for j,v in enumerate(ep['rewards'][start:])) for start in starts])
                records.extend(r);advantages.extend(a);returns.extend(t)
            tensor=lambda values:torch.as_tensor(np.asarray(values),dtype=torch.float32)
            f=tensor([r['features'] for r in records])
            actions=torch.as_tensor(np.asarray([r['action'] for r in records]),
                dtype=torch.long if role=='pricing' else torch.float32)
            old=tensor([r['logp'] for r in records]);adv=tensor(advantages)
            adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            actor=getattr(self,role);steps=0
            if role not in frozen:
                for _ in range(epochs):
                    lp,_=actor.stats(f,actions);logratio=lp-old;ratio=logratio.exp()
                    if float(((ratio-1)-logratio).mean().detach())>.02:break
                    loss=-torch.minimum(ratio*adv,ratio.clamp(.9,1.1)*adv).mean()
                    optimizer=self.actor_optimizers[role]
                    optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),.5)
                    optimizer.step();steps+=1
            with torch.no_grad():
                lp,_=actor.stats(f,actions);lr=lp-old
                metrics[role+'_kl']=float((lr.exp()-1-lr).mean())
            metrics[role+'_actor_steps']=steps
            batches[role]=(tensor([r['state'] for r in records]),tensor(returns))
            metrics[role+'_critic_target_mean']=float(np.mean(returns))
        for _ in range(epochs):
            losses=[(self.critic(role)(s).squeeze(-1)-t).square().mean() for role,(s,t) in batches.items()]
            loss=.25*sum(losses) # Equal role weight despite weekly/daily row counts.
            self.critic_optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(self.critics.parameters(),.5);self.critic_optimizer.step()
        metrics['critic_loss']=float(loss.detach())
        if not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError('nonfinite update')
        return metrics

    def checkpoint(self):
        return dict(variant='structured_mappo_v1',mode=self.mode,gamma=self.gamma,critic_target=self.critic_target,
            model=self.state_dict(),actors={k:v.state_dict() for k,v in self.actor_optimizers.items()},
            critic_optimizer=self.critic_optimizer.state_dict(),rng=torch.get_rng_state())

    def restore(self,data):
        if data['variant']!='structured_mappo_v1' or data['mode']!=self.mode or data['gamma']!=self.gamma:
            raise ValueError('checkpoint architecture mismatch')
        if data.get('critic_target','gae')!=self.critic_target:raise ValueError('checkpoint critic target mismatch')
        self.load_state_dict(data['model'])
        for k,v in self.actor_optimizers.items():v.load_state_dict(data['actors'][k])
        self.critic_optimizer.load_state_dict(data['critic_optimizer']);torch.set_rng_state(data['rng'])
