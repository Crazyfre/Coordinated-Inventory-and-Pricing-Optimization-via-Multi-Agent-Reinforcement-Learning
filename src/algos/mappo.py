"""Heterogeneous MAPPO: separate local actors, one global-state critic.

Full-episode on-policy updates avoid rollout cuts inside held-price blocks.
Pricing uses discounted interval rewards and gamma**duration/lam**duration
GAE, including a short final week. Actor execution never receives global state.
Reward scaling is numerical only; evaluation reports canonical unscaled reward.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from src.algos.ppo import mlp, orthogonal_init, compute_gae
from src.env.interface import JointAction


def interval_gae(rewards, values, starts, gamma=.99, lam=.95):
    """Variable-duration GAE for a complete finite-horizon episode."""
    adv = np.zeros(len(starts),dtype=np.float32)
    tail = 0.
    for j in reversed(range(len(starts))):
        start = starts[j]
        end = starts[j+1] if j+1 < len(starts) else len(rewards)
        duration = end-start
        interval = sum(gamma**k*rewards[start+k] for k in range(duration))
        nv = values[j+1] if j+1 < len(starts) else 0.
        delta = interval+gamma**duration*nv-values[j]
        tail = delta+(gamma*lam)**duration*tail
        adv[j] = tail
    return adv, adv+np.asarray(values,dtype=np.float32)


class Actor(nn.Module):
    def __init__(self, obs_dim, nvec, hidden):
        super().__init__()
        self.body = mlp([obs_dim, hidden, hidden])
        self.heads = nn.ModuleList([nn.Linear(hidden, int(n)) for n in nvec])
        self.apply(orthogonal_init)
        for h in self.heads:
            nn.init.orthogonal_(h.weight, 0.01)

    def forward(self, obs):
        h = self.body(obs)
        return [Categorical(logits=head(h)) for head in self.heads]

    def stats(self, obs, actions):
        ds = self(obs)
        return (sum(d.log_prob(actions[:,i]) for i,d in enumerate(ds)),
                sum(d.entropy() for d in ds))

    @torch.no_grad()
    def act(self, obs, greedy=False):
        ds = self(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        a = torch.stack([d.probs.argmax(-1) if greedy else d.sample() for d in ds], -1)
        lp = sum(d.log_prob(a[:,i]) for i,d in enumerate(ds))
        return a.squeeze(0).numpy(), lp.item()


class HeterogeneousMAPPO:
    def __init__(self, env, hidden=64, lr=3e-4, epochs=4, reward_scale=100., lam=.95):
        self.env = env
        obs, _ = env.reset(seed=0)
        self.pricing = Actor(len(obs['pricing']), [len(env.config.pricing.multipliers)], hidden)
        self.replenishment = Actor(len(obs['replenishment']), env.action_space.nvec, hidden)
        self.critic = mlp([len(env.get_state()), hidden, hidden, 1])
        self.critic.apply(orthogonal_init)
        self.optimizers = [torch.optim.Adam(g.parameters(), lr=lr)
                           for g in [self.pricing,self.replenishment,self.critic]]
        self.gamma = env.config.rl.gamma
        self.lam, self.epochs, self.reward_scale = lam, epochs, reward_scale
        if reward_scale <= 0:
            raise ValueError('reward_scale must be positive')

    @torch.no_grad()
    def value(self, state):
        return self.critic(torch.as_tensor(state, dtype=torch.float32)).item()

    def save(self, path):
        torch.save({'pricing':self.pricing.state_dict(),
                    'replenishment':self.replenishment.state_dict(),
                    'critic':self.critic.state_dict(),
                    'optimizers':[o.state_dict() for o in self.optimizers]},path)

    def load(self, path):
        data = torch.load(path,map_location='cpu',weights_only=True)
        for name in ['pricing','replenishment','critic']:
            getattr(self,name).load_state_dict(data[name])
        for o,s in zip(self.optimizers,data['optimizers']):
            o.load_state_dict(s)

    def train(self, total_updates, seed=0):
        torch.manual_seed(seed)
        metrics = []
        for update in range(total_updates):
            obs, _ = self.env.reset(seed=seed+update)
            r_rows, p_rows, rewards = [], [], []
            done = False
            while not done:
                if self.env.pricing_due:
                    state = self.env.get_state()
                    pa, plp = self.pricing.act(obs['pricing'])
                    p_rows.append((obs['pricing'].copy(), pa, plp, self.value(state),
                                   state.copy(), len(rewards)))
                    self.env.select_price(int(pa[0]))
                robs = self.env.replenishment_observation()
                state = self.env.get_state()
                ra, rlp = self.replenishment.act(robs)
                r_rows.append((robs,ra,rlp,self.value(state),state))
                obs, reward, done, _ = self.env.step(JointAction(None,ra))
                rewards.append(reward/self.reward_scale)
            rewards = np.asarray(rewards)
            values = np.array([x[3] for x in r_rows])
            dones = np.zeros(len(rewards)); dones[-1] = 1
            r_adv, r_ret = compute_gae(rewards,values,dones,0,self.gamma,self.lam)
            p_adv, p_ret = interval_gae(rewards,[x[3] for x in p_rows],
                                      [x[5] for x in p_rows],self.gamma,self.lam)
            losses = []
            for actor, optimizer, rows, adv in zip(
                    [self.pricing,self.replenishment],self.optimizers[:2],
                    [p_rows,r_rows],[p_adv,r_adv]):
                o = torch.tensor(np.stack([x[0] for x in rows]),dtype=torch.float32)
                a = torch.tensor(np.stack([x[1] for x in rows]),dtype=torch.long)
                old = torch.tensor([x[2] for x in rows],dtype=torch.float32)
                advantage = torch.tensor((adv-adv.mean())/(adv.std()+1e-8))
                for _ in range(self.epochs):
                    lp, entropy = actor.stats(o,a)
                    ratio = (lp-old).exp()
                    loss = -torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()-.01*entropy.mean()
                    optimizer.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(),.5); optimizer.step()
                losses.append(loss.item())
            states = torch.tensor(np.stack([x[4] for x in r_rows]+[x[4] for x in p_rows]),dtype=torch.float32)
            targets = torch.tensor(np.concatenate([r_ret,p_ret]),dtype=torch.float32)
            oldv = torch.tensor([x[3] for x in r_rows]+[x[3] for x in p_rows],dtype=torch.float32)
            for _ in range(self.epochs):
                v = self.critic(states).squeeze(-1)
                clipped = oldv+(v-oldv).clamp(-.2,.2)
                loss = .5*torch.maximum((v-targets)**2,(clipped-targets)**2).mean()
                self.optimizers[2].zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),.5); self.optimizers[2].step()
            metrics.append({'pricing_loss':losses[0],'replenishment_loss':losses[1],
                            'critic_loss':loss.item(),'daily_steps':len(rewards),
                            'pricing_decisions':len(p_rows),'episode_reward':float(rewards.sum()*self.reward_scale)})
        return metrics
