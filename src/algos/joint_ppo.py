"""B5 single centralized actor/critic with cadence-masked joint action."""
import numpy as np
import torch
from src.algos.ppo import MultiDiscreteActorCritic
from src.algos.ppo import compute_gae
from src.algos.stability import SignedLog
from src.algos.pricing_schema import pricing_schema, pricing_action, validate_checkpoint_schema
from src.env.interface import JointAction


class JointPPO:
    def __init__(self, env, hidden=64, lr=3e-4, epochs=4, reward_scale=100., lam=.95, target_kl=.02):
        self.env = env
        self.action_schema = pricing_schema(env)
        self.n_price_heads = len(self.action_schema["pricing_nvec"])
        env.reset(seed=0)
        nvec = [*self.action_schema["pricing_nvec"], *env.action_space.nvec]
        self.net = MultiDiscreteActorCritic(len(env.get_state())+1,nvec,hidden)
        self.transform = SignedLog()
        if not np.isfinite(reward_scale) or reward_scale <= 0 or epochs < 1:
            raise ValueError('reward_scale and epochs must be positive')
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=lr)
        self.reward_scale, self.epochs, self.lam, self.target_kl = reward_scale, epochs, lam, target_kl
        self._cached_order = None

    def observation(self):
        return np.concatenate([self.env.get_state(),[float(self.env.pricing_due)]]).astype(np.float32)

    def stats(self, obs, actions, due):
        dists, value = self.net(self.transform(obs))
        logp = sum(dists[i].log_prob(actions[:,i]) for i in range(self.n_price_heads))*due
        entropy = sum(dists[i].entropy() for i in range(self.n_price_heads))*due
        for i,dist in enumerate(dists[self.n_price_heads:],self.n_price_heads):
            logp = logp + dist.log_prob(actions[:,i])
            entropy = entropy + dist.entropy()
        return logp, entropy, value

    @torch.no_grad()
    def act(self, greedy=False):
        x = torch.tensor(self.observation()).unsqueeze(0)
        dists, value = self.net(self.transform(x))
        actions = torch.stack([d.probs.argmax(-1) if greedy else d.sample() for d in dists],-1)
        lp, _, _ = self.stats(x,actions,torch.tensor([float(self.env.pricing_due)]))
        return actions[0].numpy(), lp.item(), value.item()

    def pricing_action(self, obs=None, greedy=True):
        action, _, _ = self.act(greedy)
        self._cached_order = action[self.n_price_heads:].copy()
        return pricing_action(action[:self.n_price_heads], self.action_schema)

    def order_action(self, env, obs=None, greedy=True):
        if env is not self.env:
            raise ValueError('JointPPO evaluation callbacks must use its bound env')
        if self._cached_order is not None:
            action, self._cached_order = self._cached_order, None
            return action
        return self.act(greedy)[0][self.n_price_heads:]

    def select_action(self, greedy=False):
        action, _, _ = self.act(greedy)
        return JointAction(pricing_action(action[:self.n_price_heads], self.action_schema) if self.env.pricing_due else None,action[self.n_price_heads:])

    def save(self,path):
        torch.save({'net':self.net.state_dict(),'optimizer':self.optimizer.state_dict(),
                    'reward_scale':self.reward_scale,'action_schema':self.action_schema},path)

    def load(self,path):
        data = torch.load(path,map_location='cpu',weights_only=True)
        validate_checkpoint_schema(data, self.action_schema)
        self.net.load_state_dict(data['net'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.reward_scale = data['reward_scale']
        self._cached_order = None

    def train(self,total_updates,seed=0):
        torch.manual_seed(seed)
        history = []
        for update in range(total_updates):
            self.env.reset(seed=seed+update)
            rows, rewards, done = [], [], False
            while not done:
                x, due = self.observation(), float(self.env.pricing_due)
                action, lp, value = self.act()
                rows.append((x,action,lp,value,due))
                _, reward, done, _ = self.env.step(JointAction(pricing_action(action[:self.n_price_heads], self.action_schema) if due else None,action[self.n_price_heads:]))
                rewards.append(reward/self.reward_scale)
            dones = np.zeros(len(rows)); dones[-1] = 1
            adv, ret = compute_gae(rewards,[r[3] for r in rows],dones,0,self.env.config.rl.gamma,self.lam)
            x = torch.tensor(np.stack([r[0] for r in rows]),dtype=torch.float32)
            a = torch.tensor(np.stack([r[1] for r in rows]),dtype=torch.long)
            oldlp = torch.tensor([r[2] for r in rows],dtype=torch.float32)
            oldv = torch.tensor([r[3] for r in rows],dtype=torch.float32)
            due = torch.tensor([r[4] for r in rows],dtype=torch.float32)
            advantage = torch.tensor((adv-adv.mean())/(adv.std()+1e-8),dtype=torch.float32)
            targets = torch.tensor(ret,dtype=torch.float32)
            pi_loss = v_loss = torch.tensor(0.)
            for _ in range(self.epochs):
                lp, entropy, value = self.stats(x,a,due)
                ratio = (lp-oldlp).exp()
                kl = ((ratio-1)-(lp-oldlp)).mean().item()
                if self.target_kl is not None and kl > self.target_kl:
                    break
                pi_loss = -torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
                clipped = oldv+(value-oldv).clamp(-.2,.2)
                v_loss = .5*torch.maximum((value-targets)**2,(clipped-targets)**2).mean()
                loss = pi_loss+.5*v_loss-.01*entropy.mean()
                self.optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),.5)
                self.optimizer.step()
            with torch.no_grad():
                final_lp, final_entropy, _ = self.stats(x,a,due)
                final_ratio = (final_lp-oldlp).exp()
                kl = ((final_ratio-1)-(final_lp-oldlp)).mean().item()
            history.append({'update':update,'pi_loss':pi_loss.item(),'v_loss':v_loss.item(),
                'kl':kl,'entropy':final_entropy.mean().item(),'daily_steps':len(rows),
                'pricing_decisions':int(due.sum().item()),'episode_reward':float(sum(rewards)*self.reward_scale)})
        return history
