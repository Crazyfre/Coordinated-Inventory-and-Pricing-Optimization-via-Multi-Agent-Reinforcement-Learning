"""B4 IPPO: independent local actor/critics, optimizers and advantages.

Agents interact in the same shared-reward environment; neither optimizer sees
another agent's parameters, observations or global critic. This is simultaneous
independent learning, not pretraining against frozen heuristic counterparts.
"""
import numpy as np
import torch
from src.algos.ppo import MultiDiscreteActorCritic, PPO, compute_gae
from src.algos.mappo import interval_gae
from src.algos.stability import SignedLog
from src.algos.pricing_schema import pricing_schema, pricing_action, validate_checkpoint_schema
from src.env.interface import JointAction


class IndependentPPO:
    def __init__(self, env, hidden=64, lr=3e-4, epochs=4, reward_scale=100., lam=.95, target_kl=.02):
        if not np.isfinite(reward_scale) or reward_scale <= 0 or epochs < 1:
            raise ValueError('reward_scale and epochs must be positive')
        self.env = env
        self.action_schema = pricing_schema(env)
        obs,_ = env.reset(seed=0)
        self.pricing = torch.nn.Sequential(SignedLog(),MultiDiscreteActorCritic(
            len(obs['pricing']),self.action_schema['pricing_nvec'],hidden))
        self.replenishment = torch.nn.Sequential(SignedLog(),MultiDiscreteActorCritic(
            len(obs['replenishment']),env.action_space.nvec,hidden))
        self.p_ppo, self.r_ppo = PPO(self.pricing,lr=lr), PPO(self.replenishment,lr=lr)
        self.reward_scale, self.epochs, self.lam, self.target_kl = reward_scale, epochs, lam, target_kl

    def pricing_action(self,obs,greedy=True):
        return pricing_action(self.p_ppo.act(obs,greedy)[0], self.action_schema)

    def order_action(self,env,obs,greedy=True):
        return self.r_ppo.act(obs,greedy)[0].reshape(-1)

    def select_action(self,greedy=False):
        obs = self.env.observations()
        pa = self.pricing_action(obs['pricing'],greedy) if self.env.pricing_due else None
        if pa is not None:
            self.env.select_price(pa)
        ra = self.order_action(self.env,self.env.replenishment_observation(),greedy)
        return JointAction(None,ra)

    def save(self,path):
        torch.save({'pricing':self.pricing.state_dict(),'replenishment':self.replenishment.state_dict(),
            'p_optimizer':self.p_ppo.optimizer.state_dict(),'r_optimizer':self.r_ppo.optimizer.state_dict(),
            'reward_scale':self.reward_scale,'action_schema':self.action_schema},path)

    def load(self,path):
        data = torch.load(path,map_location='cpu',weights_only=True)
        validate_checkpoint_schema(data, self.action_schema)
        self.pricing.load_state_dict(data['pricing'])
        self.replenishment.load_state_dict(data['replenishment'])
        self.p_ppo.optimizer.load_state_dict(data['p_optimizer'])
        self.r_ppo.optimizer.load_state_dict(data['r_optimizer'])
        self.reward_scale = data['reward_scale']

    def _update(self,ppo,rows,adv,ret):
        obs = torch.tensor(np.stack([r[0] for r in rows]),dtype=torch.float32)
        actions = torch.tensor(np.stack([r[1] for r in rows]),dtype=torch.long)
        oldlp = torch.tensor([r[2] for r in rows],dtype=torch.float32)
        oldv = torch.tensor([r[3] for r in rows],dtype=torch.float32)
        advantage = torch.tensor((adv-adv.mean())/(adv.std()+1e-8),dtype=torch.float32)
        targets = torch.tensor(ret,dtype=torch.float32)
        pi_loss = v_loss = torch.tensor(0.)
        for _ in range(self.epochs):
            lp,value,entropy = ppo.evaluate(obs,actions)
            ratio = (lp-oldlp).exp()
            kl = ((ratio-1)-(lp-oldlp)).mean().item()
            if self.target_kl is not None and kl > self.target_kl:
                break
            pi_loss = -torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
            clipped = oldv+(value-oldv).clamp(-.2,.2)
            v_loss = .5*torch.maximum((value-targets)**2,(clipped-targets)**2).mean()
            loss = pi_loss+.5*v_loss-.01*entropy.mean()
            ppo.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo.net.parameters(),.5)
            ppo.optimizer.step()
        with torch.no_grad():
            final_lp, _, final_entropy = ppo.evaluate(obs,actions)
            final_ratio = (final_lp-oldlp).exp()
            kl = ((final_ratio-1)-(final_lp-oldlp)).mean().item()
        return {'pi_loss':pi_loss.item(),'v_loss':v_loss.item(),'kl':kl,'entropy':final_entropy.mean().item()}

    def train(self,total_updates,seed=0):
        torch.manual_seed(seed)
        history = []
        for update in range(total_updates):
            obs,_ = self.env.reset(seed=seed+update)
            p_rows, r_rows, rewards, starts = [], [], [], []
            done = False
            while not done:
                if self.env.pricing_due:
                    pa,lp,v = self.p_ppo.act(obs['pricing'])
                    p_rows.append((obs['pricing'].copy(),pa,lp,v))
                    starts.append(len(rewards))
                    self.env.select_price(pricing_action(pa, self.action_schema))
                robs = self.env.replenishment_observation()
                ra,lp,v = self.r_ppo.act(robs)
                r_rows.append((robs.copy(),ra,lp,v))
                obs,reward,done,_ = self.env.step(JointAction(None,ra))
                rewards.append(reward/self.reward_scale)
            dones = np.zeros(len(rewards)); dones[-1] = 1
            r_adv,r_ret = compute_gae(rewards,[r[3] for r in r_rows],dones,0,self.env.config.rl.gamma,self.lam)
            p_adv,p_ret = interval_gae(rewards,[r[3] for r in p_rows],starts,self.env.config.rl.gamma,self.lam)
            metrics = {}
            for name,ppo,rows,adv,ret in [('pricing',self.p_ppo,p_rows,p_adv,p_ret),
                                        ('replenishment',self.r_ppo,r_rows,r_adv,r_ret)]:
                metrics.update({name+'_'+k:v for k,v in self._update(ppo,rows,adv,ret).items()})
            history.append({**metrics,'update':update,'daily_steps':len(rewards),
                'pricing_decisions':len(p_rows),'episode_reward':float(sum(rewards)*self.reward_scale)})
        return history
