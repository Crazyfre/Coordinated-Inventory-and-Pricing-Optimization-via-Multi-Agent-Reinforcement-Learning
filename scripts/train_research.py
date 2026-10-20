"""Reproducible CPU development training, separate from final held-out research.

python -m scripts.train_research --methods r3 mappo independent joint --seeds 11 22 33 44 55
Uses existing development env (calendar start=1800, calibrated NB/shocks .15,
capacity=500, procurement=.6*base prices). No final-test outcomes used to tune.
"""
import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from src.agents.replenishment_agent import ReplenishmentPPOAgent
from src.algos.mappo import HeterogeneousMAPPO
from src.algos.joint_ppo import JointPPO
from src.algos.independent_ppo import IndependentPPO
from src.baselines.retail import EOQPolicy, MyopicPricingPolicy
from src.experiments.phase4 import make_env, evaluate, heuristic


def build_agent(method, env, stable=True):
    if method == 'r3':
        return ReplenishmentPPOAgent(env,rollout_steps=128,normalize_inputs=stable,
                                     reward_scale=100. if stable else 1.)
    if method == 'mappo':
        return HeterogeneousMAPPO(env,normalize_inputs=stable,target_kl=.02 if stable else None)
    if method == 'independent':
        return IndependentPPO(env)
    if method == 'joint':
        return JointPPO(env)
    raise ValueError('unknown method')


def callbacks(method, agent):
    if method == 'r3':
        return (lambda e,o:agent.act(o,True)[0]),None
    if method == 'mappo':
        return (lambda e,o:agent.replenishment.act(o,True)[0]),agent.pricing_action
    return agent.order_action,agent.pricing_action


def parameters(agent,method):
    if method in ['r3','joint']:
        return list(agent.net.parameters())
    modules = [agent.pricing,agent.replenishment]
    if method == 'mappo':
        modules.append(agent.critic)
    return [p for module in modules for p in module.parameters()]


def measure(env,method,agent,episodes):
    order,price = callbacks(method,agent)
    result = evaluate(env,order,episodes,10000,price)
    result.pop('demands')
    return result


def run_training(method, seed, updates, days, episodes, output, capacity=500., stable=True):
    if min(updates,days,episodes) < 1:
        raise ValueError('updates, days, episodes must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    config = dict(method=method,seed=seed,updates=updates,days=days,episodes=episodes,capacity=capacity,
                  stable=stable,reward_scale=100. if stable or method!='r3' else 1.,
                  input_transform='signed_log' if stable or method in ['joint','independent'] else 'identity',
                  torch_threads=1,evaluation_seed=10000,calendar_start_day_index=1800,
                  shock_std=.15,procurement_fraction=.6,python=platform.python_version(),torch=str(torch.__version__),
                  protocol='development only; no held-out final evaluation',
                  pricing_scope='shared')
    config['git_revision'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    (output/'config.json').write_text(json.dumps(config,indent=2))
    env = make_env(days,capacity,joint=method!='r3')
    from src.algos.pricing_schema import pricing_schema
    config["action_schema"] = pricing_schema(env)
    (output/'config.json').write_text(json.dumps(config,indent=2))
    agent = build_agent(method,env,stable)
    before = [p.detach().clone() for p in parameters(agent,method)]
    initial = measure(env,method,agent,episodes)
    start = time.perf_counter()
    history = agent.train(updates,seed)
    elapsed = time.perf_counter()-start
    if not all(np.isfinite(v) for row in history for v in row.values()):
        raise FloatingPointError('nonfinite training metric')
    change = sum(float((p.detach()-b).square().sum()) for p,b in zip(parameters(agent,method),before))**.5
    agent.save(output/'checkpoint.pt')
    final = measure(env,method,agent,episodes)
    fresh = build_agent(method,env,stable)
    fresh.load(output/'checkpoint.pt')
    roundtrip = measure(env,method,fresh,episodes)
    if final != roundtrip:
        raise AssertionError('checkpoint greedy evaluation mismatch')
    result = dict(initial_evaluation=initial,evaluation=final,
                  wall_seconds=elapsed,parameter_l2_change=change,
                  training_env_steps=updates*(128 if method=='r3' else days),
                  checkpoint_roundtrip_verified=True,
                  service_target_met=final['fill_rate']>=env.config.reward.service_target_fill_rate,
                  final_training_metrics=history[-1])
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    with (output/'training.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',nargs='+',choices=['r3','mappo','independent','joint'],default=['r3','mappo','independent','joint'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55])
    parser.add_argument('--updates',type=int,default=128)
    parser.add_argument('--days',type=int,default=28)
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--capacity',type=float,default=500.)
    parser.add_argument('--output',default='outputs/training_research/development')
    parser.add_argument('--legacy',action='store_true',help='disable numerical changes for R3/MAPPO comparison only')
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    results = {}
    for method in args.methods:
        results[method] = []
        for seed in args.seeds:
            row = run_training(method,seed,args.updates,args.days,args.episodes,
                out/method/f'seed_{seed}',args.capacity,not args.legacy)
            results[method].append(row)
            print(json.dumps({'method':method,'seed':seed,'seconds':row['wall_seconds'],
                'daily_reward':row['evaluation']['mean_daily_reward'],'fill_rate':row['evaluation']['fill_rate']}),flush=True)
    env = make_env(args.days,args.capacity)
    baseline = {}
    for i,m in enumerate(env.config.pricing.multipliers):
        baseline[f'ss_price_{m:.2f}'] = evaluate(env,lambda e,o:heuristic(e,'ss'),args.episodes,10000,lambda o,i=i:i)
    baseline['eoq_zero_setup_price_1.00'] = evaluate(env,EOQPolicy(),args.episodes,10000)
    baseline['forecast_myopic_ss'] = evaluate(env,lambda e,o:heuristic(e,'ss'),args.episodes,10000,MyopicPricingPolicy(env))
    for row in baseline.values():
        row.pop('demands')
    (out/'baselines.json').write_text(json.dumps(baseline,indent=2))
    aggregate = {}
    for method,rows in results.items():
        rewards = [r['evaluation']['mean_daily_reward'] for r in rows]
        aggregate[method] = dict(n_training_seeds=len(rows),mean_daily_reward=float(np.mean(rewards)),
            std_daily_reward_across_training_seeds=float(np.std(rewards,ddof=1)) if len(rows)>1 else None,
            mean_fill_rate=float(np.mean([r['evaluation']['fill_rate'] for r in rows])),
            total_training_wall_seconds=sum(r['wall_seconds'] for r in rows),
            total_training_env_steps=sum(r['training_env_steps'] for r in rows),
            all_checkpoints_verified=all(r['checkpoint_roundtrip_verified'] for r in rows))
    (out/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
    print(json.dumps({'aggregate':aggregate,'baselines':baseline},indent=2),flush=True)


if __name__ == '__main__':
    main()
