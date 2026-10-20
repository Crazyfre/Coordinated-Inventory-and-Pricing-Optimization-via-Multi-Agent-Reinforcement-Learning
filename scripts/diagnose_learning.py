"""Bounded deterministic diagnostics, NOT research results or altered production defaults."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.config.locked_config import get_locked_config
from src.env.replenishment_env import ReplenishmentEnv
from src.agents.replenishment_agent import ReplenishmentPPOAgent


def make_fixture(q_max=10, lead=2):
    c = get_locked_config()
    c = replace(c, env=replace(c.env, n_skus=1),
                replenishment=replace(c.replenishment, q_max=q_max),
                lead_time=replace(c.lead_time, min_days=lead, max_days=lead))
    return ReplenishmentEnv(c, episode_length_days=28,
        demand_source=lambda day: np.array([5.]), initial_inventory=10., capacity=30.)


def evaluate(env, policy):
    obs, _ = env.reset(seed=10000)
    rewards, actions, sales, demand = [], [], 0., 0.
    done = False
    while not done:
        action = policy(obs)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        rewards.append(reward)
        actions.append(int(action[0]))
        sales += np.asarray(info['sales']).sum()
        demand += np.asarray(info['demand']).sum()
    return dict(daily_reward=float(np.mean(rewards)), fill=float(sales/demand), actions=actions)


def run(updates=64, q_max=10, lead=2, seed=11):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    env = make_fixture(q_max, lead)
    agent = ReplenishmentPPOAgent(env, hidden=16, rollout_steps=256,
        epochs_per_update=4, normalize_inputs=True, reward_scale=10.)
    initial = evaluate(env, lambda obs: agent.act(obs, greedy=True)[0])
    start = time.perf_counter()
    curve = agent.train(updates, seed=seed)
    elapsed = time.perf_counter()-start
    final = evaluate(env, lambda obs: agent.act(obs, greedy=True)[0])
    reference = evaluate(env, lambda obs: np.array([1]))
    return dict(seed=seed, updates=updates, transitions=updates*256, seconds=elapsed,
        fixture=dict(skus=1, deterministic_demand=5, fixed_price=10, lead=lead,
                     q_max=q_max, capacity=30, diagnostic_only=True),
        initial=initial, final=final, constant_five=reference, curve=curve,
        passed=final['fill'] >= .95 and final['daily_reward'] >= reference['daily_reward']*.95)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--updates', type=int, default=64)
    p.add_argument('--q-max', type=int, default=10)
    p.add_argument('--lead', type=int, default=2)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--output', default='outputs/learning_diagnosis/reduced.json')
    args = p.parse_args()
    result = run(args.updates, args.q_max, args.lead, args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'curve'}, indent=2))
