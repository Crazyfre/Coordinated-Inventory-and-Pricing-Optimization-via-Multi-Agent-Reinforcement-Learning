"""R3 development PPO vs paired replenishment baselines; no superiority gate."""
import argparse
import csv
import json
from pathlib import Path
import torch
from src.agents.replenishment_agent import ReplenishmentPPOAgent
from src.experiments.phase4 import make_env, evaluate, heuristic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates',type=int,default=20)
    p.add_argument('--days',type=int,default=28)
    p.add_argument('--episodes',type=int,default=4)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--capacity',type=float,default=500)
    p.add_argument('--output',default='outputs/r3')
    args = p.parse_args()
    if min(args.updates,args.days,args.episodes) < 1:
        p.error('updates, days and episodes must be positive')
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    env = make_env(args.days,args.capacity,joint=False)
    agent = ReplenishmentPPOAgent(env,rollout_steps=128)
    history = agent.train(args.updates,args.seed)
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    agent.save(str(out/'checkpoint.pt'))
    summary = {}
    for name in ['ppo','zero','constant','ss']:
        policy = (lambda e,o: agent.act(o,greedy=True)[0]) if name == 'ppo' else (
            lambda e,o,kind=name: heuristic(e,kind))
        result = evaluate(env,policy,args.episodes,args.seed+1000)
        result.pop('demands'); summary[name] = result
    with (out/'training.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'run_config.json').write_text(json.dumps(vars(args),indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
