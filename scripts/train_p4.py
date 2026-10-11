"""P4 development MAPPO training/evaluation on calibrated joint daily dynamics."""
import argparse
import csv
import json
from pathlib import Path
import torch
from src.algos.mappo import HeterogeneousMAPPO
from src.experiments.phase4 import make_env, evaluate, heuristic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates',type=int,default=20)
    p.add_argument('--days',type=int,default=28)
    p.add_argument('--episodes',type=int,default=4)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--capacity',type=float,default=500)
    p.add_argument('--output',default='outputs/p4')
    args = p.parse_args()
    if min(args.updates,args.days,args.episodes) < 1:
        p.error('updates, days and episodes must be positive')
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    env = make_env(args.days,args.capacity)
    agent = HeterogeneousMAPPO(env)
    history = agent.train(args.updates,args.seed)
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    agent.save(str(out/'checkpoint.pt'))
    # Exercise actual checkpoint restoration before greedy evaluation.
    agent.load(str(out/'checkpoint.pt'))
    summary = {'mappo':evaluate(env,lambda e,o: agent.replenishment.act(o,True)[0],
                               args.episodes,args.seed+1000,
                               lambda o:agent.pricing.act(o,True)[0][0])}
    for i,m in enumerate(env.config.pricing.multipliers):
        summary[f'ss_price_{m:.2f}'] = evaluate(
            env,lambda e,o:heuristic(e,'ss'),args.episodes,args.seed+1000,lambda o,i=i:i)
    for row in summary.values():
        row.pop('demands')
    with (out/'training.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'run_config.json').write_text(json.dumps(vars(args),indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
