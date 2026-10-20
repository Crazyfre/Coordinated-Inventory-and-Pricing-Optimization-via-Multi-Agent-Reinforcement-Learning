"""Clean CPU development: random train windows, frozen train-only calibration,
validation-only policy evaluation, and strict cross-price CRN via sibling's env.

New runs default to per-SKU pricing. --pricing-scope shared retains the legacy
actor schema. The low-level research factory remains shared by default.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from src.demand.identifiable_calibration import validate_research_training_calibration
from src.env.interface import JointAction
from src.experiments.research import make_research_env, SPLITS
from scripts.train_research import build_agent, callbacks, parameters
from src.experiments.phase4 import evaluate


class TrainingWindows:
    """Transparent training-only wrapper; calendar start varies by episode seed."""
    def __init__(self,env,days):
        self.env, self.days, self.windows = env, days, []

    def __getattr__(self,name):
        return getattr(self.env,name)

    def reset(self,*,seed=None,options=None):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed or 0),314159]))
        lo,hi = SPLITS['train']
        start = int(rng.integers(lo,hi-self.days+2))
        self.env.sim.start_day_index = start
        self.windows.append(start)
        return self.env.reset(seed=seed,options=options)


class ReplenishmentOnly:
    """Gym R3 view of the clean joint env, holding base price across weeks."""
    def __init__(self,env):
        self.env = env

    def __getattr__(self,name):
        return getattr(self.env,name)

    def reset(self,*,seed=None,options=None):
        obs,info = self.env.reset(seed=seed,options=options)
        return obs['replenishment'],info

    def step(self,action):
        price = 2 if self.env.pricing_due else None
        obs,reward,done,info = self.env.step(JointAction(price,action))
        return obs['replenishment'],reward,done,False,info


def run_training(method,scenario,seed,updates,days,episodes,output,calibration, pricing_scope="per_sku"):
    calibration = Path(calibration)
    calibration_bytes = calibration.read_bytes()
    calibration_hash = hashlib.sha256(calibration_bytes).hexdigest()
    params = json.loads(calibration_bytes)
    validate_research_training_calibration(params)
    if min(updates,days,episodes) < 1 or days > 28:
        raise ValueError('positive counts and at most 28 days (validation window) required')
    if method not in ['r3','mappo','independent','joint'] or scenario not in ['S1','S2','S3']:
        raise ValueError('unsupported method or training stability scenario')
    torch.set_num_threads(1); torch.manual_seed(seed)
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    train_base = make_research_env(scenario,split='train',start=15,days=days,params=params,pricing_scope=pricing_scope)
    windows = TrainingWindows(train_base,days)
    train_env = ReplenishmentOnly(windows) if method=='r3' else windows
    validation = make_research_env(scenario,split='validation',start=1886,days=days,params=params,pricing_scope=pricing_scope)
    model = build_agent(method,train_env)
    before = [p.detach().clone() for p in parameters(model,method)]
    def measure(agent):
        order,price = callbacks(method,agent)
        row = evaluate(validation,order,episodes,10000,price)
        row.pop('demands')
        return row
    # JointPPO callbacks require the model to bind to the exact evaluated env.
    model.env = validation
    initial = measure(model)
    model.env = train_env
    start = time.perf_counter()
    history = model.train(updates,seed)
    elapsed = time.perf_counter()-start
    if not all(np.isfinite(x) for row in history for x in row.values()):
        raise FloatingPointError('nonfinite training metrics')
    change = sum(float((p.detach()-b).square().sum()) for p,b in zip(parameters(model,method),before))**.5
    model.save(output/'checkpoint.pt')
    model.env = validation
    final = measure(model)
    restored = build_agent(method,validation)
    restored.load(output/'checkpoint.pt')
    if final != measure(restored):
        raise AssertionError('clean checkpoint validation replay mismatch')
    from src.algos.pricing_schema import pricing_schema
    config = dict(pricing_scope=pricing_scope,action_schema=pricing_schema(train_base),
        calibration_status=params["status"],method=method,scenario=scenario,seed=seed,updates=updates,days=days,
        validation_episodes=episodes,validation_start=1886,validation_seed=10000,
        train_window_range=[15,1885-days+1],random_train_window_per_episode=True,
        calibration=str(calibration.resolve()),calibration_sha256=calibration_hash,
        held_out_test_used=False,strict_cross_price_crn=True,torch_threads=1,
        hidden=64,lr=.0003,epochs=4,reward_scale=100.,lam=.95,gamma=.99,
        clip_eps=.2,entropy_coef=.01,grad_norm=.5,target_kl=None if method=='r3' else .02,
        rollout_steps=128 if method=='r3' else None,
        torch=str(torch.__version__),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol='clean development; validation, NOT final held-out test')
    result = dict(initial_evaluation=initial,evaluation=final,wall_seconds=elapsed,
        training_env_steps=updates*(128 if method=='r3' else days),parameter_l2_change=change,
        training_windows=windows.windows,checkpoint_roundtrip_verified=True,held_out_test_used=False,
        service_target_met=final['fill_rate']>=.95,final_training_metrics=history[-1])
    (output/'config.json').write_text(json.dumps(config,indent=2))
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    with (output/'training.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='+',default=['mappo','independent','joint'],choices=['r3','mappo','independent','joint'])
    p.add_argument('--scenarios',nargs='+',default=['S1','S2','S3'],choices=['S1','S2','S3'])
    p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55])
    p.add_argument('--updates',type=int,default=64)
    p.add_argument('--days',type=int,default=28)
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--calibration',default='outputs/evaluation_research/calibration/train_only_nb.json')
    p.add_argument('--pricing-scope',choices=['shared','per_sku'],default='per_sku')
    p.add_argument('--output',default='outputs/training_research/clean_development')
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    aggregate = {}
    for method in args.methods:
        aggregate[method] = {}
        for scenario in args.scenarios:
            rows = []
            for seed in args.seeds:
                r = run_training(method,scenario,seed,args.updates,args.days,args.episodes,
                    out/method/scenario/f'seed_{seed}',args.calibration,args.pricing_scope)
                rows.append(r)
                print(json.dumps({'method':method,'scenario':scenario,'seed':seed,
                    'seconds':r['wall_seconds'],'daily_reward':r['evaluation']['mean_daily_reward'],
                    'fill_rate':r['evaluation']['fill_rate']}),flush=True)
            rewards = [r['evaluation']['mean_daily_reward'] for r in rows]
            aggregate[method][scenario] = dict(n_training_seeds=len(rows),updates_per_seed=args.updates,
                validation_episodes_per_checkpoint=args.episodes,mean_daily_reward=float(np.mean(rewards)),
                std_daily_reward_across_training_seeds=float(np.std(rewards,ddof=1)) if len(rows)>1 else None,
                mean_fill_rate=float(np.mean([r['evaluation']['fill_rate'] for r in rows])),
                total_training_env_steps=sum(r['training_env_steps'] for r in rows),
                total_training_seconds=sum(r['wall_seconds'] for r in rows),
                service_target_seed_count=sum(r['service_target_met'] for r in rows),
                all_checkpoints_verified=all(r['checkpoint_roundtrip_verified'] for r in rows))
            (out/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':
    main()
