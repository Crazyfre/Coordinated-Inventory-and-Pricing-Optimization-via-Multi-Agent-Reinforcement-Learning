"""Bounded clean paired operational replay (validation only)."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from src.env.interface import JointAction


def replay(env, order, price, seed, label):
    obs, _ = env.reset(seed=seed)
    rows = []
    done = False
    while not done:
        if env.pricing_due:
            env.select_price(int(price(obs['pricing'])))
        robs = env.replenishment_observation()
        b = env.observation_components()
        day = env.day
        inv = b['inventory'].copy()
        pipe = b['pipeline'].copy()
        p = env._pending_price if env._pending_price is not None else b['price']
        forecast = env.sim.mu_for(day,p)
        action = order(env,robs)
        obs, reward, done, info = env.step(JointAction(None,action))
        end = inv-info['sales']
        row = dict(label=label,seed=seed,day=day,inventory_start=inv.tolist(),
            inventory_position=(inv+pipe.sum(axis=1)).tolist(),pipeline=pipe.tolist(),
            forecast=forecast.tolist(),orders=env.grid[action].tolist(),
            inventory_end=end.tolist(),price=info['price'].tolist(),reward=reward,
            order_lead_times=np.where(np.asarray(action)>0,env._last_lead_times,0).tolist(),
            reward_terms=info['reward_terms'],terminal_pipeline=env.pipeline.outstanding().tolist() if done else None)
        for source,target in [('demand','demand'),('sales','sales'),('lost_sales','lost_sales'),
                              ('arrivals','arrivals'),('orders_delivered','accepted'),('orders_clipped','clipped')]:
            row[target] = info[source].tolist()
        rows.append(row)
    return rows


def summarize_trace(rows, capacity=500.):
    def total(key, group=rows):
        return float(sum(np.sum(r[key]) for r in group))
    demand = total('demand')
    core = [r for r in rows if 8 <= r['day'] <= 21]
    first = [r for r in rows if r['day'] <= 7]
    last = [r for r in rows if r['day'] >= 22]
    # Optimistic bound: demand-known daily allocation, ignores SKU availability,
    # procurement, order grid and supplier timing. Warm-up cannot receive before Lmin=2.
    optimistic_sales = sum(min(np.sum(r['demand']),capacity) for r in rows)
    first_two = rows[:2]
    initial = np.array(rows[0]['inventory_start'])
    optimistic_sales -= sum(min(np.sum(r['demand']),capacity) for r in first_two)
    optimistic_sales += np.minimum(initial,np.sum([r['demand'] for r in first_two],axis=0)).sum()
    return dict(label=rows[0]['label'],seed=rows[0]['seed'],days=len(rows),
        reward_day=float(np.mean([r['reward'] for r in rows])),fill=total('sales')/max(demand,1),
        core_fill=total('sales',core)/max(total('demand',core),1),
        first7_fill=total('sales',first)/max(total('demand',first),1),
        last7_fill=total('sales',last)/max(total('demand',last),1),
        demand=demand,orders=total('orders'),accepted=total('accepted'),clips=total('clipped'),
        lost=total('lost_sales'),initial_inventory=float(initial.sum()),
        mean_start_inventory=float(np.mean([sum(r['inventory_start']) for r in rows])),
        mean_end_inventory=float(np.mean([sum(r['inventory_end']) for r in rows])),
        mean_pipeline=float(np.mean([np.sum(r['pipeline']) for r in rows])),
        terminal_pipeline=float(sum(rows[-1]['terminal_pipeline'])),
        cap_full_days=sum(sum(r['inventory_start'])>=capacity-1e-6 for r in rows),
        max_grid_fraction=float(np.mean(np.array([r['orders'] for r in rows])>=200)),
        optimistic_capacity_warmup_fill=float(optimistic_sales/max(demand,1)),
        **{k:float(sum(r['reward_terms'][k] for r in rows)/len(rows)) for k in rows[0]['reward_terms']})


def main():
    from src.experiments.research import make_research_env,training_statistics_view
    from src.experiments.phase4 import heuristic
    from src.baselines.operating import ForecastBaseStock
    from src.demand.alignment import build_aligned_dataset
    from scripts.evaluate_research import discover_checkpoints,policies
    import torch
    torch.set_num_threads(1)
    ap=argparse.ArgumentParser()
    ap.add_argument('--episodes',type=int,default=3)
    ap.add_argument('--output',default='outputs/operating_diagnosis/main')
    ap.add_argument('--checkpoint-seeds',type=int,nargs='+',default=[11,22,33,44,55])
    args=ap.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():
        raise ValueError('choose fresh output; preserving existing results')
    calibration=Path('outputs/evaluation_research/calibration/train_only_nb.json')
    params=json.loads(calibration.read_text())
    assert params['fit_end_day']<=1885 and params['outcome_statistics_end_day']<=1885
    aligned=training_statistics_view(build_aligned_dataset())
    cps=[c for c in discover_checkpoints('outputs/training_research/clean_mappo_stability')
         if c['training_scenario']=='S2' and c['training_seed'] in args.checkpoint_seeds]
    assert len(cps)==len(args.checkpoint_seeds)
    summaries=[]
    started=time.perf_counter()
    labels=['ss_markup','constant_markup','base4_markup','base6_markup','base8_markup',
            'base10_markup','max_markup','base6_baseprice','base8_baseprice']
    labels += [f'mappo_{c["training_seed"]}' for c in cps]
    labels += ['learned_order_fixed_markup','ss_learned_price']
    for label in labels:
        env=make_research_env(params=params,aligned=aligned)
        price=lambda o:4
        if label.startswith('mappo_'):
            cp=next(c for c in cps if label==f'mappo_{c["training_seed"]}')
            order,price=policies('B6',env,cp)
        elif label in ('learned_order_fixed_markup','ss_learned_price'):
            learned_order,learned_price=policies('B6',env,cps[0])
            order=learned_order if label.startswith('learned_order') else lambda e,o:heuristic(e)
            price=(lambda o:4) if label.startswith('learned_order') else learned_price
        elif label.startswith('base'):
            cover=float(label.split('_')[0][4:])
            order=ForecastBaseStock(cover)
            if label.endswith('baseprice'):price=lambda o:2
        elif label=='max_markup':order=lambda e,o:np.full(e.n,len(e.grid)-1,dtype=np.int64)
        else:order=lambda e,o:heuristic(e,kind='constant' if label.startswith('constant') else 'ss')
        with (out/(label+'.jsonl')).open('w') as f:
            for seed in range(10000,10000+args.episodes):
                rows=replay(env,order,price,seed,label)
                for row in rows:f.write(json.dumps(row)+'\n')
                summaries.append(summarize_trace(rows))
        (out/'episodes.json').write_text(json.dumps(summaries,indent=2))
        print(label,flush=True)
    aggregates={label:{k:float(np.mean([r[k] for r in summaries if r['label']==label]))
                       for k in summaries[0] if k not in ('label','seed')}
                for label in labels}
    result=dict(episodes=len(summaries),trace_days=sum(r['days'] for r in summaries),
        seconds=time.perf_counter()-started,held_out_test_used=False,
        calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        checkpoints={c['path']:hashlib.sha256(Path(c['path']).read_bytes()).hexdigest() for c in cps},
        seeds=list(range(10000,10000+args.episodes)),aggregates=aggregates)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
