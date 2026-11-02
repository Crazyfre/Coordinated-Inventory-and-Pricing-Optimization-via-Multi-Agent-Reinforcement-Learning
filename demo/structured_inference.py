"""Audited structured MAPPO demonstration, fixed first seed1801 and S2 regime.

Seed1801 is chosen by its position in the prospective seed list, not its reward.
Checkpoint identity comes from the pre-assessment freeze. Inference is local;
global critic values are never consulted. No user-supplied paths.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'outputs/direct_mappo/comparison_v1/1801_mappo_final.pt'
AUDIT=ROOT/'outputs/direct_mappo/control_completion_v1_audit_v2/report.json'
FREEZE=ROOT/'outputs/direct_mappo/completed_eval_v1/freeze.json'

def status():
    row=dict(available=False,regime=dict(days=28,capacity=500,updates=128),
        label='Structured MAPPO: replay verified; improvement unproven',
        description='Original S2 regime; weekly per-SKU prices and daily structured orders. '
            'Fixed training seed 1801. The scenario seed is adjustable. Compare with the strong frozen pair.',
        error='Audited structured checkpoint unavailable.')
    try:
        expected=json.loads(FREEZE.read_text())['checkpoints'][CHECKPOINT.name]
        if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()!=expected:raise ValueError('checkpoint identity mismatch')
        audit=json.loads(AUDIT.read_text())
        if audit['status']!='passed' or audit['episodes']!=640:raise ValueError('verification incomplete')
        declared=json.loads((ROOT/'outputs/direct_mappo/control_completion_v1/manifest.json').read_text())['sources']
        for name in ('outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json',
                     'outputs/rl_discovery/parameter_confirmation_v1/sku_seed211_final.json',
                     'src/algos/structured_mappo.py','src/algos/structured_ordering_ppo.py',
                     'src/algos/structured_pricing_ppo.py'):
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=declared[name]:raise ValueError('evaluation input/source changed')
        row.update(available=True,error='',evidence=dict(means=audit['means'],gates=audit['gates'],
            evaluation_scenarios=64,training_seeds=3,limitation=audit['limitation']))
    except (OSError,KeyError,ValueError):pass
    return row

def run(seed=0,baseline=False):
    import numpy as np
    import torch
    from src.algos.structured_mappo import StructuredMAPPO
    from src.algos.structured_pricing_ppo import pricing_features
    from src.algos.structured_ordering_ppo import ordering_features,execute_cover
    from src.algos.parameter_policy_gradient import order_action
    from src.experiments.research import make_research_env
    from src.demand.identifiable_calibration import load_identifiable_training_calibration
    from src.env.interface import JointAction
    evidence=status()
    if not evidence['available']:raise ValueError(evidence['error'])
    if int(seed)!=seed or not 0<=int(seed)<=1000000:raise ValueError('Seed must be between0and1000000')
    seed=int(seed);torch.set_num_threads(1)
    calibration=ROOT/'outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json'
    covers=np.array(json.loads((ROOT/'outputs/rl_discovery/parameter_confirmation_v1/sku_seed211_final.json').read_text())['mean'])
    env=make_research_env('S2',split='train',start=15,days=28,pricing_scope='per_sku',
        params=load_identifiable_training_calibration(str(calibration)))
    model=StructuredMAPPO(env,covers,1801)
    model.restore(torch.load(CHECKPOINT,map_location='cpu',weights_only=False))
    start=int(np.random.default_rng(np.random.SeedSequence([seed,314159])).integers(15,1859))
    env.sim.start_day_index=start;env.reset(seed=seed);rows=[]
    for day in range(1,29):
        if env.pricing_due:
            price=(np.full(10,4,dtype=int) if baseline else
                   model.pricing.act(pricing_features(env.pricing_observation(),day,env.base_price),True)[0])
            env.select_price(price)
        obs=env.replenishment_observation()
        action=(order_action(obs,day,covers) if baseline else
                execute_cover(obs,model.ordering.act(ordering_features(obs,day),True)[0]))
        _,reward,done,info=env.step(JointAction(None,action))
        assert done==(day==28)
        stock=env.inventory.on_hand.copy()
        if not done:stock-=env._accepted_today
        rows.append(dict(day=day,demand=info['demand'].tolist(),sales=info['sales'].tolist(),
            lost=info['lost_sales'].tolist(),orders=action.tolist(),order_units=env.grid[action].tolist(),
            arrivals=info['arrivals'].tolist(),clipped=info['orders_clipped'].tolist(),price=info['price'].tolist(),
            on_hand=stock.tolist(),reward=float(reward),reward_terms=info['reward_terms']))
    demand=sum(sum(r['demand']) for r in rows);sales=sum(sum(r['sales']) for r in rows)
    summary=dict(profit_total=sum(r['reward'] for r in rows),profit_per_day=float(np.mean([r['reward'] for r in rows])),
        fill_rate=sales/max(demand,1),total_demand=demand,total_sales=sales,total_lost=demand-sales,
        avg_inventory=float(np.mean([sum(r['on_hand']) for r in rows])),clipped_units=sum(sum(r['clipped']) for r in rows),
        capacity_violation_events=int(info['capacity_violation_events']))
    for name,term in [('revenue_total','revenue'),('ordering_total','ordering_cost'),('holding_total','holding_cost'),
                      ('stockout_total','stockout_cost'),('price_change_total','price_change_cost')]:
        summary[name]=sum(r['reward_terms'][term] for r in rows)
    return dict(meta=dict(policy='strong_anchor' if baseline else 'structured_mappo',days=28,capacity=500,
        seed=seed,lead=[2,7],weekly_multiplier='frozen1.20' if baseline else 'learned',
        validation=evidence['label'],pricing_scope='per_sku',regime=f'S2; training-calendar start {start}; seed1801'),
        skus=env.sim.series_ids,base_prices=env.base_price.tolist(),days=rows,summary=summary,evidence=evidence.get('evidence'))

def compare(seed=0):
    return dict(runs=[run(seed),run(seed,baseline=True)],
        note='Same seeded S2 scenario. The table is one demonstration; the 3-seed study found no MAPPO improvement over the strong frozen pair.')
