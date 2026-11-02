"""Allowlisted local development checkpoints; no client-provided paths."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {'r3_ppo': ROOT/'outputs/r3/checkpoint.pt',
               'p4_mappo': ROOT/'outputs/p4/checkpoint.pt'}


def catalog():
    rows = {}
    for name, path in CHECKPOINTS.items():
        try:
            regime = json.loads(path.with_name('run_config.json').read_text())
            regime = {k: regime[k] for k in ('days', 'capacity', 'updates')}
            available = path.is_file()
            error = '' if available else 'Checkpoint unavailable; train the corresponding development policy.'
        except (OSError, ValueError, KeyError):
            regime, available, error = {}, False, 'Checkpoint run configuration unavailable or invalid.'
        rows[name] = dict(available=available, regime=regime, error=error,
                          label='Smoke-trained / not research validated')
    from demo.structured_inference import status
    rows['structured_mappo']=status()
    return rows


def compare(policy, seed=0):
    if policy=='structured_mappo':
        from demo.structured_inference import compare as structured_compare
        return structured_compare(seed)
    return {'runs': [run(policy, seed)] + [run(policy, seed, baseline=b) for b in ('zero','constant','ss')],
            'note': 'Common seeded demand stream and day/SKU supplier scenarios. Different prices can produce different realized demand; this is not a research validation.'}


def run(policy, seed=0, baseline=None):
    if policy=='structured_mappo':
        from demo.structured_inference import run as structured_run
        if baseline is not None:raise ValueError('Use the structured pair comparison')
        return structured_run(seed)
    import numpy as np
    import torch
    from src.experiments.phase4 import make_env, heuristic
    from src.agents.replenishment_agent import ReplenishmentPPOAgent
    from src.algos.mappo import HeterogeneousMAPPO
    from src.env.interface import JointAction
    if policy not in CHECKPOINTS:
        raise ValueError('Unknown learned policy')
    status = catalog()[policy]
    if not status['available']:
        raise ValueError(status['error'])
    cfg = status['regime']
    torch.set_num_threads(1)
    joint = policy == 'p4_mappo'
    env = make_env(days=cfg['days'], capacity=cfg['capacity'], joint=joint, pricing_scope='shared')
    agent = HeterogeneousMAPPO(env) if joint else ReplenishmentPPOAgent(env)
    try:
        agent.load(CHECKPOINTS[policy])
    except Exception as exc:
        raise ValueError('Checkpoint incompatible or unreadable; retrain the development policy.') from exc
    obs, _ = env.reset(seed=int(seed))
    rows, done = [], False
    while not done:
        if joint and env.pricing_due:
            price_index = 2 if baseline else agent.pricing_action(obs['pricing'], greedy=True)
            env.select_price(price_index)
        robs = env.replenishment_observation() if joint else obs
        if baseline:
            action = heuristic(env, baseline)
        elif joint:
            action = agent.replenishment.act(robs, greedy=True)[0]
        else:
            action = agent.act(robs, greedy=True)[0]
        if joint:
            obs, reward, done, diag = env.step(JointAction(None, action))
        else:
            obs, reward, term, trunc, diag = env.step(action)
            done = term or trunc
        stock = env.inventory.on_hand.copy()
        if not done:
            stock -= env._accepted_today
        rows.append(dict(day=int(diag['day']), demand=diag['demand'].tolist(),
                         sales=diag['sales'].tolist(), lost=diag['lost_sales'].tolist(),
                         orders=action.tolist(), order_units=env.grid[action].tolist(),
                         arrivals=diag['arrivals'].tolist(), clipped=diag['orders_clipped'].tolist(),
                         price=diag['price'].tolist(), on_hand=stock.tolist(),
                         reward=float(reward), reward_terms=diag['reward_terms']))
    demand = sum(sum(d['demand']) for d in rows)
    sales = sum(sum(d['sales']) for d in rows)
    summary = dict(profit_total=sum(d['reward'] for d in rows),
                   profit_per_day=float(np.mean([d['reward'] for d in rows])),
                   fill_rate=sales/max(demand, 1), total_demand=demand, total_sales=sales,
                   total_lost=demand-sales, avg_inventory=float(np.mean([sum(d['on_hand']) for d in rows])),
                   clipped_units=sum(sum(d['clipped']) for d in rows),
                   capacity_violation_events=int(diag['capacity_violation_events']))
    for key, term in [('revenue_total','revenue'), ('ordering_total','ordering_cost'),
                      ('holding_total','holding_cost'), ('stockout_total','stockout_cost'),
                      ('price_change_total','price_change_cost')]:
        summary[key] = sum(d['reward_terms'][term] for d in rows)
    return dict(meta=dict(policy=policy if not baseline else baseline, days=cfg['days'],
                          capacity=cfg['capacity'], seed=int(seed), lead=[2,7],
                          weekly_multiplier=1 if baseline or not joint else 'learned',
                          validation=status['label'], pricing_scope='shared', regime='training-compatible: calendar 1800, shock .15, calibrated forecast, 2-day initial cover'),
                skus=env.sim.series_ids, base_prices=env.base_price.tolist(),
                days=rows, summary=summary)

