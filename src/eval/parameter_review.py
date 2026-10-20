"""Review metrics for frozen structured policies; no training or selection."""
import numpy as np
from src.eval.review_metrics import return_summary, success_interval, convergence

SEEDS = [211, 222, 233, 244, 255]


def episode_metrics(trace):
    if len(trace) != 28 or [r['day'] for r in trace] != list(range(1, 29)):
        raise ValueError('one complete 28-day episode required')
    reward = np.array([r['reward'] for r in trace], float)
    demand = sum(np.asarray(r['info']['demand'], float).sum() for r in trace)
    sales = sum(np.asarray(r['info']['sales'], float).sum() for r in trace)
    if not np.isfinite(reward).all() or not 0 <= sales <= demand:
        raise ValueError('invalid reward or demand/sales accounting')
    fill = float(sales/demand) if demand else 1.
    result = dict(label=trace[0]['label'], scenario=trace[0]['scenario'],
        start=trace[0]['start'], episodic_return=float(reward.sum()), reward_day=float(reward.mean()),
        fill=fill, success=int(fill >= .95), demand=float(demand), sales=float(sales),
        lost_sales=float(demand-sales), mean_inventory=float(np.mean([r['info']['total_on_hand'] for r in trace])),
        clipped_units=float(sum(np.asarray(r['info']['orders_clipped']).sum() for r in trace)),
        delivered_units=float(sum(np.asarray(r['info']['orders_delivered']).sum() for r in trace)),
        stockout_sku_day_fraction=float(np.mean([np.asarray(r['info']['lost_sales']) > 0 for r in trace])))
    for term in ['revenue', 'ordering_cost', 'holding_cost', 'stockout_cost', 'price_change_cost']:
        result[term] = float(sum(r['info']['reward_terms'][term] for r in trace))
    reconstructed = result['revenue']-sum(result[t] for t in ['ordering_cost','holding_cost','stockout_cost','price_change_cost'])
    if abs(reconstructed-result['episodic_return']) > 1e-7:
        raise ValueError('economic reward does not reconstruct')
    return result


def crossed(values, seed=20261020):
    x = np.asarray(values, float)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError('finite training-seed by scenario matrix required')
    rng = np.random.default_rng(seed)
    r = rng.integers(x.shape[0], size=(10000, x.shape[0]))
    c = rng.integers(x.shape[1], size=(10000, x.shape[1]))
    draws = x[r[:, :, None], c[:, None, :]].mean((1, 2))
    return dict(mean=float(x.mean()), ci95=np.quantile(draws, [.025,.975]).tolist(),
                seed_means=x.mean(1).tolist(), exploratory=True)


def summarize(rows, learned_prefix='learned_'):
    index = {(r['label'], r['scenario']): r for r in rows}
    if len(index) != len(rows):
        raise ValueError('duplicate controller/scenario rows')
    keys = sorted({r['scenario'] for r in rows}); labels = sorted({r['label'] for r in rows})
    if len(index) != len(keys)*len(labels):
        raise ValueError('complete paired controller/scenario grid required')
    metrics = ['reward_day','fill','mean_inventory','clipped_units','delivered_units',
               'lost_sales','stockout_sku_day_fraction','revenue','ordering_cost','holding_cost',
               'stockout_cost','price_change_cost']
    controllers = {}
    for label in labels:
        group = [index[label, k] for k in keys]
        controllers[label] = dict(episodic_return=return_summary([r['episodic_return'] for r in group]),
            success=success_interval([r['success'] for r in group]),
            means={m: float(np.mean([r[m] for r in group])) for m in metrics})
    effects = {m: crossed([[index[f'{learned_prefix}{s}', k][m]-index['full7_5', k][m]
                            for k in keys] for s in SEEDS]) for m in ['reward_day','fill']}
    # Seed-averaged curve over scenario keys: descriptive, not independent online regret.
    shortfall = np.array([index['full7_5', k]['episodic_return']-
        np.mean([index[f'{learned_prefix}{s}', k]['episodic_return'] for s in SEEDS]) for k in keys])
    return dict(controllers=controllers, effects_vs_full7_5=effects,
        reference_shortfall=dict(keys=keys, signed_episode_differences=shortfall.tolist(),
            cumulative=np.cumsum(shortfall).tolist(), label='Feasible-reference shortfall; not optimal/online regret'))


def progress_threshold(rows):
    index = {(r['label'], r['scenario']): r for r in rows}
    keys = sorted({r['scenario'] for r in rows})
    output = {}
    for seed in SEEDS:
        evaluations = []
        for update in [0,16,32,48,64]:
            gains = [0. if update == 0 else index[f'u{update}_{seed}', k]['reward_day']-
                     index['full7_5', k]['reward_day'] for k in keys]
            evaluations.append(dict(update=update, steps=update*16*28,
                mean_evaluation_return=float(np.mean(gains))))
        output[str(seed)] = dict(evaluations=evaluations,
            threshold=convergence(evaluations, threshold=1., consecutive=3))
    return output
