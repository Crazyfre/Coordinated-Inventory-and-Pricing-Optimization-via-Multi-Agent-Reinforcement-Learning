"""Paired research evaluation and statistical reporting."""
import numpy as np


def seed_library(episodes, root=20261012):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(root).spawn(episodes)]


def evaluate_episode(env, order_policy, price_policy, seed):
    from src.env.interface import JointAction
    obs, _ = env.reset(seed=seed)
    terms = {k:0. for k in ('revenue','ordering_cost','holding_cost','stockout_cost','price_change_cost')}
    demand = sales = lost = profit = stockouts = violations = clips = 0.
    inventory, inventory_value, prices, weekly_prices = [], [], [], []
    sold_value = 0.
    done = False
    while not done:
        if env.pricing_due:
            index = price_policy(obs['pricing'])
            env.select_price(index)
            weekly_prices.append(env.base_price*np.asarray(env.config.pricing.multipliers)[index])
        action = order_policy(env, env.replenishment_observation())
        obs, reward, done, info = env.step(JointAction(None, action))
        profit += reward
        for key in terms:
            terms[key] += info['reward_terms'][key]
        demand += info['demand'].sum(); sales += info['sales'].sum()
        lost += info['lost_sales'].sum(); stockouts += (info['lost_sales'] > 0).sum()
        inv = info['total_on_hand']
        inventory.append(inv)
        # Holding cost = h * total end inventory; per-SKU end stock reconstructed
        # before tomorrow's receipts, from the current engine minus accepted next day.
        end = env.inventory.on_hand.copy()
        if not done:
            end -= env._accepted_today
        inventory_value.append(float(end @ env.unit_cost))
        sold_value += float(info['sales'] @ env.unit_cost)
        violations += int(inv > env.capacity+1e-6)
        clips += info['orders_clipped'].sum()
        prices.append(info['price'])
    p = np.array(prices); wp = np.array(weekly_prices)
    return dict(seed=int(seed), profit=float(profit), **terms,
        demand=float(demand), sales=float(sales), lost_sales=float(lost),
        fill_rate=float(sales/max(demand,1)), stockout_rate=float(stockouts/(env.n*len(p))),
        average_inventory=float(np.mean(inventory)), inventory_turnover=float(sold_value/max(np.mean(inventory_value),1e-12)),
        capacity_violations=int(violations), receiving_clipped_units=float(clips),
        average_selling_price=float(terms['revenue']/max(sales,1)),
        price_volatility=float(np.abs(np.diff(p,axis=0)).mean()) if len(p)>1 else 0.,
        discount_frequency=float((wp < env.base_price).mean()),
        price_adjustment_frequency=float(np.any(np.diff(wp,axis=0)!=0,axis=1).mean()) if len(wp)>1 else 0.,
        service_target_met=int(sales/max(demand,1)>=env.config.reward.service_target_fill_rate))


def paired_difference(left, right):
    left, right = np.asarray(left,float), np.asarray(right,float)
    if left.shape != right.shape or left.size < 2:
        raise ValueError('paired CI requires equal arrays with at least two pairs')
    return describe(left-right)


def describe(values):
    from scipy.stats import t
    x = np.asarray(values,float)
    sd = float(x.std(ddof=1)) if len(x)>1 else 0.
    half = float(t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x))) if len(x)>1 else float('nan')
    mean = float(x.mean())
    return dict(n=len(x), mean=mean, std=sd, median=float(np.median(x)),
                p05=float(np.quantile(x,.05)), ci_low=mean-half, ci_high=mean+half)


def validate_final(calibration, manifests, episodes):
    if calibration.get('fit_end_day', 1941)>1885 or calibration.get('outcome_statistics_end_day',1941)>1885:
        raise ValueError('calibration includes validation/test outcomes or has no provenance')
    if episodes < 50:
        raise ValueError('final evaluation requires at least 50 episodes')
    seeds = set()
    for m in manifests:
        if (m.get('training_end_day',1941)>1885 or m.get('split')!='train'
            or m.get('updates',0)<10 or not m.get('frozen')):
            raise ValueError('checkpoint provenance missing, unfrozen, or smoke-only')
        seeds.add(m['training_seed'])
    if len(seeds)<10:
        raise ValueError('final target requires 10 distinct training seeds per method/ablation')


def crossed_summary(rows, metric, seed=0, replicates=2000):
    """Resample independent training seeds and shared episode seeds, not flat rows."""
    ts = sorted({str(r['training_seed']) for r in rows})
    es = sorted({r['seed'] for r in rows})
    values = {(str(r['training_seed']),r['seed']):r[metric] for r in rows}
    matrix = np.array([[values[t,e] for e in es] for t in ts],float)
    rng = np.random.default_rng(seed)
    # Vectorized crossed resampling keeps the two independent axes and avoids
    # millions of Python RNG/mean calls in the complete reporting matrix.
    uniforms = rng.random((replicates,len(ts)+len(es)))
    ti = (uniforms[:,:len(ts)]*len(ts)).astype(int)
    ei = (uniforms[:,len(ts):]*len(es)).astype(int)
    samples = matrix[ti[:,:,None],ei[:,None,:]].mean(axis=(1,2))
    result = describe(matrix.ravel())
    result.update(training_seeds=len(ts),evaluation_seeds=len(es),
                  training_seed_std=float(matrix.mean(axis=1).std(ddof=1)) if len(ts)>1 else 0.,
                  ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975)))
    return result


def summarize(rows):
    return {k:describe([r[k] for r in rows]) for k in rows[0] if k != 'seed'}
