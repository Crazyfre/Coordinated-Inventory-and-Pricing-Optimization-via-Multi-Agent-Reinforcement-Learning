"""Single bounded fixed1.20x ordering experiment; fresh outputs only."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from src.demand.identifiable_calibration import validate_research_training_calibration
from src.algos.forecast_prior import ForecastPriorMAPPO
from src.algos.ordering_learning import FixedMarkupActor
from src.experiments.research import make_research_env, training_statistics_view
from src.demand.alignment import build_aligned_dataset
from src.experiments.phase4 import heuristic
from scripts.train_clean_research import TrainingWindows
from scripts.diagnose_operations import replay, summarize_trace


def run(output, updates=64, order_lr=3e-5, seed=11, residual=False, critic_free=False,
        calibration="outputs/evaluation_research/calibration/train_only_nb.json"):
    if not 1 <= updates <= 128 or not 0 < order_lr <= .0003:
        raise ValueError('bounded updates/lr required')
    torch.set_num_threads(1)
    start = time.perf_counter()
    deadline = start + 570
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    calibration = Path(calibration)
    params = json.loads(calibration.read_text())
    validate_research_training_calibration(params)
    aligned = training_statistics_view(build_aligned_dataset())
    train = TrainingWindows(make_research_env('S2', split='train', start=15, days=28, params=params, aligned=aligned, pricing_scope='shared'), 28)
    validation = make_research_env('S2', split='validation', start=1886, days=28, params=params, aligned=aligned, pricing_scope='shared')
    torch.manual_seed(seed)
    cls = ForecastPriorMAPPO
    if critic_free:
        from src.algos.ordering_critic_probe import CriticFreeForecastPriorMAPPO
        cls = CriticFreeForecastPriorMAPPO
    model = cls(train, normalize_inputs=True, target_kl=.02)
    model.pricing.__class__ = FixedMarkupActor
    if residual:
        from src.algos.ordering_residual import ForecastRelativeResidualActor
        model.replenishment.__class__ = ForecastRelativeResidualActor
    model.optimizers[1].param_groups[0]['lr'] = order_lr
    config = dict(variant='forecast_prior_fixed_markup_diagnostic', price_index=4, price_multiplier=1.2,
        scenario='S2', seed=seed, updates=updates, order_lr=order_lr, critic_lr=.0003,
        epochs=4, reward_scale=100, target_kl=.02, grid=list(range(0,201,5)),
        validation_seeds=list(range(10000,10005)), torch_threads=1, held_out_test_used=False,
        calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(), max_seconds=570)
    config['pricing_scope'] = 'shared'
    config['calibration_status'] = params['status']
    config['calibration_path'] = str(calibration)
    config['residual'] = residual
    if residual:
        config['variant'] = 'forecast_relative_residual_fixed_markup_diagnostic'
        config['residual_offsets'] = [-2,-1,0,1,2]
        config['support'] = 'local clipped offsets; duplicate physical probabilities merged'
    config['critic_free'] = critic_free
    if critic_free:
        config['variant'] += '_zero_value_baseline'
    (out/'config.json').write_text(json.dumps(config, indent=2))
    def measure(label, order):
        summaries = []
        with (out/(label+'.jsonl')).open('w') as f:
            for evseed in config['validation_seeds']:
                if time.perf_counter() > deadline: raise TimeoutError('budget exceeded')
                rows = replay(validation, order, lambda o:4, evseed, label)
                assert all(np.allclose(r['price'], validation.base_price*1.2) for r in rows)
                summaries.append(summarize_trace(rows))
                for r in rows: f.write(json.dumps(r)+'\n')
        return summaries
    initial = measure('initial', lambda e,o:model.replenishment.act(o,True)[0])
    reference = measure('constant_forecast', lambda e,o:heuristic(e,kind='constant'))
    history = []
    # Sequential one-update calls preserve original episode seed and training RNG.
    for update in range(updates):
        if time.perf_counter() > deadline: raise TimeoutError('budget exceeded')
        history.extend(model.train(1, seed+update))
    assert all(np.isfinite(v) for r in history for v in r.values())
    trained = measure('trained', lambda e,o:model.replenishment.act(o,True)[0])
    model.save(out/'checkpoint.pt')
    data = torch.load(out/'checkpoint.pt', weights_only=True)
    data['variant'] = config['variant']
    data['fixed_price_index'] = 4
    torch.save(data, out/'checkpoint.pt')
    # Explicit matching reconstruction, never vanilla/prior loader.
    restored = ForecastPriorMAPPO(validation, normalize_inputs=True, target_kl=.02)
    restored.pricing.__class__ = FixedMarkupActor
    if residual:
        restored.replenishment.__class__ = ForecastRelativeResidualActor
    restored.replenishment.load_state_dict(data['replenishment'])
    restored.critic.load_state_dict(data['critic'])
    replayed = measure('restored', lambda e,o:restored.replenishment.act(o,True)[0])
    assert [r['reward_day'] for r in replayed] == [r['reward_day'] for r in trained]
    delta = {k:[b[k]-a[k] for a,b in zip(initial,trained)] for k in ('reward_day','fill')}
    summary = dict(config=config, initial=initial, trained=trained, reference=reference,
        deltas=delta, means={stage:{k:float(np.mean([r[k] for r in rows])) for k in ('reward_day','fill','core_fill','first7_fill','last7_fill')}
        for stage,rows in [('initial',initial),('trained',trained),('reference',reference)]},
        useful_learning_gate=bool(np.mean(delta['reward_day'])>0 and np.mean(delta['fill'])>=0 and all(x>0 for x in delta['reward_day'])),
        elapsed_seconds=time.perf_counter()-start, completed_updates=len(history), training_windows=train.windows,
        checkpoint_ordering_replay_verified=True, history=history)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:summary[k] for k in ('means','deltas','useful_learning_gate','elapsed_seconds','completed_updates')},indent=2),flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--updates',type=int,default=64)
    p.add_argument('--order-lr',type=float,default=3e-5)
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--residual',action='store_true')
    p.add_argument('--critic-free',action='store_true')
    p.add_argument('--calibration',default='outputs/evaluation_research/calibration/train_only_nb.json')
    a = p.parse_args()
    run(a.output,a.updates,a.order_lr,a.seed,a.residual,a.critic_free,a.calibration)
