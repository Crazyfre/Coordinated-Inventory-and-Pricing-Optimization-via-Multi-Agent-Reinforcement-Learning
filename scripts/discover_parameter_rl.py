"""Predeclared two-arm episodic policy-gradient development comparison."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.parameter_policy_gradient import ParameterPolicyGradient, initial_mean, sigmas, order_action
from src.env.interface import JointAction
from src.experiments.phase4 import heuristic
from scripts.train_compact_residual import setup, digest, serial, write_json, INPUTS


def episode(env, params, key, label, trace, counters, deadline, reference=None):
    env.reset(seed=key); rows = []; total = 0.; discounted = 0.; sold = 0.; demand = 0.
    for day in range(1, 29):
        if time.perf_counter() > deadline:
            raise TimeoutError('declared development budget')
        if env.pricing_due:
            env.select_price(4)
        obs = env.replenishment_observation()
        if isinstance(params, str):
            if params == 'projected':
                action = reference.actor.support(torch.tensor(obs)[None])[0][0].numpy()
            else:
                action = heuristic(env, params)
        else:
            action = order_action(obs, day, params)
        _, reward, done, info = env.step(JointAction(None, action))
        counters['days'] += 1
        assert done == (day == 28) and info['total_on_hand'] <= 500+1e-8
        np.testing.assert_allclose(info['price'], env.base_price*1.2, rtol=0, atol=1e-12)
        total += reward; discounted += .99**(day-1)*reward
        sold += info['sales'].sum(); demand += info['demand'].sum()
        row = serial(dict(label=label, scenario=int(key), start=int(env.sim.start_day_index), day=day,
                          obs=obs, actions=action, reward=reward, info=info))
        trace.write(json.dumps(row, allow_nan=False)+'\n'); rows.append(row)
    trace.flush(); counters['episodes'] += 1
    return dict(label=label, scenario=int(key), reward_day=total/28, discounted_day=discounted/28,
                fill=float(sold/demand) if demand else 1.), rows


def audit_compact(out):
    """Predetermined archived day1/8/15/22 observations; no new episodes or fitting."""
    root = Path('outputs/compact_residual/study_v1')
    groups = {s: [] for s in [11, 22, 33]}
    with (root/'assessment_traces.jsonl').open() as handle:
        for line in handle:
            r = json.loads(line)
            if r['label'].startswith('greedy_') and r['day'] in [1, 8, 15, 22]:
                groups[int(r['label'].split('_')[1])].append(r)
    result = {}
    from scripts.verify_compact_study import from_checkpoint
    for seed, rows in groups.items():
        path = root/'training'/f'seed{seed}'/'final.pt'
        model, _ = from_checkpoint(path); model.load(path)
        x = torch.tensor([r['obs'] for r in rows]); days = [r['day'] for r in rows]
        with torch.no_grad():
            dist = model.actor(x, days)
            default = model.actor.support(x)[0]
            p = dist.probs.gather(-1, default[..., None]).squeeze(-1)
            logits = dist.logits.clone(); chosen = logits.gather(-1, default[..., None]).squeeze(-1)
            logits.scatter_(-1, default[..., None], -torch.inf)
            margin = chosen-logits.max(-1).values
        h = json.loads((root/'training'/f'seed{seed}'/'history.json').read_text())
        ev = [u['explained_variance'] for u in h if u['explained_variance'] is not None]
        result[str(seed)] = dict(states=len(rows), checkpoint_sha256=digest(path),
             default_probability_mean=float(p.mean()), default_probability_min=float(p.min()),
             default_logit_margin_min=float(margin.min()), default_logit_margin_mean=float(margin.mean()),
             actor_weight_norm=float(model.actor.weight.detach().norm()),
             critic_explained_variance_median=float(np.median(ev)), critic_explained_variance_final=ev[-1])
    write_json(out/'compact_archive_audit.json', dict(results=result, days=[1, 8, 15, 22],
               trace_sha256=digest(root/'assessment_traces.jsonl'), new_simulated_days=0,
               interpretation='Descriptive final-policy margins and critic fit, not causal attribution.'))
    return result


def train_arm(env, arm, seed, first_key, updates, pairs, out, trace, counters, deadline):
    learner = ParameterPolicyGradient(arm, seed)
    history = []
    for update in range(updates):
        epsilon = learner.rng.normal(size=(pairs, len(learner.mean)))
        before = learner.mean.copy(); differences = []; records = []
        for pair in range(pairs):
            key = first_key+update*pairs+pair
            params_plus = before+learner.sigma*epsilon[pair]
            params_minus = before-learner.sigma*epsilon[pair]
            plus, tp = episode(env, params_plus, key, f'{arm}_{seed}_u{update}_p{pair}_plus', trace, counters, deadline)
            minus, tm = episode(env, params_minus, key, f'{arm}_{seed}_u{update}_p{pair}_minus', trace, counters, deadline)
            assert [r['info']['demand'] for r in tp] == [r['info']['demand'] for r in tm]
            differences.append(plus['discounted_day']-minus['discounted_day'])
            records.append(dict(key=key, plus=plus, minus=minus))
        gradient = learner.update(epsilon, differences)
        history.append(dict(update=update+1, mean_before=before.tolist(), epsilon=epsilon.tolist(),
                            differences=differences, gradient=gradient.tolist(), records=records, state=learner.state()))
        write_json(out/f'{arm}_seed{seed}_history.json', history)
        if (update+1) % 8 == 0:
            print(f'{arm} seed{seed}: {update+1}/{updates} updates', flush=True)
    write_json(out/f'{arm}_seed{seed}_final.json', learner.state())
    return learner


def run(output):
    out = Path(output); out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1)
    sources = ['RL_DISCOVERY_PROTOCOL.md', 'src/algos/parameter_policy_gradient.py', __file__,
               'src/env/joint_env.py', 'src/env/replenishment_env.py', 'src/experiments/research.py',
               'scripts/train_compact_residual.py', 'scripts/train_clean_research.py']+list(INPUTS.values())
    hashes = {str(p): digest(p) for p in sources}
    write_json(out/'manifest.json', dict(hashes=hashes, arms=['sku', 'arrival'], seed=101,
               updates=24, pairs=8, development_keys=[1120000, 1120063], max_seconds=900,
               primary_execution='deterministic_mean', confirmation_used=False))
    audit_compact(out)
    started = time.perf_counter(); deadline = started+900; counters = dict(episodes=0, days=0)
    try:
        env, reference = setup(101)
        learners = {}
        with (out/'training_traces.jsonl').open('w') as trace:
            for arm, key in [('sku', 1100000), ('arrival', 1110000)]:
                learners[arm] = train_arm(env, arm, 101, key, 24, 8, out, trace, counters, deadline)
        rows = []
        with (out/'development_traces.jsonl').open('w') as trace:
            for key in range(1120000, 1120064):
                policies = {a: l.mean for a, l in learners.items()}
                policies.update(full7_5=initial_mean('sku'), projected='projected', constant='constant', ss='ss')
                for arm in ['sku', 'arrival']:
                    rng = np.random.default_rng(np.random.SeedSequence([101, key, 707]))
                    policies['initial_'+arm] = initial_mean(arm)+sigmas(arm)*rng.normal(size=len(initial_mean(arm)))
                demand = None
                for label, params in policies.items():
                    result, traces = episode(env, params, key, label, trace, counters, deadline, reference)
                    current = [r['info']['demand'] for r in traces]
                    if demand is None:
                        demand = current
                    assert current == demand
                    rows.append(result)
        means = {label: {m: float(np.mean([r[m] for r in rows if r['label'] == label]))
                         for m in ['reward_day', 'fill']} for label in policies}
        eligible = [a for a in learners if means[a]['reward_day']-means['full7_5']['reward_day'] >= .5
                    and means[a]['fill']-means['full7_5']['fill'] >= -.005]
        selected = max(eligible, key=lambda a: (means[a]['reward_day'], a == 'sku')) if eligible else None
        for p, h in hashes.items():
            assert digest(p) == h
        write_json(out/'summary.json', dict(status='complete', means=means, rows=rows, selected=selected,
                   eligible=eligible, counters=counters, seconds=time.perf_counter()-started,
                   confirmation_used=False, validation_test_episodes=0,
                   trace_hashes={n: digest(out/n) for n in ['training_traces.jsonl', 'development_traces.jsonl']}))
        print(json.dumps(dict(selected=selected, means=means, counters=counters)), flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json', dict(error=str(exc), counters=counters, seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True)
    run(p.parse_args().output)
