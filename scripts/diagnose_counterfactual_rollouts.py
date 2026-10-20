"""Training-only delayed-return branch validity audit; no policy fitting."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.diagnose_cover_search import CoverPolicy
from scripts.diagnose_residual_support import project_target
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.env.interface import JointAction
from src.experiments.research import make_research_env


def clone_environment(env):
    branch = copy.deepcopy(env)
    # step rebuilds this hook; deepcopy cannot rebind a Python closure.
    branch.pricing_hook = None
    return branch


def decision(env, center, target):
    if env.pricing_due:
        env.select_price(4)
    obs = env.replenishment_observation()
    c = center.predict(obs)
    return c, project_target(c, target.act(obs)[0])


def advance(env, action):
    day = env.day
    _, reward, done, info = env.step(JointAction(None, action))
    row = dict(day=int(day), actions=action.tolist(), reward=float(reward),
               demand=info['demand'].tolist(), sales=info['sales'].tolist(),
               leads=env._last_lead_times.tolist(), stock=float(info['total_on_hand']))
    assert row['stock'] <= 500 + 1e-8
    return row, done


def suffix(env, center, target, first, deadline):
    rows = []
    while True:
        if time.perf_counter() > deadline:
            raise TimeoutError('fixed 300-second audit budget exceeded')
        _, action = decision(env, center, target)
        row, done = advance(env, first if not rows else action)
        rows.append(row)
        if done:
            return rows


def check_branches(branches, reference):
    for b in branches:
        rows = b['trace']
        assert len(rows) == len(reference)
        assert rows[0]['reward'] == reference[0]['reward']
        for row, ref in zip(rows, reference):
            assert row['demand'] == ref['demand']
            for a, ar, lead, lr in zip(row['actions'], ref['actions'], row['leads'], ref['leads']):
                if a > 0 and ar > 0:
                    assert lead == lr


def run(output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + 300
    paths = dict(center='outputs/structured_imitation/seed11_v1/structured_policy.json',
                 selection='outputs/cover_search/bounded_v1/selection.json',
                 calibration='outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json',
                 protocol='COUNTERFACTUAL_ROLLOUT_PROTOCOL.md', source=__file__)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    hashes = {k: digest(p) for k, p in paths.items()}
    (out/'manifest.json').write_text(json.dumps(dict(paths=paths, hashes=hashes), indent=2))
    torch.set_num_threads(1)
    params = load_identifiable_training_calibration(paths['calibration'])
    env = make_research_env('S2', split='train', start=15, params=params, pricing_scope='shared')
    center = InventoryPositionLinearPolicy.load(paths['center'], env.n, env.obs_pipe_horizon)
    target = CoverPolicy(env.n, env.obs_pipe_horizon, json.loads(Path(paths['selection']).read_text())['selected']['cover'])
    records = []
    references = []
    for ep, seed in enumerate(range(920000, 920008)):
        start = int(np.random.default_rng(np.random.SeedSequence([seed, 314159])).integers(15, 1859))
        env.sim.start_day_index = start
        env.reset(seed=seed)
        original = []
        episode_records = []
        while not env._finished:
            c, action = decision(env, center, target)
            day = env.day
            if day in (1, 8, 15, 22):
                slot = (day-1)//7
                sku = (ep*4+slot) % env.n
                # Original suffix provides a direct state/RNG immutability check after all branches.
                before = suffix(clone_environment(env), center, target, action, deadline)
                branches = []
                for alternative in range(max(0, int(c[sku])-2), min(40, int(c[sku])+2)+1):
                    first = action.copy()
                    first[sku] = alternative
                    trace = suffix(clone_environment(env), center, target, first, deadline)
                    rewards = np.array([r['reward'] for r in trace])
                    branches.append(dict(action=alternative, trace=trace, total=float(rewards.sum()),
                                         discounted=float(np.dot(.99**np.arange(len(rewards)), rewards))))
                check_branches(branches, before)
                reference = next(b for b in branches if b['action'] == action[sku])
                assert reference['trace'] == before
                episode_records.append(dict(seed=seed, start=start, day=day, sku=sku,
                                            center=int(c[sku]), reference_action=int(action[sku]), branches=branches))
            row, _ = advance(env, action)
            original.append(row)
        for record in episode_records:
            ref = next(b for b in record['branches'] if b['action'] == record['reference_action'])
            assert ref['trace'] == original[record['day']-1:]
        env.reset(seed=seed)
        _, action = decision(env, center, target)
        assert suffix(env, center, target, action, deadline) == original
        references.append(dict(seed=seed, start=start, trace=original))
        records.extend(episode_records)
        print(f'Completed training window {ep+1}/8', flush=True)
    assert {k:digest(p) for k,p in paths.items()} == hashes
    changed = 0
    gains = []
    for rec in records:
        ref = next(b for b in rec['branches'] if b['action'] == rec['reference_action'])
        changed += any(b['total'] != ref['total'] for b in rec['branches'])
        gains.append(max(b['discounted'] for b in rec['branches'])-ref['discounted'])
    summary = dict(states=len(records), branches=sum(len(r['branches']) for r in records),
                   branch_days=sum(len(b['trace']) for r in records for b in r['branches']),
                   baseline_probe_days=560, original_days=224, replay_days=224,
                   states_with_changed_total_return=changed,
                   mean_hindsight_discounted_gain=float(np.mean(gains)),
                   validity_gate=True, nonzero_delayed_signal_gate=changed>0,
                   seconds=time.perf_counter()-started, rl_updates=0, held_out_test_used=False,
                   interpretation='Training-only hindsight returns; not policy improvement or RL success.')
    (out/'traces.json').write_text(json.dumps(dict(records=records, references=references)))
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
