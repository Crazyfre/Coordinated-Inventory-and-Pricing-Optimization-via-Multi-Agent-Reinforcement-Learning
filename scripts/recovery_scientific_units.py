"""Future-only deterministic reconstruction helpers; no campaign entry point.

Each paired-future unit starts from reset and pays for the baseline prefix.
Audit snapshots are compared, never used to restore hidden simulator state.
"""
from scripts.screen_cross_sku_pricing import reset
from scripts.diagnose_conditional_pricing import advance
from scripts.audit_repeated_futures import snapshot, resample_future


def counted_step(env, covers, first_day, sku, action, budget):
    budget.before_step()
    counters = {'days': 0}
    row, done = advance(env, covers, first_day, sku, action, counters, budget.deadline)
    if counters['days'] != 1:
        raise RuntimeError('one reserved step must execute exactly one simulator day')
    budget.check()
    return row, done


def reconstruct(env, covers, key, day, budget):
    if day not in (1, 8, 15, 22):
        raise ValueError('weekly decision day required')
    budget.check()
    reset(env, key)
    prefix = []
    for expected in range(1, day):
        row, done = counted_step(env, covers, 1, 0, 4, budget)
        if row['day'] != expected or done:
            raise RuntimeError('reconstruction calendar mismatch')
        prefix.append(row)
    if int(env.day) != day or not env.pricing_due:
        raise RuntimeError('reconstruction did not reach decision boundary')
    return prefix


def paired_future(env, covers, key, day, sku, keys, budget, expected_before=None):
    """One fixed 0-versus-4 future, costing (day-1)+2*(29-day) days.

The caller must bind key/day/SKU/future keys to immutable input identities.
No selection, fitting, retry decisions or scientific success claims occur here.
"""
    if not 0 <= sku < 10 or len(keys) != 2:
        raise ValueError('one SKU and two scenario keys required')
    prefix = reconstruct(env, covers, key, day, budget)
    before = snapshot(env)
    if expected_before is not None and before != expected_before:
        raise ValueError('reconstructed state differs from frozen state')
    branches = _branches(env, covers, day, sku, keys, budget)
    if snapshot(env) != before:
        raise RuntimeError('paired futures mutated original environment')
    return dict(key=key, day=day, sku=sku, keys=list(keys), prefix=prefix,
                before=before, branches=branches)


def _branches(env, covers, day, sku, keys, budget):
    branches = []
    for action in (0, 4):
        branch = resample_future(env, keys)
        initial = snapshot(branch)
        trace = []
        for expected in range(day, 29):
            row, done = counted_step(branch, covers, day, sku, action, budget)
            if row['day'] != expected or done != (expected == 28):
                raise RuntimeError('branch calendar mismatch')
            trace.append(row)
        branches.append(dict(action=action, initial=initial, trace=trace,
                             value=sum(.99**i*r['reward'] for i, r in enumerate(trace))))
    return branches


def paired_group(env, covers, key, day, sku, futures, budget, expected_before=None):
    """Four fixed consecutive replicas; one paid prefix, eight suffixes.

futures contains four {replica, keys} records from the frozen plan. The caller
must verify their seed derivation independently; this helper checks shape/order.
"""
    if day not in (1, 8, 15, 22) or type(sku) is not int or not 0 <= sku < 10:
        raise ValueError('weekly day and valid SKU required')
    replicas = [f['replica'] for f in futures]
    if (len(replicas) != 4 or type(replicas[0]) is not int or
            replicas[0] not in range(0, 32, 4) or
            replicas != list(range(replicas[0], replicas[0]+4))):
        raise ValueError('four consecutive predeclared replicas required')
    if any(len(f['keys']) != 2 or any(type(k) is not int or k < 0 for k in f['keys'])
           for f in futures):
        raise ValueError('two nonnegative integer scenario keys required')
    began_days = budget.days
    prefix = reconstruct(env, covers, key, day, budget)
    before = snapshot(env)
    if expected_before is not None and before != expected_before:
        raise ValueError('reconstructed state differs from frozen state')
    draws = []
    for future in futures:
        branches = _branches(env, covers, day, sku, future['keys'], budget)
        if snapshot(env) != before:
            raise RuntimeError('paired futures mutated original environment')
        draws.append(dict(replica=future['replica'], keys=list(future['keys']), branches=branches))
    if budget.days-began_days != (day-1)+8*(29-day):
        raise RuntimeError('group simulator day count mismatch')
    return dict(key=key, day=day, sku=sku, prefix=prefix, before=before, draws=draws)


def supervised_group(arguments, budget, context):
    """Worker callback: cached inputs/environment, but deterministic reset per unit.

Explicit file hashes are required on every call. A future dispatcher must bind
these plus group/seed identity to its prospective frozen plan.
"""
    import json
    from pathlib import Path
    import numpy as np
    import torch
    from src.experiments.execution_journal import digest
    from src.experiments.research import make_research_env
    from src.demand.identifiable_calibration import load_identifiable_training_calibration
    inputs = arguments['inputs']
    for name in ('calibration', 'ordering'):
        if digest(inputs[name]['path']) != inputs[name]['sha256']:
            raise ValueError(f'changed {name} input')
    identity = json.dumps(inputs, sort_keys=True)
    if context.get('input_identity') != identity:
        torch.set_num_threads(1)
        params = load_identifiable_training_calibration(inputs['calibration']['path'])
        context['covers'] = np.array(json.loads(Path(inputs['ordering']['path']).read_text())['mean'])
        context['env'] = make_research_env('S2', split='train', start=15,
                                          params=params, pricing_scope='per_sku')
        context['input_identity'] = identity
    budget.check()
    return paired_group(context['env'], context['covers'], arguments['key'], arguments['day'],
                        arguments['sku'], arguments['futures'], budget,
                        expected_before=arguments.get('expected_before'))
