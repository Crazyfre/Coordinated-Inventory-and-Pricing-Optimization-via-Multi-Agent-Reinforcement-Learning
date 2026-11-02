"""Future campaign plan builder; no default namespaces, budgets, or launch.

All 2436 unit identities exist before collection. Maxima reserve conservative
steps even before observable selection reveals the actual weekly states.
"""
import hashlib
import json
import math


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def build(study):
    required = {'namespaces', 'seconds', 'caps', 'source_hashes', 'inputs', 'callback'}
    if set(study) != required:
        raise ValueError('complete explicit frozen study required')
    ns = study['namespaces']
    if set(ns) != {'scan','future','assessment','bootstrap'} or any(type(v) is not int or v < 0 for v in ns.values()):
        raise ValueError('four explicit seed namespaces required')
    if len(set(ns.values())) != 4 or abs(ns['scan']-ns['assessment']) < 128:
        raise ValueError('distinct nonoverlapping original/assessment namespaces required')
    kinds = {'scan','selection','verify','group','fit','assessment','replays','summary'}
    if set(study['seconds']) != kinds or any(not math.isfinite(v) or v <= 0 for v in study['seconds'].values()):
        raise ValueError('positive explicit timeout for every unit kind required')
    if not study['source_hashes'] or not study['inputs'] or set(study['callback']) != {'module','function'}:
        raise ValueError('frozen sources, inputs and callback required')
    study_hash = identity(study)
    units = []
    def add(item, kind, days, dependencies, **parameters):
        specification = dict(item=item, kind=kind, dependencies=dependencies, **parameters)
        units.append(dict(item=item, phase='assessment' if kind in {'assessment','replays','summary'} else 'fit',
                          days=days, seconds=study['seconds'][kind],
                          input_identity=identity(dict(study=study_hash, specification=specification)),
                          specification=specification))
    for ep in range(128):
        add(f'scan_{ep:03d}', 'scan', 28, [], episode=ep)
    add('selection', 'selection', 0, [f'scan_{ep:03d}' for ep in range(128)])
    fit_dependencies = ['selection']
    for ep in range(128):
        verification = f'verify_{ep:03d}'
        add(verification, 'verify', 196, ['selection', f'scan_{ep:03d}'], episode=ep)
        fit_dependencies.append(verification)
        for rank in range(2):
            for group in range(8):
                item = f'group_{ep:03d}_{rank}_{group}'
                add(item, 'group', 224, ['selection', f'scan_{ep:03d}', verification],
                    episode=ep, rank=rank, group=group)
                fit_dependencies.append(item)
    add('fit', 'fit', 0, fit_dependencies)
    for ep in range(128):
        add(f'assess_{ep:03d}', 'assessment', 112, ['fit'], episode=ep)
    add('replays', 'replays', 112, ['fit','assess_000'])
    add('summary', 'summary', 0, [f'assess_{ep:03d}' for ep in range(128)]+['replays'])
    return dict(identity=study_hash, units=units, caps=study['caps'], max_attempts=3)


def bind_request(study, unit, dependency_records):
    """Bind actual committed dependency hashes to the predeclared dependency IDs."""
    specification = unit['specification']
    expected = next((u for u in build(study)['units'] if u['item']==unit['item']),None)
    if unit != expected:
        raise ValueError('unit differs from generated immutable plan')
    if identity(dict(study=identity(study), specification=specification)) != unit['input_identity']:
        raise ValueError('unit differs from immutable study')
    if [r['item'] for r in dependency_records] != specification['dependencies']:
        raise ValueError('exact ordered dependency records required')
    for record in dependency_records:
        if set(record) != {'item','path','sha256','input_identity'}:
            raise ValueError('committed dependency path/hash/identity required')
    return dict(study=study, unit=unit, dependencies=dependency_records)
