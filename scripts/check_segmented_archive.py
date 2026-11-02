"""Independent structural checks; this is NOT scientific result verification.

No runner imports, simulator calls, or fitting. Read each large artifact once,
retain only the metadata needed to check the ordered freeze dependencies.
"""
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_archive(root):
    """Check a complete ledger and return metadata, never a passed science flag."""
    root = Path(root).resolve()
    load = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
    manifest, journal, timer = map(load, ['manifest.json', 'journal.json', 'time.json'])
    items = ([f'scan_{i:03d}' for i in range(128)] + ['selection'] +
             [f'branch_{i:03d}' for i in range(128)] + ['fit'] +
             [f'assess_{i:03d}' for i in range(128)] + ['replays', 'summary'])
    caps = dict(fit=7200., assessment=600., total=7800.)
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(manifest['items'] == items and manifest['caps'] == caps and
            manifest['maximum_days'] == 501872, 'manifest plan')
    require(journal['plan'] == dict(identity=identity, items=items, maximum_days=501872), 'journal plan')
    require(timer['plan'] == dict(identity=identity, caps=caps), 'clock plan')
    require(journal['pending'] is None and timer['pending'] is None and timer['status'] == 'idle', 'unfinished execution')
    require(not list(root.rglob('*.tmp')), 'ambiguous atomic write')
    entries = journal['completed']
    require([r['item'] for r in entries] == items and timer['completed'] == items, 'complete ordered ledger required')
    used = timer['used']
    require(set(used) == set(caps), 'clock totals')
    require(all(isinstance(used[k], (int, float)) and math.isfinite(used[k]) and
                0 <= used[k] <= caps[k] for k in caps), 'time budget')
    require(math.isclose(used['fit'] + used['assessment'], used['total'], rel_tol=1e-12, abs_tol=1e-8), 'phase time sum')
    history=timer['history']
    require([r['item'] for r in history]==items, 'complete timing history required')
    reconstructed=dict(fit=0.,assessment=0.,total=0.)
    for index,row in enumerate(history):
        phase='fit' if index<=257 else 'assessment'
        require(row['phase']==phase, 'timing phase')
        require(all(math.isfinite(row[k]) for k in ['elapsed','start_wall','end_wall']) and row['elapsed']>=0, 'timing interval')
        gap=None if index==0 else row['start_wall']-history[index-1]['end_wall']
        require(row['boundary_gap_seconds']==gap, 'timing boundary gap')
        reconstructed[phase]+=row['elapsed'];reconstructed['total']+=row['elapsed']
    require(all(math.isclose(reconstructed[k],used[k],rel_tol=1e-12,abs_tol=1e-8) for k in used), 'timing history sum')
    require(bool(manifest['hashes']), 'empty source freeze')
    for path, expected in manifest['hashes'].items():
        require(sha256(path) == expected, f'changed source: {path}')
    days = 0
    selections = None
    length = None
    for index, entry in enumerate(entries):
        item = items[index]
        require(entry['artifact'] == f'unit_{index:04d}.json', 'artifact order/path')
        path = root / entry['artifact']
        require(path.resolve().is_relative_to(root), 'artifact outside root')
        require(sha256(path) == entry['sha256'], 'artifact hash')
        wrapper = load(entry['artifact'])
        require(wrapper['item'] == item and wrapper['days'] == entry['days'], 'artifact metadata')
        payload = wrapper['payload']
        expected_days = 0
        if item.startswith('scan_'):
            ep = int(item[5:]); expected_days = 28
            require(payload['episode'] == ep and payload['key'] == 2570000 + ep, 'scan namespace')
        elif item == 'selection':
            require(payload['source_hashes'] == manifest['hashes'], 'selection source freeze')
            require(payload['scan_hashes'] == [r['sha256'] for r in entries[:128]], 'scan freeze links')
            selections = payload['selections']
            require(len(selections) == 128, 'selection grid')
            length = 0
            for ep, row in enumerate(selections):
                require(row['episode'] == ep and row['key'] == 2570000 + ep, 'selection namespace/order')
                chosen = row['selected']
                require(len(chosen) == 2 and chosen[0] != chosen[1], 'two distinct candidates required')
                require(all(len(c) == 2 and type(c[0]) is int and type(c[1]) is int and
                            c[0] in [1, 8, 15, 22] and 0 <= c[1] < 10 for c in chosen), 'candidate domain')
                length += sum(29 - d for d, sku in chosen)
            require(length == payload['suffix_length_sum'] and
                    payload['planned_days'] == 21616 + 67 * length <= 501872, 'selection day budget')
        elif item.startswith('branch_'):
            ep = int(item[7:])
            require(payload['episode'] == ep and payload['key'] == 2570000 + ep, 'branch namespace/order')
            expected_days = 28 + 67 * sum(29 - d for d, sku in selections[ep]['selected'])
            require(len(payload['states']) == 2, 'branch state count')
            actual = sorted((s['sample']['rank'], s['sample']['day'], s['sample']['sku']) for s in payload['states'])
            require(actual == [(rank, *chosen) for rank, chosen in enumerate(selections[ep]['selected'])], 'branch selection links')
            require(all(s['sample']['episode'] == ep for s in payload['states']), 'sample episode')
        elif item == 'fit':
            require(payload['source_hashes'] == manifest['hashes'], 'fit source freeze')
            require(payload['selection_hash'] == entries[128]['sha256'], 'selection freeze link')
            require(payload['branch_hashes'] == [r['sha256'] for r in entries[129:257]], 'branch freeze links')
            require(set(payload['models']) == {'local', 'cross'}, 'both models required')
            require([(s['episode'], s['rank']) for s in payload['samples']] ==
                    [(ep, rank) for ep in range(128) for rank in [0, 1]], 'fit sample grid')
        elif item.startswith('assess_') or item == 'replays':
            ep = 0 if item == 'replays' else int(item[7:]); expected_days = 112
            require(payload['episode'] == ep, 'assessment episode')
            require([(r['label'], r['key']) for r in payload['records']] ==
                    [(label, 2590000 + ep) for label in ['cross', 'local', 'baseline', 'heuristic']], 'assessment grid')
        elif item == 'summary':
            require(payload['status'] == 'complete' and payload['test_scoring'] is False, 'summary status')
            require(payload['days'] == days and payload['suffix_length_sum'] == length, 'summary accounting')
        require(type(entry['days']) is int and entry['days'] == expected_days, 'unit day count')
        days += expected_days
    require(days == journal['reserved_days'] == 21616 + 67 * length <= 501872, 'total day count')
    return dict(status='structure_checked_only', units=388, days=days,
                scientific_verification=False, new_simulator_days=0)
