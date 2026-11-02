"""Read-only independent attempt accounting; never a scientific verifier."""
import hashlib
import json
import math
import sqlite3
from pathlib import Path


def check(root, expected_plan, require_complete=False):
    root = Path(root).resolve()
    db = sqlite3.connect((root/'attempts.sqlite').as_uri()+'?mode=ro', uri=True)
    try:
        if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
            raise ValueError('database integrity failure')
        plans = db.execute('SELECT value FROM plan').fetchall()
        if len(plans) != 1 or json.loads(plans[0][0]) != expected_plan:
            raise ValueError('plan differs from externally frozen plan')
        rows = db.execute('SELECT id,unit,status,days,seconds,artifact,sha256 '
                          'FROM attempts ORDER BY id').fetchall()
    finally:
        db.close()
    units = expected_plan['units']
    totals = dict(days=0, fit=0., assessment=0., total=0.)
    by_status = {s: dict(days=0, seconds=0.) for s in ('complete','abandoned','failed','running')}
    next_unit = 0
    count = 0
    terminal = False
    observed_overrun = 0.
    known_elapsed = 0.
    unknown_elapsed = 0
    for ordinal, (aid, ui, status, days, seconds, artifact, checksum) in enumerate(rows, 1):
        if terminal or aid != ordinal or ui != next_unit or ui >= len(units):
            raise ValueError('attempt sequence is not an ordered prefix')
        unit = units[ui]
        count += 1
        if count > expected_plan['max_attempts'] or status not in by_status:
            raise ValueError('invalid status or retry limit')
        if (type(days) is not int or not 0 <= days <= unit['days'] or
                not math.isfinite(seconds) or not 0 <= seconds <= unit['seconds']):
            raise ValueError('invalid attempt charge')
        # The full reservation had to fit BEFORE the actual-cost refund.
        reservation = dict(totals)
        reservation['days'] += unit['days']
        reservation[unit['phase']] += unit['seconds']
        reservation['total'] += unit['seconds']
        if any(reservation[k] > cap for k, cap in expected_plan['caps'].items()):
            raise ValueError('attempt could not have reserved within caps')
        if status == 'complete':
            known_elapsed += seconds
            if artifact != f'attempt_{aid:06d}.json':
                raise ValueError('unexpected artifact path')
            data = (root/artifact).read_bytes()
            if hashlib.sha256(data).hexdigest() != checksum:
                raise ValueError('artifact hash mismatch')
            payload = json.loads(data)
            if (payload['item'] != unit['item'] or
                    payload['input_identity'] != unit['input_identity'] or payload['days'] != days):
                raise ValueError('artifact identity or day count mismatch')
            if 'worker_receipt' in payload:
                receipt = payload['worker_receipt']
                request = receipt['request']
                encoded = json.dumps(request, allow_nan=False, sort_keys=True).encode('utf-8')
                stamps = [request['dispatched'], receipt['started'], receipt['ended'],
                          receipt['received'], request['deadline']]
                if (hashlib.sha256(encoded).hexdigest() != receipt['request_sha256'] or
                        request['sequence'] != receipt['sequence'] or receipt['days'] != days or
                        not 0 <= days <= request['maximum_days'] <= unit['days'] or
                        not all(math.isfinite(v) for v in stamps) or stamps != sorted(stamps) or
                        receipt['received']-request['dispatched'] > seconds+1e-6):
                    raise ValueError('worker receipt identity, timing or step mismatch')
            next_unit += 1
            count = 0
        else:
            if days != unit['days'] or seconds != unit['seconds'] or artifact is not None or checksum is not None:
                raise ValueError('unfinished attempt must retain full reservation')
            terminal = status in ('failed', 'running')
            failure_path = root/f'failure_{aid:06d}.json'
            if status == 'failed' and failure_path.exists():
                failure = json.loads(failure_path.read_text(encoding='utf-8'))
                elapsed = failure['observed_seconds']
                extra = failure['observed_overrun_seconds']
                if (failure['attempt'] != aid or failure['reserved_seconds'] != unit['seconds'] or
                        not math.isfinite(elapsed) or elapsed < 0 or
                        extra != max(0., elapsed-unit['seconds'])):
                    raise ValueError('invalid failure elapsed receipt')
                observed_overrun += extra
                known_elapsed += elapsed
            else:
                unknown_elapsed += 1
        totals['days'] += days
        totals[unit['phase']] += seconds
        totals['total'] += seconds
        by_status[status]['days'] += days
        by_status[status]['seconds'] += seconds
    complete = next_unit == len(units) and not terminal
    if require_complete and not complete:
        raise ValueError('complete committed campaign required')
    return dict(structure_checked_only=True, scientific_verification=False,
                complete=complete, completed_units=next_unit,
                attempts=len(rows), charged=totals, by_status=by_status,
                observed_failure_overrun_seconds=observed_overrun,
                known_elapsed_seconds=known_elapsed, unknown_elapsed_attempts=unknown_elapsed,
                reservation_seconds_are_not_an_actual_runtime_upper_bound=True)
