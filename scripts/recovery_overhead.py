"""Prospective overhead reservation accounting; never measures unknown time."""
import math
import json
import sqlite3
from pathlib import Path


def validate(policy):
    if set(policy)!={'per_invocation_seconds','maximum_invocations','fit','assessment','total'}:
        raise ValueError('complete explicit overhead policy required')
    if type(policy['maximum_invocations']) is not int or policy['maximum_invocations']<1:
        raise ValueError('positive invocation limit required')
    if any(not math.isfinite(policy[k]) or policy[k]<=0 for k in ('per_invocation_seconds','fit','assessment','total')):
        raise ValueError('positive finite overhead limits required')


def charges(records,policy,plan):
    validate(policy)
    totals=dict(fit=0.,assessment=0.,total=0.)
    if len(records)>policy['maximum_invocations']:
        raise RuntimeError('invocation limit exceeded')
    for record in records:
        span=record.get('span')
        if not span:raise ValueError('overhead record lacks declared span')
        phase=plan['units'][max(0,span[1]-1)]['phase']
        amount=policy['per_invocation_seconds']
        # Never refund interrupted or running overhead. A completed invocation
        # only incurs its measured tail; all earlier work is in attempt seconds.
        if record['status']=='complete':
            amount=record['trailing_seconds']
            if not math.isfinite(amount) or not 0<=amount<=policy['per_invocation_seconds']:
                raise RuntimeError('invocation tail exceeded reserved overhead')
        elif record['status'] not in ('running','interrupted'):
            raise RuntimeError('failed invocation cannot continue')
        totals[phase]+=amount;totals['total']+=amount
    if any(totals[k]>policy[k] for k in totals):
        raise RuntimeError('cumulative overhead budget exceeded')
    return totals


def combined_admission(root,plan,overhead,next_index=None):
    """Charge ledger reservations and overhead against the SAME frozen caps."""
    totals=dict(overhead)
    path=Path(root)/'attempts.sqlite'
    if path.exists():
        db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            saved=db.execute('SELECT value FROM plan').fetchall()
            if len(saved)!=1 or json.loads(saved[0][0])!=plan:
                raise ValueError('combined budget plan mismatch')
            rows=db.execute('SELECT unit,seconds FROM attempts ORDER BY id').fetchall()
        finally:db.close()
        for index,seconds in rows:
            totals[plan['units'][index]['phase']]+=seconds
            totals['total']+=seconds
    if next_index is not None:
        unit=plan['units'][next_index]
        totals[unit['phase']]+=unit['seconds'];totals['total']+=unit['seconds']
    if any(totals[k]>plan['caps'][k] for k in totals):
        raise RuntimeError('combined attempt and overhead budget exceeded')
    return totals
