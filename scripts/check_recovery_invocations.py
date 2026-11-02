"""Independent complete-invocation accounting, separate from scientific audit.

Interrupted elapsed time remains unknown; reservations are never substituted
for measured elapsed time. Does not certify operational caps or science alone.
"""
import json
import math
from pathlib import Path
import sqlite3


def check(root,declaration,plan):
    root=Path(root).resolve();control=root.with_name(root.name+'_invocations')
    if json.loads((control/'declaration.json').read_text())!=declaration:
        raise ValueError('invocation declaration mismatch')
    schedule=declaration['schedule'];cursor=0
    for start,end in schedule:
        if start!=cursor or end<=start:raise ValueError('invalid invocation schedule')
        cursor=end
    if cursor!=len(plan['units']):raise ValueError('incomplete invocation schedule')
    db=sqlite3.connect((root/'attempts.sqlite').as_uri()+'?mode=ro',uri=True)
    try:
        rows=db.execute('SELECT id,unit,status,seconds FROM attempts ORDER BY id').fetchall()
    finally:db.close()
    if [r[0] for r in rows]!=list(range(1,len(rows)+1)):
        raise ValueError('attempt sequence mismatch')
    receipts=[]
    for ordinal,path in enumerate(sorted(control.glob('invocation_*.json')),1):
        record=json.loads(path.read_text())
        if path.name!=f'invocation_{ordinal:06d}.json' or record['number']!=ordinal:
            raise ValueError('invocation sequence mismatch')
        if record['status'] not in ('complete','interrupted'):
            raise ValueError('running or failed invocation cannot verify')
        receipts.append(record)
    if not receipts or receipts[0].get('attempts_before')!=0:
        raise ValueError('missing initial attempt boundary')
    known=0.;tails=0.;unknown=0;covered=0
    phase_tails=dict(fit=0.,assessment=0.)
    for position,record in enumerate(receipts):
        before=record.get('attempts_before')
        after=(receipts[position+1].get('attempts_before') if position+1<len(receipts) else len(rows))
        if type(before) is not int or type(after) is not int or before!=covered or not before<=after<=len(rows):
            raise ValueError('invalid invocation attempt boundaries')
        prefix=sum(r[2]=='complete' for r in rows[:before])
        if record['completed_before']!=prefix:
            raise ValueError('invocation prefix differs from ledger')
        expected=([prefix,prefix] if prefix==len(plan['units']) else
                  next((span for span in schedule if span[0]<=prefix<span[1]),None))
        if record.get('span')!=expected or expected is None:
            raise ValueError('invocation span mismatch')
        subset=rows[before:after]
        if any(not prefix<=r[1]<expected[1] for r in subset):
            raise ValueError('attempt outside invocation span')
        if record['status']=='complete':
            if record.get('attempts_after')!=after or any(r[2]!='complete' for r in subset):
                raise ValueError('completed invocation has incomplete attempts')
            if [r[1] for r in subset]!=list(range(prefix,expected[1])) or record['completed_after']!=expected[1]:
                raise ValueError('completed invocation does not finish declared span')
            elapsed=record['elapsed_seconds'];tail=record['trailing_seconds']
            if not all(math.isfinite(v) and v>=0 for v in (elapsed,tail)):
                raise ValueError('invalid invocation elapsed time')
            if abs(elapsed-sum(r[3] for r in subset)-tail)>1e-5:
                raise ValueError('invocation elapsed does not reconcile with attempts and tail')
            known+=elapsed;tails+=tail
            phase_tails[plan['units'][expected[1]-1]['phase']]+=tail
        else:
            if record['elapsed_seconds'] is not None:
                raise ValueError('interrupted elapsed must remain unknown')
            unknown+=1
        covered=after
    if covered!=len(rows) or sum(r[2]=='complete' for r in rows)!=len(plan['units']):
        raise ValueError('incomplete campaign invocation accounting')
    overhead_checked=False;overhead_totals=None
    if 'overhead' in declaration:
        policy=declaration['overhead']
        if (set(policy)!={'per_invocation_seconds','maximum_invocations','fit','assessment','total'} or
                type(policy['maximum_invocations']) is not int or policy['maximum_invocations']<1 or
                any(not math.isfinite(policy[k]) or policy[k]<=0 for k in ('per_invocation_seconds','fit','assessment','total'))):
            raise ValueError('invalid overhead policy')
        if len(receipts)>policy['maximum_invocations']:raise ValueError('invocation cap exceeded')
        overhead_totals=dict(fit=0.,assessment=0.,total=0.)
        for record in receipts:
            phase=plan['units'][record['span'][1]-1]['phase']
            # Independently check admission reservation before any refund.
            reserved=dict(overhead_totals)
            reserved[phase]+=policy['per_invocation_seconds'];reserved['total']+=policy['per_invocation_seconds']
            if any(reserved[k]>policy[k] for k in reserved):raise ValueError('overhead admission exceeded cap')
            before=record['attempts_before']
            position=record['number']-1
            after=receipts[position+1]['attempts_before'] if position+1<len(receipts) else len(rows)
            combined=dict(reserved)
            for _,unit_index,_,seconds in rows[:before]:
                combined[plan['units'][unit_index]['phase']]+=seconds
                combined['total']+=seconds
            if any(combined[k]>plan['caps'][k] for k in combined):
                raise ValueError('combined invocation admission exceeded cap')
            for _,unit_index,_,seconds in rows[before:after]:
                unit=plan['units'][unit_index];phase=unit['phase']
                trial=dict(combined);trial[phase]+=unit['seconds'];trial['total']+=unit['seconds']
                if any(trial[k]>plan['caps'][k] for k in trial):
                    raise ValueError('combined attempt admission exceeded cap')
                combined[phase]+=seconds;combined['total']+=seconds
            cost=record['trailing_seconds'] if record['status']=='complete' else policy['per_invocation_seconds']
            if not 0<=cost<=policy['per_invocation_seconds']:raise ValueError('tail exceeded overhead reservation')
            overhead_totals[phase]+=cost;overhead_totals['total']+=cost
        overhead_checked=True
    return dict(invocation_accounting_checked=True,scientific_verification=False,
                overhead_limits_checked=overhead_checked,charged_overhead_seconds=overhead_totals,
                combined_budget_checked=overhead_checked,
                known_invocation_seconds=known,trailing_seconds=tails,
                trailing_seconds_by_phase=phase_tails,
                unknown_elapsed_invocations=unknown,
                exact_total_elapsed_known=unknown==0,
                final_receipt_persistence_excluded=True)
