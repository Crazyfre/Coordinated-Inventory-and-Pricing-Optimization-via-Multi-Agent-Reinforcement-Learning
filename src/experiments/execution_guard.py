"""Exclusive ownership and durable cumulative time for future work units.

The caller must hold RunLock around journal and clock access. Unfinished
work is terminal; this module never relaunches it or clears its reservation.
"""
import json
import math
import os
import time
from pathlib import Path
from src.experiments.execution_journal import atomic_json


class RunLock:
    def __init__(self,path):self.path=Path(path);self.handle=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.handle=self.path.open('a+b');self.handle.seek(0,2)
        if self.handle.tell()==0:self.handle.write(b'0');self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close();self.handle=None
            raise RuntimeError('campaign already owned by another execution') from exc
        return self
    def __exit__(self,*args):
        if self.handle is not None:
            self.handle.close();self.handle=None


class CumulativeClock:
    def __init__(self,path,identity,caps,clock=time.perf_counter,wall_clock=time.time):
        self.path=Path(path);self.clock=clock;self.wall_clock=wall_clock;self.start=None
        if set(caps)!={'fit','assessment','total'} or any(not math.isfinite(v) or v<=0 for v in caps.values()):
            raise ValueError('positive finite fit/assessment/total caps required')
        plan=dict(identity=identity,caps=caps)
        if self.path.exists():
            self.state=json.loads(self.path.read_text())
            if self.state['plan']!=plan:raise ValueError('frozen time plan mismatch')
            if self.state['status']!='idle' or self.state['pending'] is not None:
                raise RuntimeError('unfinished or failed execution; retry forbidden')
            if self.path.with_name(self.path.name+'.tmp').exists():raise RuntimeError('ambiguous clock write')
        else:
            self.state=dict(plan=plan,used=dict(fit=0.,assessment=0.,total=0.),status='idle',pending=None,completed=[],history=[])
            atomic_json(self.path,self.state)

    def begin(self,item,phase,started_at=None):
        if self.state['status']!='idle' or phase not in ['fit','assessment']:raise RuntimeError('idle clock and known phase required')
        if item in self.state['completed']:raise ValueError('completed time unit cannot repeat')
        caps=self.state['plan']['caps'];used=self.state['used']
        if used[phase]>=caps[phase] or used['total']>=caps['total']:raise TimeoutError('cumulative budget exhausted')
        now=self.clock()
        if started_at is not None and (not math.isfinite(started_at) or started_at>now):raise ValueError('invalid invocation start')
        self.start=now if started_at is None else started_at
        start_wall=self.wall_clock()-(now-self.start)
        self.state.update(status='running',pending=dict(item=item,phase=phase,start_wall=start_wall))
        atomic_json(self.path,self.state)
        return self.start+min(caps[phase]-used[phase],caps['total']-used['total'])

    def finish(self,error=None):
        if self.state['status']!='running' or self.start is None:raise RuntimeError('no running unit')
        # Include durable finalization in the measured interval. A crash here
        # leaves a non-idle state and cannot be mistaken for clean completion.
        self.state['status']='finalizing'
        atomic_json(self.path,self.state)
        elapsed=self.clock()-self.start
        if not math.isfinite(elapsed) or elapsed<0:raise ValueError('invalid elapsed time')
        p=self.state['pending'];used=self.state['used'];caps=self.state['plan']['caps']
        end_wall=self.wall_clock()
        history=self.state['history']
        gap=None if not history else p['start_wall']-history[-1]['end_wall']
        history.append(dict(item=p['item'],phase=p['phase'],elapsed=elapsed,
            start_wall=p['start_wall'],end_wall=end_wall,boundary_gap_seconds=gap))
        used[p['phase']]+=elapsed;used['total']+=elapsed
        over=used[p['phase']]>caps[p['phase']] or used['total']>caps['total']
        if error is not None or over:
            self.state.update(status='failed',error=str(error) if error is not None else 'cumulative time cap')
        else:
            self.state['completed'].append(p['item']);self.state.update(status='idle',pending=None)
        # This final timing receipt cannot include its own write duration.
        # The protocol explicitly excludes only this receipt and lock release.
        atomic_json(self.path,self.state);self.start=None
        if over:raise TimeoutError('cumulative time cap')
