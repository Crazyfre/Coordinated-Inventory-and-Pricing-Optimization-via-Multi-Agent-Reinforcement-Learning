"""Future-only transactional attempt accounting; caller MUST hold an OS lock.

Not connected to any scientific campaign. Existing terminal archives cannot
use this ledger. Full reservations survive interrupted attempts.
"""
import json
import math
import sqlite3
from pathlib import Path


class RecoveryLedger:
    def __init__(self, path, identity, units, caps, max_attempts=3):
        if not units or len({u['item'] for u in units}) != len(units):
            raise ValueError('unique ordered units required')
        if set(caps) != {'days','fit','assessment','total'}:
            raise ValueError('day and phase/total time caps required')
        if any(not math.isfinite(v) or v <= 0 for v in caps.values()):
            raise ValueError('positive finite caps required')
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError('positive integer attempt limit required')
        for u in units:
            if (u['phase'] not in ['fit','assessment'] or type(u['days']) is not int
                    or u['days'] < 0 or not math.isfinite(u['seconds']) or u['seconds'] <= 0
                    or not u['input_identity']):
                raise ValueError('bounded units with immutable inputs required')
        self.plan=dict(identity=identity,units=units,caps=caps,max_attempts=max_attempts)
        encoded=json.dumps(self.plan,sort_keys=True,allow_nan=False)
        path=Path(path);exists=path.exists()
        if exists and path.stat().st_size == 0:
            raise ValueError('empty existing ledger is ambiguous')
        self.db=sqlite3.connect(path)
        self.db.execute('PRAGMA synchronous=FULL')
        try:
            if not exists:
                with self.db:
                    self.db.execute('CREATE TABLE plan (value TEXT NOT NULL)')
                    self.db.execute('INSERT INTO plan VALUES (?)',(encoded,))
                    self.db.execute('CREATE TABLE attempts (id INTEGER PRIMARY KEY, unit INTEGER NOT NULL, '
                        'status TEXT NOT NULL, days INTEGER NOT NULL, seconds REAL NOT NULL, '
                        'artifact TEXT, sha256 TEXT)')
            rows=self.db.execute('SELECT value FROM plan').fetchall()
            if rows != [(encoded,)]:raise ValueError('immutable recovery plan mismatch')
            if self.db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                raise ValueError('ledger integrity failure')
        except BaseException:
            self.db.close();raise

    def close(self):self.db.close()

    def attempts(self):
        fields=['id','unit','status','days','seconds','artifact','sha256']
        return [dict(zip(fields,r)) for r in self.db.execute('SELECT * FROM attempts ORDER BY id')]

    def totals(self):
        out=dict(days=0,fit=0.,assessment=0.,total=0.)
        for r in self.attempts():
            phase=self.plan['units'][r['unit']]['phase']
            out['days']+=r['days'];out[phase]+=r['seconds'];out['total']+=r['seconds']
        return out

    def recover(self):
        """Only after exclusive ownership proves the previous worker is gone."""
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            if any(r['status']=='failed' for r in self.attempts()):
                raise RuntimeError('terminal failed attempt')
            # Never refund or adopt an uncommitted output after a shutdown.
            self.db.execute("UPDATE attempts SET status='abandoned' WHERE status='running'")

    def begin(self, item, input_identity):
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            rows=self.attempts()
            if any(r['status'] in ['running','failed'] for r in rows):
                raise RuntimeError('pending recovery or terminal failure')
            done=[r['unit'] for r in rows if r['status']=='complete']
            if done != list(range(len(done))):raise ValueError('invalid completed prefix')
            index=len(done)
            if index >= len(self.plan['units']):raise ValueError('all units complete')
            u=self.plan['units'][index]
            if item!=u['item'] or input_identity!=u['input_identity']:
                raise ValueError('next declared unit and identical inputs required')
            if sum(r['unit']==index for r in rows)>=self.plan['max_attempts']:
                raise RuntimeError('attempt limit reached')
            total=self.totals();total['days']+=u['days']
            total[u['phase']]+=u['seconds'];total['total']+=u['seconds']
            if any(total[k]>v for k,v in self.plan['caps'].items()):
                raise RuntimeError('reservation exceeds cumulative budget')
            cur=self.db.execute('INSERT INTO attempts(unit,status,days,seconds) VALUES (?,?,?,?)',
                (index,'running',u['days'],u['seconds']))
            return cur.lastrowid

    def finish(self, attempt, days, seconds, artifact, sha256, failed=False):
        """Caller validates/durably writes artifact first; failed cost is not refunded."""
        invalid=False
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            rows=[r for r in self.attempts() if r['id']==attempt]
            if len(rows)!=1 or rows[0]['status']!='running':
                raise ValueError('running attempt required')
            r=rows[0]
            invalid=(type(days) is not int or days<0 or days>r['days'] or
                not math.isfinite(seconds) or seconds<0 or seconds>r['seconds'] or
                not artifact or not isinstance(sha256,str) or len(sha256)!=64 or
                any(c not in '0123456789abcdef' for c in sha256))
            if failed or invalid:
                self.db.execute("UPDATE attempts SET status='failed' WHERE id=?",(attempt,))
            else:
                self.db.execute("UPDATE attempts SET status='complete',days=?,seconds=?,artifact=?,sha256=? WHERE id=?",
                    (days,seconds,artifact,sha256,attempt))
        if invalid:raise ValueError('invalid completion; terminal failure recorded')
