"""Fail-closed, single-writer journal for future segmented experiments.

Not a resume mechanism for any previously launched scientific protocol.
One process must own the journal; this class does not provide a process lock.
"""
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('x',encoding='utf-8') as f:
        json.dump(value,f,allow_nan=False,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


class ExecutionJournal:
    def __init__(self,root,identity,items,maximum_days):
        self.root=Path(root);self.path=self.root/'journal.json'
        if len(set(items))!=len(items) or not items or maximum_days<=0:raise ValueError('unique nonempty work plan and positive budget required')
        self.plan=dict(identity=identity,items=list(items),maximum_days=maximum_days)
        if self.path.exists():
            self.state=json.loads(self.path.read_text(encoding='utf-8'))
            if self.state['plan']!=self.plan:raise ValueError('frozen execution plan mismatch')
            if self.path.with_name('journal.json.tmp').exists():raise RuntimeError('unresolved atomic journal write')
            if self.state['pending'] is not None:raise RuntimeError('interrupted work item; automatic retry forbidden')
            for done in self.state['completed']:
                p=self.root/done['artifact']
                if not p.is_file() or digest(p)!=done['sha256']:raise ValueError('committed artifact missing or changed')
        else:
            self.root.mkdir(parents=True,exist_ok=False)
            self.state=dict(plan=self.plan,pending=None,completed=[],reserved_days=0)
            atomic_json(self.path,self.state)

    def claim(self,item,days):
        i=len(self.state['completed'])
        if self.state['pending'] is not None:raise RuntimeError('work already pending')
        if i>=len(self.plan['items']) or item!=self.plan['items'][i]:raise ValueError('next declared item required')
        if type(days) is not int or days<0 or self.state['reserved_days']+days>self.plan['maximum_days']:
            raise ValueError('work reservation exceeds budget')
        self.state['pending']=dict(item=item,days=days)
        self.state['reserved_days']+=days
        atomic_json(self.path,self.state)

    def commit(self,artifact,executed_days):
        pending=self.state['pending']
        if pending is None:raise RuntimeError('no claimed work')
        if executed_days!=pending['days']:raise ValueError('exact predeclared day count required')
        p=Path(artifact).resolve();root=self.root.resolve()
        if not p.is_relative_to(root) or p==self.path.resolve() or not p.is_file():raise ValueError('artifact must be a file inside run directory')
        entry=dict(**pending,artifact=p.relative_to(root).as_posix(),sha256=digest(p))
        self.state['completed'].append(entry);self.state['pending']=None
        atomic_json(self.path,self.state)
