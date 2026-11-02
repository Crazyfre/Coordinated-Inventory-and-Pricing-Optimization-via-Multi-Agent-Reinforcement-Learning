import json
import sqlite3
import pytest
from scripts.check_recovery_invocations import check


def fixture(tmp_path,interrupted=False):
    root=tmp_path/'archive';root.mkdir();control=tmp_path/'archive_invocations';control.mkdir()
    declaration=dict(schedule=[[0,2]])
    (control/'declaration.json').write_text(json.dumps(declaration))
    with sqlite3.connect(root/'attempts.sqlite') as db:
        db.execute('CREATE TABLE attempts(id INTEGER,unit INTEGER,status TEXT,seconds REAL)')
        if interrupted:db.execute("INSERT INTO attempts VALUES(1,0,'abandoned',10.)")
        offset=int(interrupted)
        db.executemany('INSERT INTO attempts VALUES(?,?,?,?)',[(offset+1,0,'complete',2.),(offset+2,1,'complete',3.)])
    records=[]
    if interrupted:records.append(dict(number=1,status='interrupted',attempts_before=0,completed_before=0,
                                      span=[0,2],elapsed_seconds=None))
    records.append(dict(number=len(records)+1,status='complete',attempts_before=int(interrupted),
                        attempts_after=2+int(interrupted),completed_before=0,completed_after=2,
                        span=[0,2],elapsed_seconds=5.25,trailing_seconds=.25))
    for r in records:(control/f"invocation_{r['number']:06d}.json").write_text(json.dumps(r))
    return root,declaration,dict(units=[{'phase':'fit','seconds':10.},{'phase':'fit','seconds':10.}],
                                 caps=dict(fit=100.,assessment=100.,total=200.)),control


@pytest.mark.parametrize('interrupted',[False,True])
def test_known_and_unknown_invocation_accounting(tmp_path,interrupted):
    root,d,p,_=fixture(tmp_path,interrupted)
    result=check(root,d,p)
    assert result['known_invocation_seconds']==5.25
    assert result['unknown_elapsed_invocations']==int(interrupted)


@pytest.mark.parametrize('field,value',[('elapsed_seconds',4.),('attempts_after',1),('completed_before',1),('trailing_seconds',0.)])
def test_timing_boundary_tampering(tmp_path,field,value):
    root,d,p,control=fixture(tmp_path)
    path=control/'invocation_000001.json';record=json.loads(path.read_text());record[field]=value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):check(root,d,p)


def test_independent_overhead_reservations(tmp_path):
    root,d,p,control=fixture(tmp_path,True)
    d['overhead']=dict(per_invocation_seconds=2.,maximum_invocations=3,fit=5.,assessment=5.,total=6.)
    (control/'declaration.json').write_text(json.dumps(d))
    result=check(root,d,p)
    assert result['overhead_limits_checked'] and result['charged_overhead_seconds']['total']==2.25
    assert result['combined_budget_checked']
    p['caps']['fit']=15.
    with pytest.raises(ValueError,match='combined'):check(root,d,p)
    p['caps']['fit']=100.
    d['overhead']['fit']=3.
    (control/'declaration.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match='admission'):check(root,d,p)
