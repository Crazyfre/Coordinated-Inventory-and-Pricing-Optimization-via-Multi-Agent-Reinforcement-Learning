"""Independent request/dependency binding checks; not scientific verification.

Consumes externally frozen study AND plan; does not import runner/dispatcher.
Full independent scientific plan reconstruction remains a separate requirement.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
from scripts.check_recovery_attempts import check as check_attempts


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def file_hash(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            hasher.update(block)
    return hasher.hexdigest()


def check(root,study,plan,require_complete=True):
    root=Path(root).resolve()
    if plan['identity']!=canonical_hash(study) or plan['caps']!=study['caps']:
        raise ValueError('external study/plan identity mismatch')
    report=check_attempts(root,plan,require_complete=require_complete)
    for path,checksum in study['source_hashes'].items():
        if file_hash(path)!=checksum:
            raise ValueError('frozen source changed')
    for name in ('ordering','calibration'):
        record=study['inputs'][name]
        if file_hash(record['path'])!=record['sha256']:
            raise ValueError('frozen scientific input changed')
    db=sqlite3.connect((root/'attempts.sqlite').as_uri()+'?mode=ro',uri=True)
    try:
        rows=db.execute("SELECT id,unit,artifact,sha256 FROM attempts WHERE status='complete' ORDER BY id").fetchall()
    finally:
        db.close()
    committed={}
    for aid,index,relative,checksum in rows:
        unit=plan['units'][index]
        expected_identity=canonical_hash(dict(study=plan['identity'],specification=unit['specification']))
        if expected_identity!=unit['input_identity']:
            raise ValueError('unit input identity mismatch')
        path=root/relative
        artifact=json.loads(path.read_text(encoding='utf-8'))
        receipt=artifact.get('worker_receipt')
        if receipt is None:
            raise ValueError('scientific artifact missing supervised receipt')
        request=receipt['request']
        if (request['module']!=study['callback']['module'] or
                request['function']!=study['callback']['function']):
            raise ValueError('worker callback mismatch')
        arguments=request['arguments']
        if arguments['study']!=study or arguments['unit']!=unit:
            raise ValueError('worker study/unit mismatch')
        expected=[]
        for item in unit['specification']['dependencies']:
            if item not in committed:
                raise ValueError('dependency not committed before consumer')
            expected.append(committed[item])
        if arguments['dependencies']!=expected:
            raise ValueError('worker dependencies differ from committed artifacts')
        if request['maximum_days']!=unit['days']:
            raise ValueError('worker day bound differs from reserved unit')
        committed[unit['item']]=dict(item=unit['item'],path=str(path.resolve()),sha256=checksum,
                                    input_identity=unit['input_identity'])
    return dict(**report,request_dependency_bindings_checked=True,
                independent_scientific_plan_checked=False)
