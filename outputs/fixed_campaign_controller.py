"""Future-run controller: fixed frozen sequence, no retries or scientific choices.

Freeze this file's hash into a NEW campaign declaration before use. Never use
this controller to continue a failed or interrupted archive. Launch hidden with
Start-Process and redirect output; status.json is the durable notification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from src.experiments.execution_guard import RunLock
from src.experiments.execution_journal import atomic_json


def checked_packet(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('external packet checksum mismatch')
    packet = json.loads(raw)
    controller_name = Path(__file__).resolve().relative_to(REPOSITORY).as_posix()
    declared_controller = packet['declaration']['study']['source_hashes'].get(controller_name)
    if declared_controller != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('controller must be included in prospective source freeze')
    root = Path(packet['root'])
    if not root.is_absolute() or not root.resolve().is_relative_to(REPOSITORY):
        raise ValueError('archive must be inside repository')
    schedule = packet['declaration']['schedule']
    cursor = 0
    for start, end in schedule:
        if type(start) is not int or type(end) is not int or start != cursor or end <= start:
            raise ValueError('invalid contiguous schedule')
        cursor = end
    if cursor != len(packet['plan']['units']):
        raise ValueError('schedule must cover frozen plan')
    return packet


def completed_batches(packet):
    """Strict clean-boundary check; all substantive checks stay in frozen CLI."""
    root = Path(packet['root'])
    control = root.with_name(root.name+'_invocations')
    receipts = sorted(control.glob('invocation_*.json'))
    schedule = packet['declaration']['schedule']
    if len(receipts) > len(schedule):
        raise ValueError('unexpected invocation count')
    prefix = 0
    for i, path in enumerate(receipts):
        r = json.loads(path.read_text())
        if (path.name != f'invocation_{i+1:06d}.json' or r.get('number') != i+1 or
            r.get('status') != 'complete' or r.get('span') != schedule[i] or
            r.get('completed_before') != prefix or r.get('completed_after') != schedule[i][1] or
            r.get('attempts_before') != prefix or r.get('attempts_after') != schedule[i][1]):
            raise RuntimeError('not a verified clean batch boundary; no automatic recovery')
        prefix = schedule[i][1]
    ledger = root/'attempts.sqlite'
    if ledger.exists():
        db = sqlite3.connect(ledger.resolve().as_uri()+'?mode=ro', uri=True)
        try:
            rows = db.execute('SELECT unit,status FROM attempts ORDER BY id').fetchall()
            saved = json.loads(db.execute('SELECT value FROM plan').fetchone()[0])
        finally:
            db.close()
        if saved != packet['plan'] or rows != [(i,'complete') for i in range(prefix)]:
            raise RuntimeError('ledger does not match complete receipt boundary')
    elif root.exists() or prefix:
        raise RuntimeError('missing ledger or ambiguous archive')
    return len(receipts)


def drive(packet, invoke, inspect=completed_batches, publish=lambda **kw:None):
    count = inspect(packet)
    for index in range(count, len(packet['declaration']['schedule'])):
        publish(status='running', next_batch=index+1)
        if invoke('run', index+1) != 0:
            raise RuntimeError(f'batch {index+1} failed; no retry')
        if inspect(packet) != index+1:
            raise RuntimeError('child exited without exact verified advancement')
        publish(status='batch_complete', completed_batches=index+1)
    publish(status='verifying')
    if invoke('verify', 0) != 0:
        raise RuntimeError('final independent audit failed')
    publish(status='sequence_complete', scientific_success='read final audit gates; not implied')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    packet_path = Path(args.packet).resolve()
    packet = checked_packet(packet_path, args.sha256)
    root = Path(packet['root'])
    directory = root.with_name(root.name+'_controller')
    with RunLock(root.with_name(root.name+'.controller.lock')):
        directory.mkdir(exist_ok=False)  # Never overwrite or auto-resume controller state.
        def publish(**values):
            atomic_json(directory/'status.json', dict(updated_wall=time.time(), **values))
        def invoke(operation, number):
            checked_packet(packet_path, args.sha256)
            cmd = [sys.executable,'-m','scripts.run_recovery_campaign',operation,
                   '--packet',str(packet_path),'--sha256',args.sha256]
            # The frozen runner owns worker cleanup, deadlines and all budget admissions.
            with (directory/f'{operation}_{number:03d}.log').open('x') as log:
                return subprocess.run(cmd, cwd=REPOSITORY, stdout=log,
                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).returncode
        try:
            drive(packet, invoke, publish=publish)
        except BaseException as exc:
            publish(status='stopped', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()
