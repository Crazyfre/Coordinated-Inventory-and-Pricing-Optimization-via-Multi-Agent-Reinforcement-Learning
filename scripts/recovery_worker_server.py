"""Internal sequential server; only launched inside an owned Windows job."""
import importlib
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from src.experiments.execution_journal import digest
from src.experiments.recovery_json import atomic_json
from src.experiments.recovery_execution import StepBudget
from src.experiments.transport_read import read_bytes


def run(root):
    root = Path(root)
    sequence = 1
    context = {}  # Cache only. Scientific callbacks must reset all per-unit state.
    while True:
        request_path = root/f'request_{sequence:06d}.json'
        started = time.perf_counter()
        checksum = None
        try:
            while not request_path.exists():
                time.sleep(.01)
            started = time.perf_counter()
            # Until the request is readable its deadline is unknown. Limit this
            # admission read to1s; parent's existing request deadline still kills
            # the worker sooner if necessary. No extra campaign time is granted.
            raw = read_bytes(request_path,started+1.)
            request = json.loads(raw)
            checksum = hashlib.sha256(raw).hexdigest()
            response = dict(sequence=sequence, request_sha256=checksum, started=started)
            if request['sequence'] != sequence:
                raise ValueError('out-of-sequence worker request')
            budget = StepBudget(request['maximum_days'], request['deadline'], time.perf_counter)
            budget.check()
            callback = getattr(importlib.import_module(request['module']), request['function'])
            budget.check()
            payload = callback(request['arguments'], budget, context)
            budget.check()
            response.update(status='complete', days=budget.days, ended=time.perf_counter(), payload=payload)
            atomic_json(root/f'response_{sequence:06d}.json', response)
        except BaseException as exc:
            # A failed payload serialization may leave response.json.tmp. Use
            # a separate filename and exclude the offending payload entirely.
            diagnostic=dict(sequence=sequence,request_sha256=checksum,status='failed',
                started=started,ended=time.perf_counter(),error=f'{type(exc).__name__}: {exc}',
                traceback=traceback.format_exc())
            atomic_json(root/f'error_{sequence:06d}.json', diagnostic)
            return 1
        sequence += 1


if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1]))
