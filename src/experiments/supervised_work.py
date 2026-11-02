"""Persistent owned worker for future-only ledger callbacks.

Lazy launch occurs inside a reserved attempt, so imports/setup are charged.
One sequential command at a time; scratch outputs never become committed data
without execute_attempt's durable commit. The caller freezes callback sources.
"""
import json
import hashlib
import math
import shutil
from pathlib import Path
import sys
import tempfile
import time
import traceback
from src.experiments.execution_journal import atomic_json, digest
from src.experiments.windows_worker import OwnedWorker
from src.experiments.transport_read import read_bytes


class SupervisedWork:
    def __init__(self, scratch_parent):
        self.scratch_parent = Path(scratch_parent)
        self.worker = None
        self.temporary = None
        self.sequence = 0
        self.failed = False
        self.failure = None
        self.failure_path = None

    def call(self, module, function, arguments, budget):
        if self.failed:
            raise RuntimeError('failed supervised worker cannot be reused')
        budget.check()
        try:
            if self.worker is None:
                self.scratch_parent.mkdir(parents=True, exist_ok=True)
                self.temporary = Path(tempfile.mkdtemp(prefix='worker_', dir=self.scratch_parent))
                self.worker = OwnedWorker([sys.executable, '-m', 'scripts.recovery_worker_server',
                                           str(self.temporary)], cwd=str(Path.cwd()))
            self.sequence += 1
            root = self.temporary
            request_path = root/f'request_{self.sequence:06d}.json'
            response_path = root/f'response_{self.sequence:06d}.json'
            request = dict(sequence=self.sequence, module=module, function=function,
                           arguments=arguments, maximum_days=budget.maximum-budget.days,
                           deadline=budget.deadline, dispatched=time.perf_counter())
            atomic_json(request_path, request)
            checksum = hashlib.sha256(read_bytes(request_path,budget.deadline)).hexdigest()
            while not response_path.exists():
                budget.check()  # On timeout the except block kills the whole job.
                code = self.worker.poll()
                if code is not None:
                    # A response can be atomically published between exists()
                    # and poll(). Inspect it before reporting an unexplained exit.
                    if response_path.exists():
                        break
                    diagnostic = root/f'error_{self.sequence:06d}.json'
                    if diagnostic.exists():
                        detail = json.loads(read_bytes(diagnostic,budget.deadline))
                        raise RuntimeError(f"worker failed: {detail.get('error', 'unknown')}")
                    raise RuntimeError(f'supervised worker exited before response: {code}')
                time.sleep(min(.01, max(0., budget.deadline-time.perf_counter())))
            budget.check()
            response = json.loads(read_bytes(response_path,budget.deadline))
            received = time.perf_counter()
            if response.get('request_sha256') != checksum or response.get('sequence') != self.sequence:
                raise ValueError('worker response identity mismatch')
            if response.get('status') != 'complete':
                raise RuntimeError(f"worker callback failed: {response.get('error', 'unknown')}")
            days = response['days']
            began, ended = response['started'], response['ended']
            if (type(days) is not int or not 0 <= days <= request['maximum_days'] or
                    not all(math.isfinite(v) for v in (began, ended)) or
                    not request['dispatched'] <= began <= ended <= received <= budget.deadline):
                raise ValueError('invalid worker execution receipt')
            budget.days += days
            budget.worker_receipt = {k: response[k] for k in
                                     ('sequence','request_sha256','days','started','ended')}
            budget.worker_receipt.update(request=request, received=received)
            budget.check()
            return response['payload']
        except BaseException as exc:
            self.failed = True
            self.failure = dict(sequence=self.sequence,error=f'{type(exc).__name__}: {exc}',
                                traceback=traceback.format_exc(),status='failed_not_reusable')
            self.close()
            raise

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        if self.temporary is not None:
            if self.failed:
                # Retain requests, responses and partial atomic writes after the
                # process tree has stopped. These are diagnostics, never commits.
                self.failure_path = self.temporary
                atomic_json(self.temporary/'parent_failure.json', self.failure)
            else:
                shutil.rmtree(self.temporary)
            self.temporary = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
