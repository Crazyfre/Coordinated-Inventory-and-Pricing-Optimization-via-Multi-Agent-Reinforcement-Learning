"""Retry transient access denial only; never rerun a callback or extend a deadline."""
from pathlib import Path
import time


def read_bytes(path,deadline,*,clock=time.perf_counter,sleep=time.sleep):
    while True:
        if clock()>=deadline:raise TimeoutError('transport read deadline exceeded')
        try:
            raw=Path(path).read_bytes()
        except PermissionError as exc:
            remaining=deadline-clock()
            if remaining<=0:raise TimeoutError('transport access denial exceeded deadline') from exc
            sleep(min(.01,remaining))
            continue
        if clock()>=deadline:raise TimeoutError('transport read deadline exceeded')
        return raw
