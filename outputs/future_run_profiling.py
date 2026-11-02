"""Opt-in future-process phase timing and device update; no current-run hooks."""
from contextlib import contextmanager
import json
from pathlib import Path
import time
import torch


class PhaseTimings:
    """Inclusive/exclusive wall time; observation only, never consumes policy RNG."""
    def __init__(self):
        self.stack = []
        self.rows = {}

    @contextmanager
    def phase(self, name):
        if not isinstance(name, str) or not name: raise ValueError('named phase required')
        frame = [time.perf_counter(), 0.]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - frame[0]
            assert self.stack.pop() is frame
            row = self.rows.setdefault(name, dict(calls=0, inclusive_seconds=0., exclusive_seconds=0.))
            row['calls'] += 1
            row['inclusive_seconds'] += elapsed
            row['exclusive_seconds'] += max(0., elapsed-frame[1])
            if self.stack: self.stack[-1][1] += elapsed

    def wrapped(self, name, function):
        def invoke(*args, **kwargs):
            with self.phase(name): return function(*args, **kwargs)
        return invoke

    def snapshot(self):
        if self.stack: raise RuntimeError('cannot summarize an open timing phase')
        return {name: dict(row) for name, row in self.rows.items()}


def optimizer_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                # Original Adam is noncapturable; its scalar step belongs on CPU.
                state[key] = value.to('cpu' if key=='step' else device)


def move_training_state(model, device):
    model.to(device)
    for optimizer in list(model.actor_optimizers.values()) + [model.critic_optimizer]:
        optimizer_device(optimizer, device)


def device_update(model, episodes, device):
    """Unchanged original update; CPU is the reference-preserving default path."""
    if device not in ('cpu', 'cuda'): raise ValueError('fixed CPU or CUDA path required')
    if device=='cpu':
        if any(p.device.type!='cpu' for p in model.parameters()): raise ValueError('CPU model required')
        return model.update(episodes)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone()
    move_training_state(model, 'cuda')
    try:
        with torch.device('cuda'):
            result = model.update(episodes)
        torch.cuda.synchronize()
    finally:
        move_training_state(model, 'cpu')
        torch.cuda.synchronize()
    if not torch.equal(before,torch.get_rng_state()) or not torch.equal(cuda_before,torch.cuda.get_rng_state()):
        raise RuntimeError('network update unexpectedly consumed policy RNG')
    return result


def verified_update_device(report_path):
    """Prospective callers must bind this report and all sources before launch."""
    from src.experiments.execution_journal import digest
    path=Path(report_path);report=json.loads(path.read_text())
    if (report.get('status')!='COMPLETE_CLOSED_FIXTURE_PROFILE_ONLY'
            or len(report.get('workloads',[]))!=8 or report.get('calibrated_fits')!=0
            or not all(digest(p)==h for p,h in report['sources'].items())):
        raise RuntimeError('complete source-bound device-profile receipt required')
    selected=report['selected_device']
    if selected=='cpu':return selected
    if (selected!='cuda' or not report['cuda_bitwise_equivalent']
            or report['cuda_median_sum_seconds']>=report['cpu_median_sum_seconds']):
        raise RuntimeError('CUDA requires complete equivalence AND measured speed advantage')
    return selected
