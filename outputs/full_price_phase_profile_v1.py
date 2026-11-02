"""Observation-only timings on new workers; no environment or RNG changes."""
from contextlib import contextmanager
from outputs.future_run_profiling import PhaseTimings
from src.env.joint_env import JointRetailEnv

TIMINGS = PhaseTimings()


@contextmanager
def simulation_timing():
    original = JointRetailEnv.step
    JointRetailEnv.step = TIMINGS.wrapped('simulation', original)
    try:
        yield
    finally:
        JointRetailEnv.step = original
