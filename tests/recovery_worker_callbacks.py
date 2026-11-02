"""Engineering fixtures only; never scientific data generation."""
import os
import time


def deterministic(arguments, budget, context):
    for _ in range(arguments['days']):
        budget.before_step()
    return dict(values=list(range(arguments['days'])))


def fail(arguments, budget, context):
    raise ValueError('fixture callback error')


def hang(arguments, budget, context):
    time.sleep(60)


def hard_exit(arguments, budget, context):
    os._exit(23)


def invalid_json(arguments, budget, context):
    return {'invalid': float('nan')}
