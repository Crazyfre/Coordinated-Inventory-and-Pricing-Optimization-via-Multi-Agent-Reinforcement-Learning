"""Diagnostic only: remove learned baseline from otherwise unchanged GAE.

Lambda .95 remains unchanged: this is NOT an unbiased Monte-Carlo estimator.
Critic still fits but its predictions cannot affect collected advantages.
"""
from src.algos.forecast_prior import ForecastPriorMAPPO


class CriticFreeForecastPriorMAPPO(ForecastPriorMAPPO):
    def value(self, state):
        return 0.
