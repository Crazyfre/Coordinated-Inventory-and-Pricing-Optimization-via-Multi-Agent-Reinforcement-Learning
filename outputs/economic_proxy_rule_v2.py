"""Fixed local weekly heuristic, not simulator economics or a fitted actor."""
import numpy as np
from scipy.stats import nbinom


def expected_sales(stock, mean, size):
    stock, mean, size = np.broadcast_arrays(np.asarray(stock, float), np.asarray(mean, float), np.asarray(size, float))
    if not np.isfinite([stock, mean, size]).all() or np.any(stock < 0) or np.any(mean < 0) or np.any(size <= 0):
        raise ValueError('finite nonnegative stock/mean and positive NB size required')
    probability = size / (size + mean)
    integer = np.floor(stock)
    return mean * nbinom.cdf(integer - 1, size + 1, probability) + stock * nbinom.sf(integer, size, probability)


def scores(observation, day, base_prices, phi):
    x, base, phi = np.asarray(observation, float), np.asarray(base_prices, float), np.asarray(phi, float)
    if x.ndim != 1 or len(x) < 180 or base.shape != (10,) or phi.shape != (10,) or not 1 <= day <= 28:
        raise ValueError('declared local pricing observation/day/static parameters required')
    if not np.isfinite(x).all() or not np.isfinite(base).all() or np.any(base <= 0) or not np.isfinite(phi).all() or np.any(phi <= 0):
        raise ValueError('finite local/static parameters required')
    stock, forecast, previous = x[:10], x[-20:-10], x[150:160]
    if np.any(stock < 0) or np.any(forecast < 0) or np.any(previous <= 0):
        raise ValueError('nonnegative stock/forecast and positive previous prices required')
    remaining = 29 - day
    horizon = min(7, remaining)
    prices = base[:, None] * np.array([.8, .9, 1., 1.1, 1.2])
    mean = horizon * forecast[:, None] * (prices / previous[:, None]) ** -1.5
    sold = expected_sales(stock[:, None], mean, horizon * phi[:, None])
    exposure = .6 * base * max(0., 1. - 4.5 / remaining)
    return ((prices - exposure[:, None]) * sold - (mean - sold)
            - .02 * horizon * (stock[:, None] - sold / 2)
            - .05 * np.abs(prices - previous[:, None]))


def action(observation, day, base_prices, phi):
    values = scores(observation, day, base_prices, phi)
    return (4 - values[:, ::-1].argmax(1)).astype(np.int64)
