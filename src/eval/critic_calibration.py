"""Training-only normalized value fitting and scenario-level validation metrics."""
import copy
import numpy as np
import torch


def episode_data(data, kind):
    starts = list(range(0, 28, 7)) if kind == 'pricing' else list(range(28))
    if len(data['rewards']) != 28 or len(data['decisions']) != len(starts):
        raise ValueError('complete episode with correct cadence required')
    rewards = np.asarray(data['rewards'], float) / 100
    return dict(states=[d['state'] for d in data['decisions']],
        targets=[float(sum(.99 ** k * r for k, r in enumerate(rewards[start:]))) for start in starts])


def arrays(data):
    x = np.asarray([s for e in data for s in e['states']], np.float32)
    y = np.asarray([t for e in data for t in e['targets']], float)
    if len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite matched states and targets required')
    return x, y


def fit(critic, data, epochs=200):
    x, y = arrays(data)
    mean, std = float(y.mean()), max(float(y.std()), 1e-6)
    t = x[:, -1].astype(float)
    coefficients = np.linalg.lstsq(np.stack([np.ones(len(t)), t, t*t], -1), y, rcond=None)[0]
    state = torch.tensor(x); target = torch.tensor((y - mean) / std, dtype=torch.float32)
    optimizer = torch.optim.Adam(critic.parameters(), lr=.003)
    history = []
    for epoch in range(epochs):
        loss = .5 * (critic(state).squeeze(-1) - target).square().mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.)
        optimizer.step()
        history.append(copy.deepcopy(dict(epoch=epoch + 1, critic=critic.state_dict(),
            optimizer=optimizer.state_dict(), loss=float(loss.detach()))))
    return dict(mean=mean, std=std, time_coefficients=coefficients.tolist()), history


def predictions(critic, original, calibration, data):
    x, y = arrays(data); t = x[:, -1].astype(float)
    with torch.no_grad():
        learned = critic(torch.tensor(x)).squeeze(-1).numpy() * calibration['std'] + calibration['mean']
        initial = original(torch.tensor(x)).squeeze(-1).numpy()
    design = np.stack([np.ones(len(t)), t, t*t], -1)
    return dict(learned=learned, original=initial, constant=np.full(len(y), calibration['mean']),
                time=design @ np.asarray(calibration['time_coefficients']))


def metrics(data, predicted):
    _, y = arrays(data); count = len(data); length = len(data[0]['targets'])
    draws = np.random.default_rng(20261027).integers(count, size=(10000, count))
    summary = {}; mse_by_episode = {}
    for name, p in predicted.items():
        error = np.asarray(p) - y
        mse_by_episode[name] = error.reshape(count, length).__pow__(2).mean(1)
        summary[name] = dict(mse=float(np.mean(error**2)), rmse=float(np.sqrt(np.mean(error**2))),
            bias=float(error.mean()), explained_variance=float(1 - np.var(error) / np.var(y)))
    differences = {}
    for ref in ['constant', 'time', 'original']:
        d = mse_by_episode['learned'] - mse_by_episode[ref]
        differences[ref] = dict(mean=float(d.mean()), ci95=np.quantile(d[draws].mean(1), [.025, .975]).tolist())
    passed = summary['learned']['explained_variance'] >= .2 and all(
        summary['learned']['mse'] <= .9 * summary[ref]['mse'] for ref in ['constant', 'time'])
    return dict(controllers=summary, mse_differences=differences, passed=bool(passed))
