"""Fixed small regression tree for archive diagnosis, not a deployed policy."""
import numpy as np


def fit_tree(x, y, depth=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or not len(y):
        raise ValueError('nonempty matching matrix and targets required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite data required')
    node = dict(count=len(y), mean=float(y.mean()), depth=depth)
    if depth >= 4 or len(y) < 16:
        return node
    best_loss = float(np.sum((y-y.mean())**2))
    best = None
    for col in range(x.shape[1]):
        values = np.unique(x[:, col])
        for lo, hi in zip(values[:-1], values[1:]):
            cut = float(lo+(hi-lo)/2)
            left = x[:, col] <= cut
            if left.sum() < 8 or (~left).sum() < 8:
                continue
            loss = float(np.sum((y[left]-y[left].mean())**2)+
                         np.sum((y[~left]-y[~left].mean())**2))
            if loss < best_loss:
                best_loss, best = loss, (col, cut, left)
    if best is not None:
        col, cut, left = best
        node.update(feature=col, threshold=cut,
                    left=fit_tree(x[left], y[left], depth+1),
                    right=fit_tree(x[~left], y[~left], depth+1))
    return node


def predict_tree(tree, x):
    result = []
    for row in np.asarray(x):
        node = tree
        while 'feature' in node:
            node = node['left' if row[node['feature']] <= node['threshold'] else 'right']
        result.append(node['mean'])
    return np.array(result)
