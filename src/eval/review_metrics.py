"""Review metrics with explicit pairing and no invented missing measurements."""
import math
import numpy as np
from scipy.stats import t


def _vector(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('expected a nonempty finite vector')
    return x


def return_summary(values):
    x = _vector(values)
    sd = float(x.std(ddof=1)) if len(x) > 1 else None
    half = float(t.ppf(.975, len(x)-1)*sd/math.sqrt(len(x))) if sd is not None else None
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                std=sd, p05=float(np.quantile(x, .05)),
                ci95=None if half is None else [float(x.mean()-half), float(x.mean()+half)])


def success_interval(successes):
    """Wilson 95% interval over independent evaluation episodes, not seed clusters."""
    x = _vector(successes)
    if not np.isin(x, [0, 1]).all():
        raise ValueError('successes must be binary')
    n, p, z = len(x), float(x.mean()), 1.959963984540054
    den = 1 + z*z/n
    center = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(n=n, successes=int(x.sum()), rate=p,
                ci95=[max(0., center-half), min(1., center+half)])


def paired_shortfall(policy, reference, metric='return'):
    """Signed reference-minus-policy differences, joined by episode seed.

    Reference is a declared feasible controller, never called an optimal oracle.
    Cumulative values follow sorted seed order and are not online-learning regret.
    """
    def index(rows):
        result = {}
        for row in rows:
            if row['seed'] in result:
                raise ValueError('duplicate seed: separate training cohorts first')
            result[row['seed']] = row[metric]
        return result
    a, b = index(policy), index(reference)
    if not a or a.keys() != b.keys():
        raise ValueError('complete matching episode seeds required')
    seeds = sorted(a)
    differences = _vector([b[s]-a[s] for s in seeds])
    return dict(reference_shortfall=return_summary(differences), seeds=seeds,
                differences=differences.tolist(), cumulative=np.cumsum(differences).tolist())


def convergence(evaluations, threshold, consecutive=3):
    """Threshold attainment on chronological *evaluation* returns only.

    Return both first step in the qualifying run and confirmation step. Missing
    evaluation histories are unavailable, rather than fabricated nonconvergence.
    """
    if not math.isfinite(threshold) or consecutive < 1 or int(consecutive) != consecutive:
        raise ValueError('finite threshold and positive integer run length required')
    if not evaluations:
        return dict(status='unavailable', onset_step=None, confirmed_step=None)
    steps = _vector([r['steps'] for r in evaluations])
    scores = _vector([r['mean_evaluation_return'] for r in evaluations])
    if np.any(steps < 0) or np.any(np.diff(steps) <= 0):
        raise ValueError('evaluation steps must strictly increase')
    run = 0
    for i, score in enumerate(scores):
        run = run+1 if score >= threshold else 0
        if run >= consecutive:
            return dict(status='reached', onset_step=float(steps[i-consecutive+1]),
                        confirmed_step=float(steps[i]))
    return dict(status='not_reached', onset_step=None, confirmed_step=None)


def generalization_gap(seen, unseen):
    """Independent episode uncertainty; caller must match checkpoint and horizon."""
    a, b = _vector(seen), _vector(unseen)
    gap = float(a.mean()-b.mean())
    if min(len(a), len(b)) < 2:
        return dict(gap=gap, ci95=None)
    va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
    se = math.sqrt(va+vb)
    dof = (va+vb)**2/(va*va/(len(a)-1)+vb*vb/(len(b)-1)) if se else 1
    half = float(t.ppf(.975, dof)*se)
    return dict(gap=gap, ci95=[gap-half, gap+half])
