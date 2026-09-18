"""Shared restart policy for internal surrogate optimizers."""

import numpy as np

from ..core import spawn_seed


def resolveRestarts(optimizer, count):
    if count is None:
        family = getattr(optimizer, 'alg_type', getattr(optimizer, 'type', None))
        return 4 if family == 'MP' else 1
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError('nRestartTimes must be a non-negative integer.')
    return int(count)


def runLocalRestarts(model, problem, paraInfos):
    lower, upper = problem.lb.ravel(), problem.ub.ravel()
    initial = np.empty(lower.size)
    for name, indices in paraInfos.items():
        value = model.setting.get(name)
        initial[indices] = np.log(value) if model.setting.parLog[name] else value
    initial = np.clip(initial, lower, upper)
    bestPoint, bestValue = None, np.inf
    for index in range(model.nRes + 1):
        seed = spawn_seed(model.rng)
        start = initial if index == 0 else np.random.default_rng(seed).uniform(lower, upper)
        point, _ = model.optimizer.run(problem, xInit=start.copy(), seed=seed)
        point = np.asarray(point, dtype=float).ravel()
        if point.size != lower.size or not np.all(np.isfinite(point)):
            continue
        if np.any(point < lower) or np.any(point > upper):
            continue
        # Compare the objective at the returned point, not a stale optimizer score.
        value = float(np.asarray(problem.objFunc(point)).item())
        if np.isfinite(value) and value < bestValue:
            bestPoint, bestValue = point.copy(), value
    if bestPoint is None:
        raise RuntimeError('No restart returned a finite, in-bounds hyperparameter solution.')
    return bestPoint, bestValue
