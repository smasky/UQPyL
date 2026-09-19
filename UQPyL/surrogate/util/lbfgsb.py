import numpy as np
from scipy.optimize import minimize


class LBFGSB():
    """
    Internal MP optimizer backed by scipy.optimize.minimize(method='L-BFGS-B').

    Returns the best finite evaluated point, including when SciPy stops early.
    ``lastResult`` retains SciPy's raw result and convergence status; its x/fun
    may differ from this wrapper's returned pair. Objectives must be deterministic.
    """

    type = "MP"
    name = "LBFGSB"

    def __init__(self, options=None) -> None:
        self.options = {} if options is None else dict(options)
        self.lastResult = None

    def run(self, problem, xInit=None, seed=None):
        self.lastResult = None
        lb = np.asarray(problem.lb, dtype=float).ravel()
        ub = np.asarray(problem.ub, dtype=float).ravel()
        if lb.size != problem.nInput or ub.size != problem.nInput or not lb.size:
            raise ValueError('LBFGSB bounds must match problem.nInput.')
        if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)) or np.any(lb > ub):
            raise ValueError('LBFGSB requires finite, ordered bounds.')

        if xInit is None:
            rng = np.random.default_rng(seed)
            xInit = rng.uniform(lb, ub, problem.nInput)
        else:
            xInit = np.asarray(xInit, dtype=float).ravel()
            if xInit.size != lb.size or not np.all(np.isfinite(xInit)):
                raise ValueError('xInit must contain one finite value per input.')

        xInit = np.clip(xInit, lb, ub)
        bounds = list(zip(lb, ub))

        bestDec, bestObj = None, np.inf

        def obj_func(x):
            nonlocal bestDec, bestObj
            point = np.asarray(x, dtype=float).ravel().copy()
            value = float(np.asarray(problem.objFunc(point.copy())).item())
            if np.isfinite(value) and np.all(point >= lb) and np.all(point <= ub) and value < bestObj:
                bestDec, bestObj = point, value
            return value

        res = minimize(
            obj_func,
            xInit,
            method="L-BFGS-B",
            bounds=bounds,
            options=self.options,
        )

        self.lastResult = res
        # An abnormal line-search exit can leave res.fun inconsistent with res.x.
        point = np.asarray(res.x, dtype=float).ravel()
        if point.size == lb.size and np.all(np.isfinite(point)) and np.all(point >= lb) and np.all(point <= ub):
            obj_func(point)
        if bestDec is None:
            raise RuntimeError('LBFGSB found no finite, in-bounds objective value.')

        return bestDec.copy(), bestObj
