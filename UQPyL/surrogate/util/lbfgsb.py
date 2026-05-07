import numpy as np
from scipy.optimize import minimize


class LBFGSB():
    """
    Internal MP optimizer backed by scipy.optimize.minimize(method='L-BFGS-B').
    """

    type = "MP"
    name = "LBFGSB"

    def __init__(self, options=None) -> None:
        self.options = {} if options is None else dict(options)

    def run(self, problem, xInit=None, seed=None):
        lb = np.asarray(problem.lb, dtype=float).ravel()
        ub = np.asarray(problem.ub, dtype=float).ravel()

        if xInit is None:
            if seed is not None:
                np.random.seed(seed)
            xInit = np.random.uniform(lb, ub, problem.nInput)
        else:
            xInit = np.asarray(xInit, dtype=float).ravel()

        xInit = np.clip(xInit, lb, ub)
        bounds = list(zip(lb, ub))

        def obj_func(x):
            value = problem.objFunc(np.asarray(x, dtype=float).ravel())
            return float(np.asarray(value).reshape(-1)[0])

        res = minimize(
            obj_func,
            xInit,
            method="L-BFGS-B",
            bounds=bounds,
            options=self.options,
        )

        bestDec = np.asarray(res.x, dtype=float).ravel()
        bestObj = float(np.asarray(res.fun).reshape(-1)[0])

        return bestDec, bestObj
