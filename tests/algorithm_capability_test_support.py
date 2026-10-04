"""Problem factory shared by algorithm capability and persistence tests."""

import numpy as np
from UQPyL.problem import Problem


def makeProblem(nObj=1, calls=None, constant=False):
    def objective(x):
        if calls is not None:
            calls.append(x.copy())
        return np.ones((len(x), nObj)) if constant else np.column_stack([(x[:, 0] - i) ** 2 for i in range(nObj)])

    return Problem(nInput=1, nObj=nObj, lb=0.0, ub=1.0, objFunc=objective)
