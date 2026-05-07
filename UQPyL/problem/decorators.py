import numpy as np

from .eval import Eval


def singleFunc(func):
    """
    Decorator to ensure a function works with 2D input data.

    :param func: Function to wrap.
    :return: Wrapped function.
    """

    def wrapper(X):
        X = np.atleast_2d(X)
        evals = []

        for x in X:
            eval = func(x)
            evals.append(np.atleast_1d(eval))

        return np.vstack(evals)

    return wrapper


def singleEval(func):
    """
    Decorator to ensure an evaluation function works with 2D input data.

    :param func: Evaluation function to wrap.
    :return: Wrapped function.
    """

    def wrapper(X):
        X = np.atleast_2d(X)

        objs = []
        cons = []

        for x in X:
            eval = func(x)
            if not isinstance(eval, Eval):
                raise TypeError("singleEval-wrapped function must return Eval.")
            objs.append(np.atleast_1d(eval.objs))
            if eval.cons is not None:
                cons.append(np.atleast_1d(eval.cons))

        cons_arr = np.vstack(cons) if len(cons) != 0 else None
        return Eval(objs=np.vstack(objs), cons=cons_arr)

    return wrapper
