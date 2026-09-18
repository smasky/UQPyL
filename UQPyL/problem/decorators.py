import numpy as np


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
