import numpy as np


def singleFunc(func):
    """
    Decorator to ensure a function works with 2D input data.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function.
    """

    def wrapper(X):
        X = np.atleast_2d(X)
        evals = []

        for x in X:
            eval = func(x)
            # Model adapters may return the same work buffer on every call.
            evals.append(np.atleast_1d(eval).copy())

        return np.vstack(evals)

    return wrapper
