"""Independent dense Delta reference for data without cutoff ties.

Do not use for tied cutoffs: their fractional weighting has its own reference.
"""

import numpy as np


def pairwiseDelta(X, Y, neighbors=2):
    distances = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1)[:, :neighbors]
    return 0.5 * np.mean((Y[:, None, :] - Y[nearest]) ** 2)
