import numpy as np


def sqliteSimFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2))
    sim[:, 0] = X[:, 0]
    sim[:, 1] = X[:, 1]
    return sim
