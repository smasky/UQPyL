import numpy as np
from scipy.spatial.distance import cdist

def IGD(popObjs, optimum):
    popObjs = np.atleast_2d(np.asarray(popObjs, dtype=float))
    optimum = np.atleast_2d(np.asarray(optimum, dtype=float))

    if popObjs.size == 0:
        raise ValueError("`popObjs` must contain at least one point.")
    if optimum.size == 0:
        raise ValueError("`optimum` must contain at least one point.")

    distances = cdist(optimum, popObjs, metric='euclidean')
    minDist = np.min(distances, axis=1)
    igd = np.mean(minDist)
    return igd
