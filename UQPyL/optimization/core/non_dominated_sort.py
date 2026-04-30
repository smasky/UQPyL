import numpy as np

from ..core.constraint import calcConstraintViolation


def NDSort(popObjs, popCons=None, nSort=None):
    popObjs = np.atleast_2d(np.asarray(popObjs, dtype=float))
    N, M = popObjs.shape

    if nSort is None:
        nSort = N

    frontNo = np.inf * np.ones(N)
    maxFrontNo = 0

    if popCons is not None:
        cv = calcConstraintViolation(popCons)
        feasible = cv <= 0
    else:
        cv = None
        feasible = np.ones(N, dtype=bool)

    feasibleIdx = np.where(feasible)[0]
    infeasibleIdx = np.where(~feasible)[0]

    if feasibleIdx.size > 0:
        feasibleObjs = popObjs[feasibleIdx]
        uniqueObjs, indices = np.unique(feasibleObjs, axis=0, return_inverse=True)
        nUnique = len(uniqueObjs)
        feasibleFrontNoUnique = np.inf * np.ones(nUnique)

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        sn = 0
        while sn < min(nSort, feasibleIdx.size):
            maxFrontNo += 1
            for i in range(nUnique):
                if feasibleFrontNoUnique[i] == np.inf:
                    dominated = False
                    for j in range(nUnique):
                        if feasibleFrontNoUnique[j] == maxFrontNo and dominates(
                            uniqueObjs[j], uniqueObjs[i]
                        ):
                            dominated = True
                            break
                    if not dominated:
                        feasibleFrontNoUnique[i] = maxFrontNo
                        sn += np.sum(indices == i)

        frontNo[feasibleIdx] = feasibleFrontNoUnique[indices]

    if infeasibleIdx.size > 0:
        order = np.argsort(cv[infeasibleIdx])
        ranks = np.empty(infeasibleIdx.size, dtype=float)
        ranks[order] = np.arange(1, infeasibleIdx.size + 1)
        frontNo[infeasibleIdx] = maxFrontNo + ranks

    if np.any(np.isfinite(frontNo)):
        maxFrontNo = int(np.max(frontNo[np.isfinite(frontNo)]))
    else:
        maxFrontNo = 0
    return frontNo, maxFrontNo
