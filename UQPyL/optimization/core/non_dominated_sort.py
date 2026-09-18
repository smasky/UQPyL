import numpy as np

from ..core.constraint import calcConstraintViolation


def NDSort(popObjs, popCons=None, nSort=None, conWgt=None):
    popObjs = np.atleast_2d(np.asarray(popObjs, dtype=float))
    N, M = popObjs.shape

    if nSort is None:
        nSort = N

    frontNo = np.inf * np.ones(N)
    maxFrontNo = 0

    if popCons is not None:
        cv = calcConstraintViolation(popCons, conWgt)
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
        # Equal violation means equal constraint rank; diversity breaks ties.
        _, ranks = np.unique(cv[infeasibleIdx], return_inverse=True)
        frontNo[infeasibleIdx] = maxFrontNo + ranks + 1

    if np.any(np.isfinite(frontNo)):
        # Return the last front needed for nSort, not every infeasible rank.
        rankIndex = min(max(int(nSort), 1), N) - 1
        maxFrontNo = int(np.sort(frontNo)[rankIndex])
    else:
        maxFrontNo = 0
    return frontNo, maxFrontNo
