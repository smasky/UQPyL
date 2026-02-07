#Non-dominated Sorting
import numpy as np

def NDSort(popObjs, popCons = None, nSort = None):
    
    N, M = popObjs.shape
    
    if nSort is None:
        nSort = N

    if popCons is not None:
        popCons_ =  popCons * 10
        infeasible = (popCons_ > 0).any(axis=1)
        violation = np.max(popCons_[infeasible], axis=1, keepdims=True)
        # violation = np.maximum(0, popCons_[infeasible]*conWgt) if conWgt is not None else np.maximum(0, popCons_[infeasible]).sum(axis=1)
        maxObj = np.max(popObjs, axis=0, keepdims=True)
        popObjs[infeasible] = maxObj + violation
        
    uniqueObjs, indices = np.unique(popObjs, axis=0, return_inverse=True)
    N_unique = len(uniqueObjs)
    frontNo_unique = np.inf * np.ones(N_unique)
    maxFrontNo = 0

    def dominates(a, b):
        return np.all(a <= b) and np.any(a < b)

    sn = 0
    while sn < min(nSort, N_unique):
        maxFrontNo += 1
        for i in range(N_unique):
            if frontNo_unique[i] == np.inf:
                dominated = False
                for j in range(N_unique):
                    if frontNo_unique[j] == maxFrontNo and dominates(uniqueObjs[j], uniqueObjs[i]):
                        dominated = True
                        break
                if not dominated:
                    frontNo_unique[i] = maxFrontNo
                    sn += 1

    frontNo = frontNo_unique[indices]

    return frontNo, maxFrontNo