import numpy as np


def NDSort(popObjs, popCons=None, nSort=None):

    popObjs = np.asarray(popObjs, dtype=float)
    N, M = popObjs.shape

    if nSort is None:
        nSort = N


    objs = popObjs.copy()
    if popCons is not None:
        popCons_ = np.asarray(popCons, dtype=float) * 10
        infeasible = (popCons_ > 0).any(axis=1)
        if np.any(infeasible):
            violation = np.max(popCons_[infeasible], axis=1, keepdims=True)
            maxObj = np.max(objs, axis=0, keepdims=True)
            objs[infeasible] = maxObj + violation


    uniqueObjs, inv = np.unique(objs, axis=0, return_inverse=True)
    N_unique = uniqueObjs.shape[0]

    if N_unique == 1:
      
        frontNo = np.ones(N, dtype=int)
        return frontNo, 1


    S = [[] for _ in range(N_unique)]      
    n = np.zeros(N_unique, dtype=int)       
    fronts = [[]]                           

    for p in range(N_unique):
        p_obj = uniqueObjs[p]

    
        p_le_q = np.all(p_obj <= uniqueObjs, axis=1)
        p_lt_q = np.any(p_obj < uniqueObjs, axis=1)


        q_le_p = np.all(uniqueObjs <= p_obj, axis=1)
        q_lt_p = np.any(uniqueObjs < p_obj, axis=1)

  
        p_le_q[p] = False
        q_le_p[p] = False

        p_dom_q = p_le_q & p_lt_q
        q_dom_p = q_le_p & q_lt_p

        S[p] = np.where(p_dom_q)[0].tolist()
        n[p] = int(np.count_nonzero(q_dom_p))

        if n[p] == 0:
            fronts[0].append(p)

    frontNo_unique = np.zeros(N_unique, dtype=int)
    assigned = 0
    i = 0

    while i < len(fronts) and fronts[i] and assigned < nSort:
        next_front = []
        for p in fronts[i]:
            if frontNo_unique[p] != 0:
                continue
            frontNo_unique[p] = i + 1
            assigned += 1
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1

    maxFrontNo = int(np.max(frontNo_unique))

    frontNo_unique[frontNo_unique == 0] = maxFrontNo + 1


    frontNo = frontNo_unique[inv]

    return frontNo, maxFrontNo