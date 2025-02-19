import numpy as np


def crowdingDistance(pop, frontNo = None):
    
    popObjs = pop.objs
    
    n, m = popObjs.shape

    if frontNo is None:
        frontNo = np.ones(n)
    
    crowdDis = np.zeros(n)
    
    fronts = np.setdiff1d(np.unique(frontNo), np.inf)
    
    for f in fronts:
        front = np.where(frontNo == f)[0]
        fmax = np.max(popObjs[front, :], axis=0)
        fmin = np.min(popObjs[front, :], axis=0)
        
        for i in range(m):
            
            rank = np.argsort(popObjs[front, i])
            crowdDis[front[rank[0]]] = np.inf
            crowdDis[front[rank[-1]]] = np.inf
            
            for j in range(1, len(front) - 1):
                crowdDis[front[rank[j]]] += (popObjs[front[rank[j+1]], i] - popObjs[front[rank[j-1]], i]) / (fmax[i] - fmin[i])
                    
    return crowdDis