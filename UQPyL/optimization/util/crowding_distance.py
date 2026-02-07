import numpy as np

def crowdingDist(popObjs, frontNo = None):
    
    N, M = popObjs.shape
    
    if frontNo is None:
        frontNo = np.ones(N, dtype=int)

    crowdDis = np.zeros(N, dtype=float)

    finite_mask = ~np.isinf(frontNo)
    
    if not np.any(finite_mask):
        return crowdDis
    fronts = np.unique(frontNo[finite_mask])

    for f in fronts:
        front_idx = np.flatnonzero(frontNo == f)
        n_f = front_idx.size
        if n_f == 0:
            continue
        if n_f == 1:
            crowdDis[front_idx[0]] = np.inf
            continue
        if n_f == 2:
            crowdDis[front_idx] = np.inf
            continue
        
        F = popObjs[front_idx, :]                 # (n_f, M)
        fmax = F.max(axis=0)
        fmin = F.min(axis=0)
        denom = fmax - fmin
        
        valid_dim = denom > 0
        
        if not np.any(valid_dim):
            
            crowdDis[front_idx[0]]   = np.inf
            crowdDis[front_idx[-1]]  = np.inf
            continue

        Fn = np.empty_like(F, dtype=float)
        Fn[:, valid_dim] = (F[:, valid_dim] - fmin[valid_dim]) / denom[valid_dim]
        
        if np.any(~valid_dim):
            Fn[:, ~valid_dim] = 0.0

        for i in range(M):
            if not valid_dim[i]:
                continue
            order = np.argsort(Fn[:, i])
            vals  = Fn[order, i]

            crowdDis[front_idx[order[0]]]  = np.inf
            crowdDis[front_idx[order[-1]]] = np.inf

            contrib = vals[2:] - vals[:-2]
            mid_idx = order[1:-1]
            
            crowdDis[front_idx[mid_idx]] += contrib

    return crowdDis
