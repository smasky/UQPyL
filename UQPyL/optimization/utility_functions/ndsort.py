#Non-dominated Sorting
import numpy as np

def NDSort(pop, nSort = None):
        '''
            Non-dominated Sorting
        '''
        if nSort is None:
            nSort = len(pop)
        
        popObjs = np.copy(pop.objs)
        
        _, M = popObjs.shape
        
        if pop.cons is not None:
            popCons_ = np.copy(pop.cons)
            popCons = popCons_ * pop.conWgt if pop.conWgt is not None else popCons_ * 10
            Infeasible = np.any(popCons > 0, axis=1)
            popObjs[Infeasible, :] = (
                np.tile(np.max(popObjs, axis=0), (np.sum(Infeasible), 1)) +
                np.tile(np.sum(np.maximum(0, popCons[Infeasible, :]), axis=1).reshape(-1, 1), (1, M))
            )
        
        popObjs, indices = np.unique(popObjs, axis=0, return_inverse=True)
       
        table = np.bincount(indices)
        n, d = popObjs.shape
        frontNo = np.inf * np.ones(n)
        maxFrontNo = 0

        while np.sum(table[frontNo < np.inf]) < min(nSort, len(indices)):
            maxFrontNo += 1
            for i in range(n):
                if frontNo[i] == np.inf:
                    Dominated = False
                    for j in range(i-1, -1, -1):
                        if frontNo[j] == maxFrontNo:
                            m = 1
                            while m < d and popObjs[i, m] >= popObjs[j, m]:
                                m += 1
                            Dominated = m == d
                            if Dominated or d == 2:
                                break
                    if not Dominated:
                        frontNo[i] = maxFrontNo

        frontNo = frontNo[indices]

        return frontNo, maxFrontNo

