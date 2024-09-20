#Non-dominated Sorting
import numpy as np

def NDSort(pop, nSort=None):
        '''
            Non-dominated Sorting
        '''
        if nSort is None:
            nSort=len(pop)
        popObjs, indices = np.unique(pop.objs, axis=0, return_inverse=True)
       
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

