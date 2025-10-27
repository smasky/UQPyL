#Non-dominated Sorting
import numpy as np

def NDSort(YPop, NSort):
        '''
            Non-dominated Sorting
        '''
        
        PopObj, indices = np.unique(YPop, axis=0, return_inverse=True)
       
        Table = np.bincount(indices)
        N, M = PopObj.shape
        frontNo = np.inf * np.ones(N)
        maxFNo = 0

        while np.sum(Table[frontNo < np.inf]) < min(NSort, len(indices)):
            maxFNo += 1
            for i in range(N):
                if frontNo[i] == np.inf:
                    Dominated = False
                    for j in range(i-1, -1, -1):
                        if frontNo[j] == maxFNo:
                            m = 1
                            while m < M and PopObj[i, m] >= PopObj[j, m]:
                                m += 1
                            Dominated = m == M
                            if Dominated or M == 2:
                                break
                    if not Dominated:
                        frontNo[i] = maxFNo

        frontNo = frontNo[indices]

        return frontNo, maxFNo