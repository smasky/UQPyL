#Non-dominated Sorting
import numpy as np

def NDSort(YPop, NSort):
        '''
            Non-dominated Sorting
        '''
        
        PopObj, indices = np.unique(YPop, axis=0, return_inverse=True)
       
        Table = np.bincount(indices)
        N, M = PopObj.shape
        FrontNo = np.inf * np.ones(N)
        MaxFNo = 0

        while np.sum(Table[FrontNo < np.inf]) < min(NSort, len(indices)):
            MaxFNo += 1
            for i in range(N):
                if FrontNo[i] == np.inf:
                    Dominated = False
                    for j in range(i-1, -1, -1):
                        if FrontNo[j] == MaxFNo:
                            m = 1
                            while m < M and PopObj[i, m] >= PopObj[j, m]:
                                m += 1
                            Dominated = m == M
                            if Dominated or M == 2:
                                break
                    if not Dominated:
                        FrontNo[i] = MaxFNo

        FrontNo = FrontNo[indices]

        return FrontNo, MaxFNo