import numpy as np

def HV(pop, refPoint=None, normalize=True):
    """计算Pareto前沿的超体积指标。
    
    参数:
        pop: Population对象, 包含目标函数值
        refPoint: numpy数组, 参考点。如果为None, 将自动生成。
                  参考点应在所有目标维度上比任何解要差。
        
    返回:
        float: 计算得到的超体积值
    """
    popObjs = pop.getBest().objs
    _, m = popObjs.shape
    
    if popObjs.size == 0:
        return 0.0
    
    if refPoint is None:
        
        refPoint = np.max(popObjs, axis=0) * 1.1
        
    if normalize:
        
        fmin = np.min(np.vstack((popObjs, np.zeros((1, m)))), axis=0)
        fmax = np.max(np.vstack((popObjs, np.ones((1, m)))), axis=0)
        
        popObjs = (popObjs - fmin) / (fmax - fmin)
        
        refPoint = (refPoint - fmin) / (fmax - fmin)
    
    if m < 4:    
        pl = popObjs[np.lexsort(popObjs.T[::-1])]
        
        S = [(1, pl)]
        
        for k in range(m-1):
            S_ = []
            for i in range(len(S)):
                Stemp = slice(S[i][1], k, refPoint)
                for j in range(len(Stemp)):
                    temp = (Stemp[j][0] * S[i][0], Stemp[j][1])
                    S_.append(temp)
            S = S_
        
        hyperVolume = 0
        
        for i in range(len(S)):
            p = S[i][1][0]
            hyperVolume += S[i][0] * np.abs(p[m-1] - refPoint[m-1])
            
    else:
         upperBounds = np.max(np.vstack((popObjs, refPoint)), axis=0)
         lowerBounds = np.min(np.vstack((popObjs, refPoint)), axis=0)
         
         totalHyperVolume = np.prod(upperBounds - lowerBounds)
         
         nSamples = int(1e6)
         
         samples = np.random.uniform(lowerBounds, upperBounds, (int(nSamples), m))
         
         dominated = np.any(np.all(popObjs <= samples[:, None], axis=2), axis=1)
         hyperVolume = np.sum(dominated) / nSamples * totalHyperVolume
    
    return hyperVolume

def slice(pl, k, refPoint):
    p = head(pl)
    pl =  tail(pl)
    ql = []
    S = []
    while len(pl) > 0:
        ql = insert(p, k+1, ql)
        p_ = head(pl)
        cell_ = [abs(p[k] - p_[k]), ql]
        S = add(cell_, S)
        p = p_
        pl = tail(pl)
    
    ql = insert(p, k+1, ql)
    cell_ = [abs(p[k] - refPoint[k]), ql]
    S = add(cell_, S)
    return S
def insert(p, k, pl):
    flag1 = 0
    flag2 = 0
    ql = []
    
    while len(pl) > 0 and head(pl)[k] < p[k]:
        ql.append(head(pl))
        pl = tail(pl)
    
    ql.append(p)
    m = len(p)
    
    while len(pl) > 0:
        q = head(pl)
        for i in range(k, m):
            if p[i] < q[i]:
                flag1 = 1
            elif p[i] > q[i]:
                flag2 = 1
        
        if not (flag1 == 1 and flag2 == 0):
            ql.append(head(pl))
        pl = tail(pl)
    
    return ql

def head(pl):
    if len(pl) == 0:
        return []
    else:
        return pl[0]

def tail(pl):
    if len(pl) < 2:
        return []
    else:
        return pl[1:]

def add(cell_, S):
    n = len(S)
    m = 0
    for k in range(n):
        if np.array_equal(S[k][1], cell_[1]):
            S[k][0] += cell_[0]
            m = 1
            break
    if m == 0:
        S.append(cell_)
    return S