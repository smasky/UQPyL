import numpy as np


def HV(popObjs, refPoint=None, normalize=True, nSamples: int = 1_000_000, rng=None):
    popObjs = np.atleast_2d(np.asarray(popObjs, dtype=float))
    _, m = popObjs.shape

    if popObjs.size == 0:
        return 0.0

    if refPoint is None:
        refPoint = np.max(popObjs, axis=0) * 1.1
    else:
        refPoint = np.asarray(refPoint, dtype=float)

    if normalize:
        fmin = np.min(np.vstack((popObjs, np.zeros((1, m)))), axis=0)
        fmax = np.max(np.vstack((popObjs, np.ones((1, m)))), axis=0)

        denom = fmax - fmin
        popObjs = (popObjs - fmin) / denom
        refPoint = (refPoint - fmin) / denom

    # Only keep the part of the set that can dominate some volume w.r.t. the reference point.
    feasible = np.all(popObjs <= refPoint, axis=1)
    if not np.any(feasible):
        return 0.0
    popObjs = popObjs[feasible]

    if m < 4:
        pl = popObjs[np.lexsort(popObjs.T[::-1])]
        S = [(1, pl)]

        for k in range(m - 1):
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
            width = refPoint[m - 1] - p[m - 1]
            if width > 0:
                hyperVolume += S[i][0] * width
    else:
        upperBounds = refPoint
        lowerBounds = np.min(np.vstack((popObjs, refPoint)), axis=0)
        totalHyperVolume = np.prod(upperBounds - lowerBounds)
        if totalHyperVolume <= 0:
            return 0.0

        nSamples = int(nSamples)
        if rng is None:
            rng = np.random.default_rng()
        samples = rng.uniform(lowerBounds, upperBounds, (nSamples, m))
        dominated = np.any(np.all(popObjs <= samples[:, None], axis=2), axis=1)
        hyperVolume = np.sum(dominated) / nSamples * totalHyperVolume

    return hyperVolume

def slice(pl, k, refPoint):
    p = head(pl)
    pl = tail(pl)
    ql = []
    S = []
    while len(pl) > 0:
        ql = insert(p, k + 1, ql)
        p_ = head(pl)
        width = p_[k] - p[k]
        if width > 0:
            cell_ = [width, ql]
            S = add(cell_, S)
        p = p_
        pl = tail(pl)

    ql = insert(p, k + 1, ql)
    width = refPoint[k] - p[k]
    if width > 0:
        cell_ = [width, ql]
        S = add(cell_, S)
    return S


def insert(p, k, pl):
    ql = []

    while len(pl) > 0 and head(pl)[k] < p[k]:
        ql.append(head(pl))
        pl = tail(pl)

    ql.append(p)
    m = len(p)

    while len(pl) > 0:
        q = head(pl)
        flag1 = 0
        flag2 = 0
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
            # S stores tuples (value, list); tuples are immutable, so replace entry.
            S[k] = (S[k][0] + cell_[0], S[k][1])
            m = 1
            break
    if m == 0:
        S.append(cell_)
    return S
