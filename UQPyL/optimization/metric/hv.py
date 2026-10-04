import numpy as np
import warnings
from ..core.numerical import automaticReference


def HV(popObjs, refPoint=None, normalize=True, nSamples: int = 1_000_000, rng=None, *, batchSize: int = 4096):
    """Compute exact HV below four objectives, otherwise estimate by sampling.

    ``batchSize`` limits samples held at once in the Monte Carlo branch.
    Population comparisons also use blocks of at most 256 points. With a
    seeded NumPy generator, batching preserves the samples and RNG state.
    """
    popObjs = np.atleast_2d(np.asarray(popObjs, dtype=float))
    _, m = popObjs.shape

    if popObjs.size == 0:
        return 0.0

    if refPoint is None:
        worst = np.max(popObjs, axis=0)
        refPoint = automaticReference(worst, 0.1)
    else:
        refPoint = np.asarray(refPoint, dtype=float).reshape(-1)

    if refPoint.shape != (m,) or not np.all(np.isfinite(refPoint)):
        raise ValueError("refPoint must contain one finite value per objective.")
    if not np.all(np.isfinite(popObjs)):
        raise ValueError("popObjs must contain finite objective values.")

    # Keep the ordinary path (including its sampling stream) unchanged. Extreme
    # axes need separate binary scales so intermediate volumes do not overflow
    # before multiplication by a very small width in a later dimension.
    magnitude = np.maximum(np.max(np.abs(popObjs), axis=0), np.abs(refPoint))
    powers = np.frexp(magnitude)[1]
    extreme = (
        np.any(np.abs(powers) > 400) or np.sum(np.maximum(powers, 0)) > 900 or np.sum(np.minimum(powers, 0)) < -900
    )
    if extreme and not normalize:
        with np.errstate(under="ignore"):
            scaled = np.ldexp(popObjs, -powers)
            reference = np.ldexp(refPoint, -powers)
        value = HV(scaled, reference, normalize=False, nSamples=nSamples, rng=rng, batchSize=batchSize)
        with np.errstate(over="ignore", under="ignore"):
            result = np.ldexp(value, int(np.sum(powers)))
        if value > 0 and (result == 0 or np.isinf(result)):
            warnings.warn(
                "Hypervolume exceeds floating-point range; returning zero or infinity.", RuntimeWarning, stacklevel=2
            )
        return float(result)

    if normalize:
        if extreme:
            # Include zero/one normalization anchors in the same coordinate map.
            powers = np.maximum(powers, 1)
            popObjs = np.ldexp(popObjs, -powers)
            refPoint = np.ldexp(refPoint, -powers)
            upperAnchor = np.ldexp(np.ones((1, m)), -powers)
        else:
            upperAnchor = np.ones((1, m))
        fmin = np.min(np.vstack((popObjs, np.zeros((1, m)))), axis=0)
        fmax = np.max(np.vstack((popObjs, upperAnchor)), axis=0)

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
        if nSamples <= 0:
            raise ValueError("nSamples must be positive.")
        if isinstance(batchSize, (bool, np.bool_)) or not isinstance(batchSize, (int, np.integer)) or batchSize <= 0:
            raise ValueError("batchSize must be a positive integer.")
        if rng is None:
            rng = np.random.default_rng()
        dominatedCount = 0
        for start in range(0, nSamples, batchSize):
            size = min(batchSize, nSamples - start)
            samples = rng.uniform(lowerBounds, upperBounds, (size, m))
            dominated = np.zeros(size, dtype=bool)
            for pointStart in range(0, len(popObjs), 256):
                points = popObjs[pointStart : pointStart + 256]
                dominated |= np.any(np.all(points <= samples[:, None], axis=2), axis=1)
                if np.all(dominated):
                    break
            dominatedCount += int(np.count_nonzero(dominated))
        hyperVolume = dominatedCount / nSamples * totalHyperVolume

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
