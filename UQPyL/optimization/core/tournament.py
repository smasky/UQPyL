import numpy as np


def tourSelect(K, N, *fitnesses, rng=None):
    if len(fitnesses) == 0:
        raise ValueError("At least one fitness matrix is required.")

    if rng is None:
        rng = np.random.default_rng()

    validFitnesses = [f for f in fitnesses if f is not None]
    F = np.column_stack(validFitnesses)

    n = F.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    order = np.lexsort(np.fliplr(F).T)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)

    candidates = rng.integers(0, n, size=(N, K))
    winners = candidates[np.arange(N), np.argmin(rank[candidates], axis=1)]
    return winners
