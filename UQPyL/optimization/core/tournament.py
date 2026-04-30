import numpy as np


def tourSelect(K, N, *fitnesses):
    if len(fitnesses) == 0:
        raise ValueError("At least one fitness matrix is required.")

    validFitnesses = [f for f in fitnesses if f is not None]
    F = np.column_stack(validFitnesses)

    n = F.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    order = np.lexsort(np.fliplr(F).T)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)

    candidates = np.random.randint(0, n, size=(N, K))
    winners = candidates[np.arange(N), np.argmin(rank[candidates], axis=1)]
    return winners
