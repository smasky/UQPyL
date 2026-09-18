import numpy as np
from typing import Literal


def _resolveRng(seed=None, rng=None):
    if seed is not None and rng is not None:
        raise ValueError("Provide only one of seed or rng.")
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        return rng
    return np.random.default_rng(seed)


def _sampleCount(X):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("Split input must be a two-dimensional sample matrix.")
    return len(X)


class RandSelect:
    def __init__(self, pTest: float = 5):
        if not np.isscalar(pTest) or not np.isfinite(pTest) or not 0 < pTest < 100:
            raise ValueError("pTest must be a finite percentage strictly between 0 and 100.")
        self.pTest = pTest / 100

    def split(self, X, *, seed=None, rng=None):
        """Return disjoint indices without consuming global NumPy randomness."""
        nSample = _sampleCount(X)
        generator = _resolveRng(seed, rng)
        if nSample <= 1:
            return np.arange(nSample), np.array([], dtype=int)
        nTest = max(1, min(int(nSample*self.pTest), nSample-1))
        index = generator.permutation(nSample)
        return index[nTest:].copy(), index[:nTest].copy()


class KFold:
    def __init__(self, n_splits: int = 5):
        if isinstance(n_splits, (bool, np.bool_)) or not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
            raise ValueError("n_splits must be an integer of at least 2.")
        self.n_splits = n_splits

    def split(self, X, mode: Literal['full', 'single'] = 'full', *, seed=None, rng=None):
        nSample = _sampleCount(X)
        if self.n_splits > nSample:
            raise ValueError("n_splits must not exceed the number of samples.")
        if mode not in ('full', 'single'):
            raise ValueError("mode must be 'full' or 'single'.")
        index = _resolveRng(seed, rng).permutation(nSample)
        folds = np.array_split(index, self.n_splits)
        train, test = [], []
        for i in range(self.n_splits if mode == 'full' else 1):
            test.append(folds[i].copy())
            train.append(np.concatenate(folds[:i]+folds[i+1:]))
        return train, test
