"""Unit-coordinate boundaries shared by surrogate-assisted optimizers."""

import numpy as np
from itertools import product

from ..base import AlgorithmABC
from ...core import spawn_seed


class SurrogateOptimization(AlgorithmABC):
    constraintHandling = "evaluation_only"

    def validateProblem(self, problem):
        super().validateProblem(problem)
        # Candidate generation currently models objectives only (C17).
        optimizer = getattr(self, "optimizer", None)
        if not callable(getattr(optimizer, "run", None)):
            raise TypeError("optimizer must provide a run method.")
        if isinstance(optimizer, AlgorithmABC):
            optimizer._checkObjectiveCount(problem.nObj)
        if hasattr(self, "surrogate"):
            if not all(callable(getattr(self.surrogate, name, None)) for name in ("fit", "predict")):
                raise TypeError("surrogate must provide fit and predict methods.")
            if self.requiresUncertainty and getattr(self.surrogate, "supportsUncertainty", None) is False:
                raise ValueError(f"{self.name} requires a surrogate with predictive variance support.")

    def _fitSurrogate(self, surrogate, pop):
        unitX = self.problem.canonicalize_unit(pop.decs)
        _, indices = np.unique(unitX, axis=0, return_index=True)
        indices = np.sort(indices)
        # The same physical solution must not become multiple model locations.
        surrogate.rng = np.random.default_rng(spawn_seed(self.rng))
        surrogate.fit(unitX[indices], pop.objs[indices])

    def _novelCandidates(self, candidates, pop, count=1, tolerance=1e-12):
        known = self.problem.canonicalize_unit(pop.decs)
        selected = []

        def consider(rows):
            nonlocal known
            for row in self.problem.canonicalize_unit(rows):
                if not len(known) or np.min(np.linalg.norm(known - row, axis=1)) > tolerance:
                    selected.append(row.copy())
                    known = np.vstack((known, row))
                    if len(selected) >= count:
                        return True
            return False

        if count <= 0:
            return np.empty((0, self.problem.nInput))
        if not consider(candidates):
            # Exhaust small finite domains exactly instead of repeatedly
            # proposing encodings of already evaluated discrete solutions.
            levels = []
            for index in range(self.problem.nInput):
                if index in self.problem.idxD:
                    size = len(self.problem.varSet[index])
                elif index in self.problem.idxI:
                    size = int(np.floor(self.problem.ub[0, index]) - np.ceil(self.problem.lb[0, index]) + 1)
                elif self.problem.ub[0, index] == self.problem.lb[0, index]:
                    size = 1
                else:
                    levels = []
                    break
                levels.append(size)
            if levels and np.prod(np.asarray(levels, dtype=object)) <= 4096:
                grid = np.array(list(product(*[(np.arange(n) + 0.5) / n for n in levels])))
                consider(grid[self.rng.permutation(len(grid))])
                return np.asarray(selected).reshape(-1, self.problem.nInput)
            # Bounded retries for continuous or large finite domains.
            for _ in range(8):
                if consider(self.rng.random((64, self.problem.nInput))):
                    break
        return np.asarray(selected).reshape(-1, self.problem.nInput)
