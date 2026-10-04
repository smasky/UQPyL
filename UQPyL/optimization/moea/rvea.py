# Reference vector guided evolutionary algorithm (RVEA) <Multi>
import numpy as np
from ..core.numerical import shiftedObjectives, unitVectors
from scipy.spatial.distance import cdist
from typing import Optional

from ..base import AlgorithmABC
from ..core import uniformPoint, gaOperator
from ..population import Population
from ..core.constraint import calcConstraintViolation


class RVEA(AlgorithmABC):
    """
    Multi-objective reference vector guided evolutionary algorithm.

    Examples:
        >>> rvea = RVEA(nPop=100, maxFEs=5000)
        >>> res = rvea.run(problem, seed=1234)
        >>> print(res.bestObjs)

    References:
        [1] R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff, A reference vector guided
            evolutionary algorithm for many-objective optimization, IEEE Transactions
            on Evolutionary Computation, vol. 20, no. 5, pp. 773-791, 2016.
    """

    name = "RVEA"
    alg_type = "MOEA"

    def __init__(
        self,
        alpha: float = 2.0,
        fr: float = 0.1,
        nPop: int = 50,
        maxFEs: int = 50000,
        maxIters: int = 1000,
        maxTolerates=None,
        tolerate=1e-6,
        verboseFlag: bool = True,
        verboseFreq: int = 10,
        logFlag: bool = True,
        saveFlag: bool = True,
        saveFreq: int = 100,
        hvRefPoint=None,
        historyFreq: int = 10,
        hvFlag: bool = True,
        hvFreq: int = 10,
        hvSamples: int = 10_000,
    ):
        """
        Initialize the algorithm.

        Args:
            alpha: Angle penalty parameter.
            fr: Reference vector adaptation frequency.
            nPop: Population size.
            maxFEs: Maximum number of function evaluations.
            maxIters: Maximum number of iterations.
            maxTolerates: Maximum tolerated non-improving iterations.
            tolerate: Improvement tolerance.
            verboseFlag: Whether to print terminal output.
            verboseFreq: Summary output frequency.
            logFlag: Whether to save full text logs.
            saveFlag: Whether to save sqlite results.
            saveFreq: SQLite snapshot save frequency.
            historyFreq: Full in-memory snapshot interval; None keeps only the final snapshot.
            hvFlag: Enable hypervolume diagnostics.
            hvFreq: Hypervolume interval in completed iterations; initialization and final are included.
            hvSamples: Monte Carlo sample budget for four or more objectives.
        """
        super().__init__(
            maxFEs,
            maxIters,
            maxTolerates,
            tolerate,
            verboseFlag,
            verboseFreq,
            logFlag,
            saveFlag,
            saveFreq,
            hvRefPoint=hvRefPoint,
            historyFreq=historyFreq,
            hvFlag=hvFlag,
            hvFreq=hvFreq,
            hvSamples=hvSamples,
        )

        # Set user-defined parameters
        self.set("alpha", alpha)
        self.set("fr", fr)
        self.set("nPop", nPop)

    def run(self, problem, seed: Optional[int] = None, initialPop=None):
        """
        Run the algorithm on the given problem.

        Args:
            problem: Problem instance.
            seed: Random seed.
            initialPop: Optional initial population or decision matrix.

        Returns:
            OptResult: Final optimization result.
        """
        # setup algorithm
        self.setup(problem, seed)

        # Parameters setting
        alpha, fr = self.get("alpha", "fr")
        if self.maxFEs is None and self.maxIter is None:
            raise ValueError("RVEA requires an evaluation or iteration budget for its angle schedule.")
        nPop = self.get("nPop")

        # Generate initial reference vectors
        V0, nPop = uniformPoint(nPop, problem.nOutput)
        V = np.copy(V0)
        self.referenceScale = np.ones(problem.nOutput)
        self.referenceScaleExponent = 0

        # Generate initial population
        pop = self.initPop(nPop, initialPop=initialPop)
        self.update(pop)

        # Iterative process
        while self.checkTermination(pop):
            # Select mating pool randomly
            matingPoolIdx = self.rng.integers(0, len(pop), nPop)
            matingPool = pop[matingPoolIdx]
            # Generate offspring using genetic operations
            offspringDecs = gaOperator(matingPool.decs, self.searchUb, self.searchLb, rng=self.rng)
            offspring = Population(offspringDecs)

            # Evaluate the offspring
            self.evaluate(offspring)

            # Environmental selection
            pop.merge(offspring)
            progress = self.FEs / self.maxFEs if self.maxFEs is not None else (self.iters + 1) / self.maxIter
            nextIdx = self.environmentSelection(pop.objs, V, min(1.0, progress) ** alpha, pop.cons, pop.conWgt)
            pop = pop[nextIdx]

            # Check if reference vectors need to be updated
            if self.maxFEs is not None:
                condition = not (np.ceil(self.FEs / nPop) % np.ceil(fr * self.maxFEs / nPop))
            else:
                condition = not ((self.iters + 1) % max(1, int(np.ceil(fr * self.maxIter))))

            if condition:
                # Update reference vectors
                V = self.updateReferenceVector(pop.objs, V0)
            self.update(pop, completed=True)

        # Return the final result
        return self.finalize()

    def updateReferenceVector(self, popObjs, V):
        """
        Update the reference vectors based on the current population.

        Args:
            pop: Current population.
            V: Initial reference vectors.

        Returns:
            Updated reference vectors.
        """
        shifted, exponent = shiftedObjectives(popObjs, perColumn=True)
        span = np.max(shifted, axis=0)
        previous = getattr(self, "referenceScale", np.ones(popObjs.shape[1]))
        previousExponent = getattr(self, "referenceScaleExponent", 0)
        self.referenceScale = np.where(span > 0, span, previous)
        self.referenceScaleExponent = np.where(span > 0, exponent, previousExponent)
        # Preserve physical vectors when representable. Otherwise row scaling
        # keeps their directions, including axis vectors with very small spans.
        weighted = V * self.referenceScale
        with np.errstate(over="ignore", under="ignore"):
            restored = np.ldexp(weighted, self.referenceScaleExponent)
            if np.all(np.isfinite(restored)) and np.all((weighted == 0) | (restored != 0)):
                return restored
            rowExponent = np.max(np.where(weighted != 0, self.referenceScaleExponent, -1075), axis=1)
            scaled = np.ldexp(weighted, self.referenceScaleExponent - rowExponent[:, None])
        return unitVectors(scaled)

    def environmentSelection(self, popObjs, V, theta, popCons=None, conWgt=None):
        """
        Perform environmental selection to choose the next generation.

        Args:
            pop: Merged population of current and offspring.
            V: Reference vectors.
            theta: Angle control parameter.

        Returns:
            Selected population for the next generation.
        """
        if popCons is not None:
            violation = calcConstraintViolation(popCons, conWgt)
            feasible = np.flatnonzero(violation <= 0)
            if not feasible.size:
                return np.argsort(violation, kind="stable")[: V.shape[0]]
            # Preserve reference-vector selection among feasible solutions.
            selected = self.environmentSelection(popObjs[feasible], V, theta)
            return feasible[selected]

        M = popObjs.shape[1]

        nV = V.shape[0]

        # Normalize the objective values
        popObjs, _ = shiftedObjectives(popObjs)

        unitV = unitVectors(V)
        cosine = np.clip(unitV @ unitV.T, -1.0, 1.0)
        np.fill_diagonal(cosine, -1.0)
        gamma = np.maximum(np.min(np.arccos(cosine), axis=1), 1e-12)
        objectiveNorm = np.hypot.reduce(popObjs, axis=1, keepdims=True)
        unitObjs = np.divide(popObjs, objectiveNorm, out=np.zeros_like(popObjs), where=objectiveNorm > 0)
        angle = np.arccos(np.clip(unitObjs @ unitV.T, -1.0, 1.0))
        # The ideal point has zero APD in every direction, without undefined angles.
        angle[objectiveNorm[:, 0] == 0] = 0.0

        # Associate each solution with a reference vector
        associate = np.argmin(angle, axis=1)

        next = np.ones(nV, dtype=np.int32) * -1

        for i in np.unique(associate):
            current1 = np.where(associate == i)[0]

            if len(current1) > 0:
                # Calculate the APD value for each solution
                APD = (1 + M * theta * angle[current1, i] / gamma[i]) * objectiveNorm[current1, 0]
                # Select the one with the minimum APD value
                best = np.argmin(APD)
                next[i] = current1[best]

        nextIdx = next[next != -1].astype(int)

        return nextIdx
