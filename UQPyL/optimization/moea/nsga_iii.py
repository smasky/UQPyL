# Non-dominated Sorting Genetic Algorithm III (NSGA-III) <Multi>
import numpy as np
from typing import Optional

from ..base import AlgorithmABC
from ..population import Population
from ..core.numerical import shiftedObjectives, unitVectors
from ..core import tourSelect, uniformPoint, NDSort, crowdingDist, gaOperator


class NSGAIII(AlgorithmABC):
    """
    Multi-objective NSGA-III algorithm.

    Examples:
        >>> nsgaiii = NSGAIII(nPop=92, maxFEs=5000)
        >>> res = nsgaiii.run(problem, seed=1234)
        >>> print(res.bestObjs)

    References:
        [1] K. Deb and H. Jain, An evolutionary many-objective optimization algorithm
            using reference-point-based nondominated sorting approach, part I:
            solving problems with box constraints, IEEE Transactions on Evolutionary
            Computation, vol. 18, no. 4, pp. 577-601, 2014.
    """

    name = "NSGAIII"
    alg_type = "MOEA"

    def __init__(
        self,
        proC: float = 1.0,
        disC: float = 20.0,
        proM: float = 1.0,
        disM: float = 20.0,
        nPop: int = 50,
        maxFEs=50000,
        maxIters=1000,
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
            proC: Crossover probability.
            disC: Crossover distribution index.
            proM: Mutation probability.
            disM: Mutation distribution index.
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
        self.set("proC", proC)
        self.set("disC", disC)
        self.set("proM", proM)
        self.set("disM", disM)
        self.set("nPop", nPop)

    # -------------------------Public Functions------------------------#
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

        # Parameter Setting
        proC, disC, proM, disM = self.get("proC", "disC", "proM", "disM")
        nPop = self.get("nPop")

        # Generate uniform reference points
        Z, nPop = uniformPoint(nPop, problem.nOutput)

        # Generate initial population
        pop = self.initPop(nPop, initialPop=initialPop)
        self.update(pop)

        # Perform non-dominated sorting
        frontNo, _ = NDSort(pop.objs, pop.cons, conWgt=pop.conWgt)

        # Iterative process
        while self.checkTermination(pop):
            # Calculate crowding distance
            crowdDis = crowdingDist(pop.objs, frontNo)

            # Select mating pool using tournament selection
            matingIdx = tourSelect(2, len(pop), frontNo, -crowdDis, rng=self.rng)
            matingPool = pop[matingIdx]

            # Generate offspring using genetic operations
            offspringDecs = gaOperator(
                matingPool.decs, self.searchUb, self.searchLb, proC, disC, proM, disM, rng=self.rng
            )
            offspring = Population(offspringDecs)

            # Evaluate the offspring
            self.evaluate(offspring)

            # Merge offspring with current population
            pop.merge(offspring)

            # Update the minimum objective values
            Zmin = np.min(pop.objs, axis=0, keepdims=True)

            # Select the best individuals to form the new population
            nextIdx, frontNo = self.environmentSelection(pop.objs, pop.cons, Z, Zmin, conWgt=pop.conWgt)
            pop = pop[nextIdx]

            pop.frontNo = frontNo
            self.update(pop, completed=True)

        # Return the final result
        return self.finalize()

    def environmentSelection(self, popObjs, popCons, Z, Zmin, conWgt=None):
        """
        Perform environmental selection to choose the next generation.

        Args:
            popObjs: Objective values of current population.
            Z: Reference points.
            Zmin: Minimum objective values.

        Returns:
            Selected offspring for the next generation.
        """

        N = Z.shape[0]

        # Perform non-dominated sorting
        frontNo, maxFNo = NDSort(popObjs, popCons, N, conWgt=conWgt)

        # Determine which individuals to keep
        nextIdx = frontNo < maxFNo

        # Identify the last front
        lastIdx = np.where(frontNo == maxFNo)[0]

        # Separate the population into selected and last front individuals
        popObjs1 = popObjs[nextIdx]
        popObjs2 = popObjs[lastIdx]

        # Select individuals from the last front
        choose = self.lastSelection(popObjs1, popObjs2, N - popObjs1.shape[0], Z, Zmin)

        # Update the selection
        nextIdx[lastIdx[choose]] = True

        return nextIdx, frontNo[nextIdx]

    def lastSelection(self, PopObj1, PopObj2, K, Z, Zmin):
        """
        Select individuals from the last front based on reference points.

        Args:
            PopObj1: Objective values of selected individuals.
            PopObj2: Objective values of individuals in the last front.
            K: Number of individuals to select.
            Z: Reference points.
            Zmin: Minimum objective values.

        Returns:
            Boolean array indicating selected individuals.
        """

        PopObj, _ = shiftedObjectives(np.vstack((np.atleast_2d(Zmin), PopObj1, PopObj2)))
        PopObj = PopObj[1:]
        N, M = PopObj.shape
        N1 = PopObj1.shape[0]
        N2 = PopObj2.shape[0]
        NZ = Z.shape[0]

        # Normalization
        # Detect the extreme points
        Extreme = np.zeros(M, dtype=int)
        w = np.zeros((M, M)) + 1e-6 + np.eye(M)
        for i in range(M):
            Extreme[i] = np.argmin(np.max(PopObj / w[i], axis=1))

        # Calculate the intercepts of the hyperplane constructed by the extreme points
        span = np.max(PopObj, axis=0)
        fallback = np.where(span > 0, span, 1.0)
        try:
            hyperplane = np.linalg.solve(PopObj[Extreme, :], np.ones(M))
            with np.errstate(over="ignore", divide="ignore"):
                a = np.divide(1.0, hyperplane, out=fallback.copy(), where=hyperplane > 0)
            if np.any(hyperplane <= 0) or not np.all(np.isfinite(a)):
                a = fallback
        except np.linalg.LinAlgError:
            a = fallback
        PopObj = PopObj / a

        unitZ = unitVectors(Z)
        projection = PopObj @ unitZ.T
        Distance = np.linalg.norm(PopObj[:, None, :] - projection[:, :, None] * unitZ, axis=2)

        # Find the nearest reference point for each solution
        d = np.min(Distance, axis=1)
        pi = np.argmin(Distance, axis=1)

        # Calculate the number of associated solutions for each reference point
        rho = np.histogram(pi[:N1], bins=np.arange(NZ + 1))[0]

        # Environmental selection
        Choose = np.zeros(N2, dtype=bool)
        Zchoose = np.ones(NZ, dtype=bool)

        # Select K solutions one by one
        while np.sum(Choose) < K:
            # Find the least crowded reference point
            Temp = np.where(Zchoose)[0]
            if Temp.size == 0:
                break
            Jmin = Temp[np.where(rho[Temp] == np.min(rho[Temp]))[0]]
            j = Jmin[int(self.rng.integers(len(Jmin)))]

            # Find unselected solutions associated with this reference point
            indices = np.where((~Choose) & (pi[N1:] == j))[0]

            if indices.size > 0:
                if rho[j] == 0:
                    s = np.argmin(d[N1 + indices])
                else:
                    s = int(self.rng.choice(indices.size))
                Choose[indices[s]] = True
                rho[j] += 1
            else:
                Zchoose[j] = False

        return Choose
