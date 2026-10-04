# Non-dominated Sorting Genetic Algorithm II (NSGA-II) <Multi>
import numpy as np
from typing import Optional

from ..base import AlgorithmABC
from ..population import Population
from ..core import NDSort, crowdingDist, tourSelect, gaOperator


class NSGAII(AlgorithmABC):
    """
    Non-dominated Sorting Genetic Algorithm II <Multi>
    ------------------------------------------------

    Examples:
        >>> nsgaii = NSGAII(nPop=50, maxFEs=5000)
        >>> res = nsgaii.run(problem, seed=1234)
        >>> print(res.bestObjs)

    References:
        [1] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, A fast and elitist
            multiobjective genetic algorithm: NSGA-II, IEEE Transactions on
            Evolutionary Computation, vol. 6, no. 2, pp. 182-197, 2002.
    """

    name = "NSGAII"
    alg_type = "MOEA"

    def __init__(
        self,
        proC: float = 1.0,
        disC: float = 20.0,
        proM: float = 1.0,
        disM: float = 20.0,
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
        Initialize the NSGA-II algorithm with user-defined parameters.

        Args:
            proC: Crossover probability.
            disC: Crossover distribution index.
            proM: Mutation probability.
            disM: Mutation distribution index.
            nPop: Population size.
            maxFEs: Maximum number of function evaluations.
            maxIters: Maximum number of iterations.
            maxTolerateTimes: Maximum number of tolerated iterations without improvement.
            tolerate: Tolerance for improvement.
            verbose: Flag to enable verbose output.
            verboseFreq: Frequency of verbose output.
            logFlag: Flag to enable logging.
            saveFlag: Flag to enable saving results.
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
        Execute the NSGA-II algorithm on the specified problem.

        Args:
            problem: Problem instance.
                This object defines the optimization problem, including
                input dimension, objective dimension, bounds, and evaluation methods.

            initialPop: Optional initial population or decision matrix.

        Returns:
            OptResult: Final optimization result.
        """
        # setup algorithm
        self.setup(problem, seed)

        # Parameter Setting
        proC, disC, proM, disM = self.get("proC", "disC", "proM", "disM")
        nPop = self.get("nPop")

        # Generate initial population
        pop = self.initPop(nPop, initialPop=initialPop)

        # Perform environmental selection
        _, frontNo, CrowdDis = self.environmentalSelection(pop.decs, pop.objs, pop.cons, pop.conWgt, nPop)
        pop.frontNo = frontNo
        pop.crowdDis = CrowdDis
        self.update(pop)

        # Iterative process
        while self.checkTermination(pop):
            # Select mating pool using tournament selection
            matingIdx = tourSelect(2, len(pop), frontNo, -CrowdDis, rng=self.rng)
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

            # Perform environmental selection
            nextIdx, frontNo, CrowdDis = self.environmentalSelection(pop.decs, pop.objs, pop.cons, pop.conWgt, nPop)
            pop = pop[nextIdx]
            pop.frontNo = frontNo
            pop.crowdDis = CrowdDis
            self.update(pop, completed=True)

        # Return the final result
        return self.finalize()

    # -------------------------Private Functions--------------------------#
    def environmentalSelection(self, popDecs, popObjs, popCons=None, conWgt=None, n=None):
        """
        Perform environmental selection to choose the next generation.

        Args:
            pop: Current population.
            n: Number of individuals to select.

        Returns:
            The next population, front numbers, and crowding distances.
        """

        # Non-dominated sorting
        frontNo, maxFNo = NDSort(popObjs, popCons, n, conWgt=conWgt)

        # Determine the next population
        nextIdx = frontNo < maxFNo

        # Handle the last front
        mask_last = frontNo == maxFNo
        if np.any(mask_last):
            crowdDis = np.zeros(popDecs.shape[0])
            crowdDis[mask_last] = crowdingDist(popObjs[mask_last], np.ones(np.sum(mask_last)))
        else:
            crowdDis = np.zeros(popDecs.shape[0])

        numSelected = n - np.sum(nextIdx)
        if numSelected > 0:
            last_indices = np.flatnonzero(mask_last)
            rank = np.argpartition(-crowdDis[last_indices], numSelected - 1)[:numSelected]
            nextIdx[last_indices[rank]] = True

        # Form the next population
        nextFrontNo = frontNo[nextIdx]
        nextCrowdDis = crowdDis[nextIdx]

        return nextIdx, nextFrontNo, nextCrowdDis
