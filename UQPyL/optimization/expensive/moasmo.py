### Multi-Objective Adaptive Surrogate Modelling-based Optimization
import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional

from ..moea.nsga_ii import NSGAII
from ..base import AlgorithmABC
from ._base import SurrogateOptimization
from ..core import NDSort
from ..core.numerical import shiftedObjectives
from ..population import Population
from ...core import spawn_seed

from ...problem import Problem
from ...surrogate import MultiSurrogate
from ...surrogate.rbf.radial_basis_function import RBF


class MOASMO(SurrogateOptimization):
    """
    Multi-objective adaptive surrogate modelling-based optimization algorithm.

    Examples:
        >>> moasmo = MOASMO(nInit=50, maxFEs=100)
        >>> res = moasmo.run(problem, seed=1234)
        >>> print(res.bestObjs)

    References:
        [1] W. Gong, J. Duan, L. Li, and W. Wang, Multiobjective adaptive surrogate
            modeling-based optimization for parameter estimation of hydrologic models,
            Water Resources Research, vol. 51, no. 8, pp. 6691-6711, 2015.
    """

    name = "MOASMO"
    alg_type = "MOEA"

    def __init__(
        self,
        surrogates: MultiSurrogate = None,
        optimizer: AlgorithmABC = None,
        pct: float = 0.2,
        nInit: int = 50,
        nPop: int = 50,
        advance_infilling: bool = False,
        maxFEs: int = 1000,
        maxIters: int = 100,
        maxTolerates: int = None,
        tolerate: float = 1e-6,
        verboseFlag: bool = True,
        verboseFreq: int = 1,
        logFlag: bool = False,
        saveFlag: bool = False,
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
            surrogates: Surrogate ensemble.
            optimizer: Inner optimizer.
            pct: Infill percentage.
            nInit: Number of initial samples.
            nPop: Population size for the default inner optimizer; a supplied optimizer keeps its own size.
            advance_infilling: Whether to use advanced infilling.
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
        self.set("pct", pct)
        self.set("nInit", nInit)
        self.set("advance_infilling", advance_infilling)

        # Initialize surrogate models
        self.surrogates = surrogates
        self._autoSurrogates = None

        # Initialize optimizer
        if optimizer is not None:
            if not isinstance(optimizer, AlgorithmABC):
                raise ValueError("Please append the type of optimizer!")
            self.optimizer = optimizer
        else:
            self.optimizer = NSGAII(nPop=nPop, maxFEs=5000, hvFlag=False)

        self.optimizer.verboseFlag, self.optimizer.logFlag, self.optimizer.saveFlag = False, False, False

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

        # Initialize surrogate models
        nObj = problem.nObj
        if self.surrogates is None or self.surrogates is self._autoSurrogates:
            self.surrogates = MultiSurrogate(n_surrogates=nObj, models_list=[RBF() for _ in range(nObj)])
            self._autoSurrogates = self.surrogates
        else:
            if not all(callable(getattr(self.surrogates, name, None)) for name in ("fit", "predict")):
                raise TypeError("surrogates must provide fit and predict methods.")
            if getattr(self.surrogates, "n_surrogates", nObj) != nObj:
                raise ValueError("surrogates.n_surrogates must match problem.nObj.")
            if isinstance(self.surrogates, MultiSurrogate):
                self.surrogates._validateModels(self.surrogates.models_list)
                if len(self.surrogates.models_list) != nObj:
                    raise ValueError("surrogates.models_list must contain one model per objective.")

        # Retrieve parameter values
        pct = self.get("pct")
        nInit = self.get("nInit")
        advance_infilling = self.get("advance_infilling")

        nInfilling = max(1, int(pct * nInit))

        # Create a subproblem for surrogate model optimization
        subProblem = Problem(
            nInput=problem.nInput,
            nObj=nObj,
            ub=1.0,
            lb=0.0,
            objFunc=self._predictUnit,
            optType="min",
            xLabels=problem.xLabels,
        )

        # Generate initial population
        pop = self.initPop(nInit, initialPop=initialPop)
        self.update(pop)

        # Iterative optimization process
        while self.checkTermination(pop):
            # Build surrogate models
            self._fitSurrogate(self.surrogates, pop)

            # Run optimization on the surrogate model
            res = self.optimizer.run(subProblem, seed=spawn_seed(self.rng))
            bestDecs = np.asarray(res.bestDecs)
            bestObjs = np.asarray(res.bestObjs)
            offSpring = Population(decs=bestDecs, objs=bestObjs)

            if not advance_infilling:
                if offSpring.nPop > nInfilling:
                    bestOff = offSpring.getBest(nInfilling)
                else:
                    bestOff = offSpring

            else:
                if offSpring.nPop > nInfilling:
                    Known_FrontNo, _ = NDSort(pop.objs, pop.cons, conWgt=pop.conWgt)
                    Unknown_FrontNo, _ = NDSort(offSpring.objs, offSpring.cons, conWgt=offSpring.conWgt)

                    Known_best_Y = pop.objs[np.where(Known_FrontNo == 1)]
                    Unknown_best_Y = offSpring.objs[np.where(Unknown_FrontNo == 1)]
                    Unknown_best_X = offSpring.decs[np.where(Unknown_FrontNo == 1)]

                    added_points_X = []

                    # The first front can be smaller than the requested batch.
                    # Remaining slots are filled by _novelCandidates below.
                    for _ in range(min(nInfilling, len(Unknown_best_X))):
                        distanceValues, _ = shiftedObjectives(np.vstack((Unknown_best_Y, Known_best_Y)))
                        distances = cdist(distanceValues[: len(Unknown_best_Y)], distanceValues[len(Unknown_best_Y) :])

                        max_distance_index = np.argmax(np.min(distances, axis=1))

                        added_point = Unknown_best_Y[max_distance_index]
                        added_points_X.append(Unknown_best_X[max_distance_index])
                        Known_best_Y = np.append(Known_best_Y, [added_point], axis=0)

                        Unknown_best_Y = np.delete(Unknown_best_Y, max_distance_index, axis=0)
                        Unknown_best_X = np.delete(Unknown_best_X, max_distance_index, axis=0)

                    BestX = np.copy(np.array(added_points_X))
                    bestOff = Population(decs=BestX)
                else:
                    bestOff = offSpring

            # Compare canonical inputs, including duplicates within the batch.
            decs = self._novelCandidates(bestOff.decs, pop, count=nInfilling)
            if not len(decs):
                self.state.stopReason = "no_novel_candidates"
                break
            bestOff = Population(decs)
            self.evaluate(bestOff)

            pop.add(bestOff)
            self.update(pop, completed=True)

        return self.finalize()

    def _predictUnit(self, X):
        return self.surrogates.predict(self.problem.canonicalize_unit(X))
