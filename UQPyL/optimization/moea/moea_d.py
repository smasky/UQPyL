# Multi-objective Evolutionary Algorithm based on Decomposition (MOEAD) <Multi>
import math
import numpy as np
from typing import Literal, Optional
from scipy.spatial import distance

from ..base import AlgorithmABC
from ..population import Population
from ..core import uniformPoint, gaOperatorHalf, calcConstraintViolation
from ..core.numerical import shiftedObjectives


class MOEAD(AlgorithmABC):
    """
    Multi-objective evolutionary algorithm based on decomposition.

    Examples:
        >>> moead = MOEAD(nPop=100, maxFEs=5000)
        >>> res = moead.run(problem, seed=1234)
        >>> print(res.bestObjs)

    References:
        [1] Q. Zhang and H. Li, MOEA/D: A multiobjective evolutionary algorithm based
            on decomposition, IEEE Transactions on Evolutionary Computation,
            vol. 11, no. 6, pp. 712-731, 2007.
    """

    name = "MOEA_D"
    alg_type = "MOEA"

    def __init__(
        self,
        aggregation: Literal["PBI", "TCH", "TCH_N", "TCH_M"] = "TCH",
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
            aggregation: Aggregation method.
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

        # Initialize the base class with common parameters
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

        # Set specific parameters for MOEAD
        self.set("aggregation", aggregation)
        self.set("nPop", nPop)

    # -------------------Public Functions-----------------------#
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

        # Retrieve parameter values
        aggregation = self.get("aggregation")
        if aggregation not in {"PBI", "TCH", "TCH_N", "TCH_M"}:
            raise ValueError("Unsupported MOEAD aggregation.")

        nPop = self.get("nPop")

        # Generate uniform weight vectors
        W, N = uniformPoint(nPop, problem.nOutput)

        # Adjust population size
        nPop = N
        if nPop < 2:
            raise ValueError("MOEAD requires at least two reference directions.")
        T = min(nPop, max(2, math.ceil(nPop / 10)))

        # Calculate the distance matrix and sort neighbors
        B = distance.cdist(W, W, metric="euclidean")
        B = np.argsort(B, axis=1)
        B = B[:, 0:T]

        # Generate initial population
        pop = self.initPop(nPop, initialPop=initialPop)
        self.update(pop)

        # Initialize the ideal point
        Z = np.min(pop.objs, axis=0).reshape(1, -1)

        # Main loop of the algorithm
        while self.checkTermination(pop):
            for i in range(nPop):
                # Select parents from the neighborhood
                P = B[i, self.rng.permutation(B.shape[1])].ravel()

                # Generate offspring using genetic operations
                subPop = pop[P[0:2]]
                offspringDecs = gaOperatorHalf(subPop.decs, self.searchUb, self.searchLb, 1, 20, 1, 20, rng=self.rng)
                offspring = Population(offspringDecs)
                # Evaluate the offspring
                self.evaluate(offspring)

                # Update the ideal point
                Z = np.min(np.vstack((Z, offspring.objs)), axis=0).reshape(1, -1)

                # A common binary scale preserves decomposition scores while
                # keeping subtraction and norms in range. TCH_N normalizes axes.
                scaled, _ = shiftedObjectives(
                    np.vstack((Z, pop.objs, offspring.objs)), perColumn=aggregation == "TCH_N"
                )
                deltaP = scaled[1:-1][P]
                deltaO = scaled[-1:]
                if aggregation == "PBI":
                    unitW = W[P] / np.linalg.norm(W[P], axis=1, keepdims=True)
                    projectionP = np.sum(deltaP * unitW, axis=1)
                    projectionO = np.sum(deltaO * unitW, axis=1)
                    residualP = deltaP - projectionP[:, None] * unitW
                    residualO = deltaO - projectionO[:, None] * unitW
                    g_old = projectionP + 5 * np.hypot.reduce(residualP, axis=1)
                    g_new = projectionO + 5 * np.hypot.reduce(residualO, axis=1)
                elif aggregation == "TCH_N":
                    span = np.max(scaled[1:-1], axis=0)
                    span = np.where(span > 0, span, 1.0)
                    g_old = np.max(deltaP / span * W[P], axis=1)
                    g_new = np.max(deltaO / span * W[P], axis=1)
                elif aggregation == "TCH_M":
                    g_old = np.max(deltaP / W[P], axis=1)
                    g_new = np.max(deltaO / W[P], axis=1)
                else:
                    g_old = np.max(deltaP * W[P], axis=1)
                    g_new = np.max(deltaO * W[P], axis=1)

                parentCons = None if pop.cons is None else pop.cons[P]
                offspringCons = None if offspring.cons is None else np.repeat(offspring.cons, len(P), axis=0)
                parentCV = calcConstraintViolation(parentCons, pop.conWgt)
                offspringCV = calcConstraintViolation(offspringCons, pop.conWgt)

                if parentCV is None:
                    replaceMask = g_old >= g_new
                else:
                    parentFeasible = parentCV <= 0
                    offspringFeasible = offspringCV <= 0
                    replaceMask = np.zeros(len(P), dtype=bool)
                    replaceMask[offspringFeasible & ~parentFeasible] = True
                    bothInfeasible = ~offspringFeasible & ~parentFeasible
                    replaceMask[bothInfeasible] = offspringCV[bothInfeasible] < parentCV[bothInfeasible]
                    bothFeasible = offspringFeasible & parentFeasible
                    replaceMask[bothFeasible] = g_old[bothFeasible] >= g_new[bothFeasible]

                # Replace individuals in the population based on feasibility/CV first, then aggregation
                pop.replace(P[replaceMask], offspring)
            self.update(pop, completed=True)

        # Return the final result
        return self.finalize()
