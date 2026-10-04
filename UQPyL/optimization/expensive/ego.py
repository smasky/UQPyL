# Efficient global optimization
import numpy as np
from scipy.stats import norm

from typing import Optional

from ..soea.ga import GA
from ..base import AlgorithmABC
from ._base import SurrogateOptimization
from ..population import Population
from ...core import spawn_seed

from ...problem import Problem
from ...surrogate.kriging import KRG


class EGO(SurrogateOptimization):
    """
    Single-objective efficient global optimization algorithm.

    Examples:
        >>> ego = EGO(nInit=20, maxFEs=100)
        >>> res = ego.run(problem, seed=1234)
        >>> print(res.bestObjs)

    References:
        [1] D. R. Jones, M. Schonlau, and W. J. Welch, Efficient global optimization
            of expensive black-box functions, Journal of Global Optimization,
            vol. 13, no. 4, pp. 455-492, 1998.
    """

    name = "EGO"
    alg_type = "EA"
    requiresUncertainty = True

    def __init__(
        self,
        nInit: int = 50,
        maxFEs: int = 1000,
        maxIters: int = 1000,
        maxTolerates: int = None,
        verboseFlag: bool = True,
        verboseFreq: int = 1,
        logFlag: bool = False,
        saveFlag=False,
        saveFreq: int = 100,
        historyFreq: int = 10,
        surrogate=None,
        optimizer=None,
    ):
        """
        Initialize the algorithm.

        Args:
            nInit: Number of initial samples.
            maxFEs: Maximum number of function evaluations.
            maxIters: Maximum number of iterations.
            maxTolerates: Maximum tolerated non-improving iterations.
            verboseFlag: Whether to print terminal output.
            verboseFreq: Summary output frequency.
            logFlag: Whether to save full text logs.
            saveFlag: Whether to save sqlite results.
            saveFreq: SQLite snapshot save frequency.
            historyFreq: Full in-memory snapshot interval; None keeps only the final snapshot.
            surrogate: Predictive-variance surrogate, defaulting to a fresh KRG.
            optimizer: Single-objective acquisition optimizer, defaulting to a fresh GA.
        """
        super().__init__(
            maxFEs=maxFEs,
            maxIters=maxIters,
            maxTolerates=maxTolerates,
            verboseFlag=verboseFlag,
            verboseFreq=verboseFreq,
            logFlag=logFlag,
            saveFlag=saveFlag,
            saveFreq=saveFreq,
            historyFreq=historyFreq,
        )

        self.set("nInit", nInit)

        self.surrogate = KRG() if surrogate is None else surrogate

        # Initialize the optimizer (Genetic Algorithm)
        self.optimizer = (
            GA(maxFEs=10000, verboseFlag=False, saveFlag=False, logFlag=False) if optimizer is None else optimizer
        )

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

        # Initialization
        nInit = self.get("nInit")

        # Define a sub-problem for the optimizer
        subProblem = Problem(problem.nInput, 1, 1.0, 0.0, objFunc=self.EI, optType="min")

        # Generate initial population
        pop = self.initPop(nInit, initialPop=initialPop)
        self.update(pop)

        # Iterative process
        while self.checkTermination(pop):
            # Build surrogate model
            self._fitSurrogate(self.surrogate, pop)

            res = self.optimizer.run(subProblem, seed=spawn_seed(self.rng))
            bestDecs = self._novelCandidates(np.asarray(res.bestDecs), pop)
            if not len(bestDecs):
                self.state.stopReason = "no_novel_candidates"
                break

            # Create offspring population
            offSpring = Population(decs=bestDecs)

            # Evaluate the offspring
            self.evaluate(offSpring)

            # Add offspring to the current population
            pop.add(offSpring)
            self.update(pop, completed=True)

        # Return the final result
        return self.finalize()

    def EI(self, X):
        """
        Calculate the Expected Improvement (EI) for a given set of decision variables.

        Args:
            X: Unit-space decisions with shape (n_samples, n_input).

        Returns:
            np.ndarray: Negative expected improvement, shape (n_samples, 1),
                for minimization by the inner optimizer. Uses predictive variance
                and the incumbent in internal objective direction.
        """

        unitX = self.problem.canonicalize_unit(X)
        objs, variances = self.surrogate.predict(unitX, returnVar=True)
        std = np.sqrt(np.maximum(variances, 0.0))
        improvement = self.state.bestObjs - objs
        z = np.divide(improvement, std, out=np.zeros_like(improvement), where=std > 0)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        ei = np.where(std > 0, ei, np.maximum(improvement, 0.0))
        return -ei
