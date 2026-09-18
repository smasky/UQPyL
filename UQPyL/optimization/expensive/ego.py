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
    
    def __init__(self, nInit: int = 50,
                 maxFEs: int = 1000,
                 maxIters: int = 1000,
                 maxTolerates: int = None,
                 verboseFlag: bool = True, verboseFreq: int = 1, logFlag: bool = False, saveFlag = False,
                 saveFreq: int = 100, historyFreq: int = 10):
        
        """
        Initialize the algorithm.

        :param nInit: Number of initial samples.
        :param maxFEs: Maximum number of function evaluations.
        :param maxIters: Maximum number of iterations.
        :param maxTolerates: Maximum tolerated non-improving iterations.
        :param verboseFlag: Whether to print terminal output.
        :param verboseFreq: Summary output frequency.
        :param logFlag: Whether to save full text logs.
        :param saveFlag: Whether to save sqlite results.
        :param saveFreq: SQLite snapshot save frequency.
        :param historyFreq: Full in-memory snapshot interval; None keeps only the final snapshot.
        """      
        super().__init__(maxFEs = maxFEs, maxIters = maxIters, maxTolerates = maxTolerates, 
                            verboseFlag = verboseFlag, verboseFreq = verboseFreq, 
                            logFlag = logFlag, saveFlag = saveFlag, saveFreq = saveFreq, historyFreq=historyFreq)
        
        self.set('nInit', nInit)

        self.surrogate = KRG()
        
        # Initialize the optimizer (Genetic Algorithm)
        optimizer = GA(maxFEs = 10000, verboseFlag = False, saveFlag = False, logFlag = False)
        self.optimizer = optimizer
        
    def run(self, problem, seed: Optional[int] = None, initialPop=None):
        """
        Run the algorithm on the given problem.

        :param problem: Problem instance.
        :param seed: Random seed.
        :param initialPop: Optional initial population or decision matrix.
        :return OptResult: Final optimization result.
        """
        # setup algorithm
        self.setup(problem, seed)
        
        # Initialization
        nInit = self.get('nInit')
        
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

        :param X: np.ndarray
                  Decision variables for which to calculate the EI.

        :return ei: np.ndarray
                    The expected improvement values for the given decision variables.
        """
        
        unitX = self.problem.canonicalize_unit(X)
        objs, variances = self.surrogate.predict(unitX, returnVar=True)
        std = np.sqrt(np.maximum(variances, 0.0))
        improvement = self.state.bestObjs - objs
        z = np.divide(improvement, std, out=np.zeros_like(improvement), where=std > 0)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        ei = np.where(std > 0, ei, np.maximum(improvement, 0.0))
        return -ei
