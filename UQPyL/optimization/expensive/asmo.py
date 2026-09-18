# Adaptive Surrogate Modelling-based Optimization <Single> <Surrogate>

import numpy as np

from typing import Literal

from ..soea.sce_ua import SCE_UA
from ..base import AlgorithmABC
from ._base import SurrogateOptimization
from ..population import Population
from ...core import spawn_seed

from ...problem import Problem
from ...surrogate import SurrogateABC
from ...surrogate.kriging import KRG

class ASMO(SurrogateOptimization):
    """
    Single-objective adaptive surrogate modelling-based optimization algorithm.

    Examples:
        >>> asmo = ASMO(nInit=20, maxFEs=100)
        >>> res = asmo.run(problem, seed=1234)
        >>> print(res.bestObjs)

    References:
        [1] H. Wang, W. Duan, S. Wang, Y. Han, and X. Li, Adaptive surrogate model
            based optimization with application to aerodynamic design,
            Environmental Modelling and Software, vol. 60, pp. 33-46, 2014.
    """
    
    name = "ASMO"
    alg_type = "EA"
    
    def __init__(self, nInit: int = 50, 
                 surrogate: SurrogateABC = None,
                 optimizer: AlgorithmABC = None,
                 euclidThres: float = 1e-5,
                 maxFEs: int = 1000,
                 maxIters: int = 1000,
                 maxTolerates: int = None,
                 verboseFlag: bool = True, verboseFreq: int = 1, logFlag: bool = False, saveFlag = True,
                 saveFreq: int = 100, historyFreq: int = 10):
        """
        Initialize the algorithm.

        :param nInit: Number of initial samples.
        :param surrogate: Surrogate model.
        :param optimizer: Inner optimizer.
        :param euclidThres: Euclidean distance threshold.
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
                         verboseFlag = verboseFlag, verboseFreq = verboseFreq, logFlag = logFlag, saveFlag = saveFlag,
                         saveFreq = saveFreq, historyFreq=historyFreq)
        
        self.set('nInit', nInit)
        self.set('euclidThres', euclidThres)
        
        if surrogate is None:
            surrogate = KRG()
            
        self.surrogate = surrogate
        
        if optimizer is None:
            # Default optimizer is SCE_UA
            optimizer = SCE_UA(maxFEs = 5000, verboseFlag = False, saveFlag = False, logFlag = False)
        
        self.optimizer = optimizer
        self.optimizer.verboseFlag, self.optimizer.logFlag, self.optimizer.saveFlag = False, False, False
        
    def run(self, problem, seed = None, oneStep = False, initialPop=None):
        """
        Run the algorithm on the given problem.

        :param problem: Problem instance.
        :param seed: Random seed.
        :param oneStep: Whether to perform only one iteration.
        :param initialPop: Optional initial population or decision matrix.
        :return OptResult: Final optimization result.
        """
        # setup algorithm
        self.setup(problem, seed)
        
        # Initialization
        nInit = self.get('nInit')
        euclidThres = self.get('euclidThres')
        
        # Define a subproblem using the surrogate model
        subProblem = Problem(objFunc=self._predictUnit, nInput=problem.nInput,
                             nObj=1, ub=1.0, lb=0.0, optType="min")

        # Generate initial population
        pop = self.initPop(nInit, initialPop=initialPop)
        self.update(pop)
        
        # Iterative process
        while self.checkTermination(pop):
            
            # Build surrogate model
            self._fitSurrogate(self.surrogate, pop)
            
            # Run optimizer on the surrogate model
            res = self.optimizer.run(subProblem, seed=spawn_seed(self.rng))
            
            # Evaluate the offspring
            decs = self._novelCandidates(np.asarray(res.bestDecs), pop,
                                         tolerance=euclidThres)
            if not len(decs):
                break

            offSpring = Population(decs = decs)
            
            self.evaluate(offSpring)
            
            # Merge offspring with current population
            pop.add(offSpring)
            self.update(pop, completed=True)
            
            if oneStep:
                break
                    
        return self.finalize()

    def _predictUnit(self, X):
        return self.surrogate.predict(self.problem.canonicalize_unit(X))
