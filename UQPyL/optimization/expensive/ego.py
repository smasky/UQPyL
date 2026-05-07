# Efficient global optimization
import numpy as np
from scipy.stats import norm

from typing import Optional

from ..soea.ga import GA
from ..base import AlgorithmABC
from ..population import Population
from ...core import spawn_seed

from ...problem import Problem
from ...surrogate.kriging import KRG

class EGO(AlgorithmABC):
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
                 saveFreq: int = 100):
        
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
        :param saveFreq: Snapshot save frequency.
        """      
        super().__init__(maxFEs = maxFEs, maxIters = maxIters, maxTolerates = maxTolerates, 
                            verboseFlag = verboseFlag, verboseFreq = verboseFreq, 
                            logFlag = logFlag, saveFlag = saveFlag, saveFreq = saveFreq)
        
        self.set('nInit', nInit)

        self.surrogate = KRG()
        
        # Initialize the optimizer (Genetic Algorithm)
        optimizer = GA(maxFEs = 10000, verboseFlag = False, saveFlag = False, logFlag = False)
        self.optimizer = optimizer
        
    def run(self, problem, xInit = None, yInit = None, seed: Optional[int] = None):
        """
        Run the algorithm on the given problem.

        :param problem: Problem instance.
        :param xInit: Optional initial decision variables.
        :param yInit: Optional initial objective values.
        :param seed: Random seed.
        :return OptResult: Final optimization result.
        """
        # setup algorithm
        self.setup(problem, seed)
        
        # Initialization
        nInit = self.get('nInit')
        
        # Define a sub-problem for the optimizer
        subProblem = Problem(problem.nInput, 1, problem.ub, problem.lb, objFunc = self.EI, 
                             varType = problem.varType, varSet = problem.varSet, optType = "min")
        
        # Generate initial population
        if xInit is not None:
            if yInit is not None:
                pop = Population(xInit, yInit)
            else:
                pop = Population(xInit)
                self.evaluate(pop)
            
            if nInit > len(pop):
                pop.merge(self.initPop(nInit - len(pop)))
            
        else:
            pop = self.initPop(nInit)
        self.update(pop)
        
        # Iterative process
        while self.checkTermination(pop):
            
            # Build surrogate model
            self.surrogate.fit(pop.decs, pop.objs)
            
            res = self.optimizer.run(subProblem, seed=spawn_seed(self.rng))
            bestDecs = np.asarray(res.bestDecs)

            # Create offspring population
            offSpring = Population(decs=bestDecs)
            
            # Evaluate the offspring
            self.evaluate(offSpring)
            
            # Add offspring to the current population
            pop.add(offSpring)
            self.update(pop)
            
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
        
        # Predict objective values and mean squared errors using the surrogate model
        objs, mses = self.surrogate.predict(X, only_value=False)
        
        tmp, mse= self.surrogate.predict(self.result.bestDecs, only_value=False)
        
        ss = np.sqrt(mse)
        
        
        
        # Calculate the standard deviation
        s = np.sqrt(mses)
        
        # Retrieve the best objective value found so far
        bestObjs = self.result.bestObjs
        
        # Calculate the expected improvement
        ei = -(bestObjs - objs) * norm.cdf((bestObjs - objs) / s) - s * norm.pdf((bestObjs - objs) / s)
        
        e = -(bestObjs - tmp) * norm.cdf((bestObjs - tmp) / ss) - ss * norm.pdf((bestObjs - tmp) / ss)
        
        return ei
