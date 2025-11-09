# Efficient global optimization
import numpy as np
from scipy.stats import norm

from typing import Optional

from .ga import GA
from ..base import AlgorithmABC, Verbose
from ..population import Population

from ...problem import Problem
from ...surrogate.kriging import KRG
from ...util.scaler import StandardScaler

class EGO(AlgorithmABC):
    """
    Efficient Global Optimization (EGO) Algorithm
    ---------------------------------------------
    This class implements the EGO algorithm, which is used for single-objective optimization
    by building a surrogate model to approximate the objective function and iteratively
    improving the solution.

    Methods:
        run(problem, xInit=None, yInit=None):
            Executes the EGO algorithm on a given problem.
            - problem: Problem
                The problem to solve, which includes attributes like nInput, ub, lb, and evaluate.
            - xInit: np.ndarray, optional
                Initial decision variables.
            - yInit: np.ndarray, optional
                Initial objective values corresponding to xInit.

    References:
        [1] Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. Journal of Global Optimization, 13(4), 455-492.
    """
    
    name = "EGO"
    type = "EA" 
    
    def __init__(self, nInit: int = 50,
                 maxFEs: int = 1000,
                 maxIters: int = 1000,
                 maxTolerates: int = None,
                 verboseFlag: bool = True, verboseFreq: int = 1, logFlag: bool = False, saveFlag = False):
        
        """
        Initialize the EGO algorithm with user-defined parameters.

        :param nInit: Number of initial samples.
        :param maxFEs: Maximum number of function evaluations.
        :param maxTolerateTimes: Maximum number of tolerated iterations without improvement.
        :param verboseFlag: Flag to enable verbose output.
        :param verboseFreq: Frequency of verbose output.
        :param logFlag: Flag to enable logging.
        :param saveFlag: Flag to enable saving results.
        """      
        super().__init__(maxFEs = maxFEs, maxIters = maxIters, maxTolerates = maxTolerates, 
                            verboseFlag = verboseFlag, verboseFreq = verboseFreq, 
                            logFlag = logFlag, saveFlag = saveFlag)
        
        self.setParaVal('nInit', nInit)

        # Initialize the scaler and surrogate model
        scalers = (StandardScaler(0, 1), StandardScaler(0, 1))
        self.surrogate = KRG(scalers = scalers)
        
        # Initialize the optimizer (Genetic Algorithm)
        optimizer = GA(maxFEs = 10000, verboseFlag = False, saveFlag = False, logFlag = False)
        self.optimizer = optimizer
        
    @Verbose.run
    def run(self, problem, xInit = None, yInit = None, seed: Optional[int] = None):
        """
        Execute the EGO algorithm on the specified problem.

        :param problem: An instance of a class derived from ProblemABC.
                        This object defines the optimization problem, including
                        the number of inputs (nInput), number of outputs (nOutput),
                        upper bounds (ub), lower bounds (lb), and evaluation methods.
        :param xInit: np.ndarray, optional
                      Initial decision variables.
        :param yInit: np.ndarray, optional
                      Initial objective values corresponding to xInit.

        :return Result: An instance of the Result class, which contains the
                        optimization results, including the best decision variables,
                        objective values, and constraint violations encountered during
                        the optimization process.
        """
        # setup algorithm
        self.setup(problem, seed)
        
        # Initialization
        nInit = self.getParaVal('nInit')
        
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
        
        # Iterative process
        while self.checkTermination(pop):
            
            # Build surrogate model
            self.surrogate.fit(pop.decs, pop.objs)
            
            # Run optimizer on the sub-problem
            res = self.optimizer.run(subProblem)
            
            # Create offspring population
            offSpring = Population(decs = res.bestDecs)
            
            # Evaluate the offspring
            self.evaluate(offSpring)
            
            # Add offspring to the current population
            pop.add(offSpring)
            
        # Return the final result
        return self.result
    
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