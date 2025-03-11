# Efficient global optimization
import numpy as np
from scipy.stats import norm

from .ga import GA
from ..algorithmABC import Algorithm, Population, Verbose
from ...problems import Problem
from ...surrogates import Surrogate
from ...surrogates.kriging import KRG
from ...utility.scalers import StandardScaler

class EGO(Algorithm):
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
                 maxTolerateTimes: int = 100,
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
        super().__init__(maxFEs = maxFEs, maxTolerateTimes = maxTolerateTimes, 
                            verboseFlag = verboseFlag, verboseFreq = verboseFreq, 
                            logFlag = logFlag, saveFlag = saveFlag)
        
        self.setPara('nInit', nInit)

        # Initialize the scaler and surrogate model
        scaler = (StandardScaler(0, 1), StandardScaler(0, 1))
        surrogate = KRG()
        self.surrogate = surrogate
        
        # Initialize the optimizer (Genetic Algorithm)
        optimizer = GA(maxFEs = 10000, verboseFlag = False, saveFlag = False, logFlag = False)
        self.optimizer = optimizer
        
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem, xInit = None, yInit = None):
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
        
        # Initialization
        nInit = self.getParaVal('nInit')
        
        # Set the problem to solve
        self.problem = problem
        
        # Define a sub-problem for the optimizer
        subProblem = Problem(problem.nInput, 1, problem.ub, problem.lb, objFunc = self.EI, 
                             varType = problem.varType, varSet = problem.varSet)
        
        # Initialize termination conditions
        self.FEs = 0; self.iters = 0; self.tolerateTimes = 0
    
        # Generate initial population
        if xInit is not None:
            if yInit is not None:
                pop = Population(xInit, yInit)
            else:
                pop = Population(xInit)
                self.evaluate(pop)
            
            if nInit > len(pop):
                pop.merge(self.initialize(nInit - len(pop)))
            
        else:
            pop = self.initialize(nInit)
        
        # Record initial population state
        self.record(pop)
        
        # Iterative process
        while self.checkTermination():
            # Build surrogate model
            self.surrogate.fit(pop.decs, pop.objs)
            
            # Run optimizer on the sub-problem
            res = self.optimizer.run(subProblem)
            
            # Create offspring population
            offSpring = Population(decs = res.bestDec)
            
            # Evaluate the offspring
            self.evaluate(offSpring)
            
            # Add offspring to the current population
            pop.add(offSpring)
            
            # Record the current state of the population
            self.record(pop)
    
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
        
        # Calculate the standard deviation
        s = np.sqrt(mses)
        
        # Retrieve the best objective value found so far
        bestObj = self.result.bestObj
        
        # Calculate the expected improvement
        ei = -(bestObj - objs) * norm.cdf((bestObj - objs) / s) - s * norm.pdf((bestObj - objs) / s)
        
        return ei