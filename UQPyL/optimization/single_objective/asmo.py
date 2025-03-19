# Adaptive Surrogate Modelling-based Optimization <Single> <Surrogate>

from .sce_ua import SCE_UA
from ..algorithmABC import Algorithm, Population, Verbose
from ...problems import Problem
from ...surrogates import Surrogate
from ...surrogates.kriging import KRG
from ...utility.scalers import StandardScaler

class ASMO(Algorithm):
    '''
    Adaptive Surrogate Modelling-based Optimization <Single> <Surrogate>
    ----------------------------------------------
    This class implements an adaptive surrogate modeling-based optimization algorithm for single-objective problems.
    
    Attributes:
        problem: Problem
            The problem to solve, which includes attributes like n_input, ub, lb, and evaluate.
        surrogate: Surrogate
            The surrogate model to use for optimization.
        n_init: int, default=50
            Number of initial samples for surrogate modeling.
    
    Methods:
        run(problem, xInit=None, yInit=None, oneStep=False):
            Executes the ASMO algorithm on a given problem.
            - problem: Problem
                The problem to solve.
            - xInit: Optional initial decision variables.
            - yInit: Optional initial objective values.
            - oneStep: If True, the algorithm performs only one iteration.
    '''
    
    name = "ASMO"
    type = "EA"
    
    def __init__(self, nInit: int = 50, 
                 surrogate: Surrogate = None,
                 optimizer: Algorithm = None,
                 maxFEs: int = 1000,
                 maxTolerateTimes: int = 100,
                 verboseFlag: bool = True, verboseFreq: int = 1, logFlag: bool = False, saveFlag = True):
        '''
        Initialize the ASMO algorithm with user-defined parameters.
        
        :param nInit: Number of initial samples for surrogate modeling.
        :param surrogate: Surrogate model to use. Defaults to Kriging if None.
        :param optimizer: Optimizer to use. Defaults to SCE_UA if None.
        :param maxFEs: Maximum number of function evaluations.
        :param maxTolerateTimes: Maximum number of tolerated iterations without improvement.
        :param verbose: Flag to enable verbose output.
        :param verboseFreq: Frequency of verbose output.
        :param logFlag: Flag to enable logging.
        :param saveFlag: Flag to enable saving results.
        '''
        
        super().__init__(maxFEs = maxFEs, maxTolerateTimes = maxTolerateTimes, 
                         verboseFlag = verboseFlag, verboseFreq = verboseFreq, logFlag = logFlag, saveFlag = saveFlag)
        
        self.setPara('nInit', nInit)
        
        if surrogate is None:
            # Default surrogate model is Kriging with standard scaling
            scaler = (StandardScaler(0, 1), StandardScaler(0, 1))
            surrogate = KRG(scalers=scaler)
            
        self.surrogate = surrogate
        
        if optimizer is None:
            # Default optimizer is SCE_UA
            optimizer = SCE_UA(maxFEs=5000, verbose=False, saveFlag=False, logFlag=False)
        
        optimizer.verboseFlag, optimizer.logFlag, optimizer.saveFlag = False, False, False
        self.optimizer = optimizer
        self.optimizer.verbose = False
        
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem, xInit = None, yInit = None, oneStep = False):
        '''
        Main procedure to execute the ASMO algorithm on the specified problem.

        :param problem: An instance of a class derived from Problem.
                        This object defines the optimization problem, including
                        the number of inputs (nInput), upper bounds (ub), lower bounds (lb), and evaluation methods.
        :param xInit: Optional initial decision variables.
        :param yInit: Optional initial objective values.
        :param oneStep: If True, the algorithm performs only one iteration.
        
        :return Result: An instance of the Result class, which contains the
                        optimization results, including the best decision variables,
                        objective values, and constraint violations encountered during
                        the optimization process.
        '''
        
        # Initialization
        nInit = self.getParaVal('nInit')
        
        # Set the problem to solve
        self.problem = problem
        
        # Define a subproblem using the surrogate model
        subProblem = Problem(objFunc = self.surrogate.predict, nInput = problem.nInput, nOutput = 1, ub = problem.ub, lb = problem.lb, varType = problem.varType, varSet = problem.varSet)
        
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
        
        # Iterative process
        while self.checkTermination():
            
            # Build surrogate model
            self.surrogate.fit(pop.decs, pop.objs)
            
            # Run optimizer on the surrogate model
            res = self.optimizer.run(subProblem)
            
            # Evaluate the offspring
            offSpring = Population(decs=res.bestDecs)
            self.evaluate(offSpring)
            
            # Merge offspring with current population
            pop.add(offSpring)
            
            # Record the current state of the population
            self.record(pop)
            
            if oneStep:
                break
                    
        return self.result
            
        
            
            
