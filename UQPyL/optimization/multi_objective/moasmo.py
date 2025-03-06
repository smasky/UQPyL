### Multi-Objective Adaptive Surrogate Modelling-based Optimization
import numpy as np
from scipy.spatial.distance import cdist

from .nsga_ii import NSGAII
from ..algorithmABC import Algorithm, Population, Verbose
from ..utility_functions import NDSort
from ...DoE import LHS
from ...problems import Problem
from ...surrogates import MultiSurrogates
from ...surrogates.rbf.radial_basis_function import RBF

class MOASMO(Algorithm):
    '''
    Multi-Objective Adaptive Surrogate Modelling-based Optimization <Multi-objective> <Surrogate>
    -----------------------------------------------------------------
    This class implements a multi-objective optimization algorithm using adaptive surrogate models.
    
    Methods:
        run()
            Executes the optimization process.
    
    References:
        [1] W. Gong et al., Multiobjective adaptive surrogate modeling-based optimization for parameter estimation of large, complex geophysical models, 
                            Water Resour. Res., vol. 52, no. 3, pp. 1984–2008, Mar. 2016, doi: 10.1002/2015WR018230.
    '''
    
    name = "MOASMO"
    type = "MOEA"
    
    def __init__(self, surrogates: MultiSurrogates = None,
                 optimizer: Algorithm = None,
                 pct: float = 0.2, nInit: int = 50, nPop: int = 50, 
                 advance_infilling: bool = False,
                 maxFEs: int = 1000, 
                 maxIterTimes: int = 100,
                 maxTolerateTimes: int = None, tolerate: float = 1e-6,
                 verboseFlag: bool = True, verboseFreq: int = 1, logFlag: bool = False, saveFlag: bool = False):
        '''
        Initialize the MOASMO algorithm with user-defined parameters.
        
        :param surrogates: Surrogates - The surrogate models to use.
        :param optimizer: Algorithm - The optimization algorithm to use.
        :param pct: float - Percentage of the population for infilling.
        :param nInit: int - Number of initial samples.
        :param nPop: int - Population size for the optimizer.
        :param advance_infilling: bool - Use advanced infilling if True.
        :param maxFEs: int - Maximum number of function evaluations.
        :param maxIterTimes: int - Maximum number of iterations.
        :param maxTolerateTimes: int - Maximum number of tolerated iterations without improvement.
        :param tolerate: float - Tolerance for improvement.
        :param verbose: bool - Enable verbose output if True.
        :param verboseFreq: int - Frequency of verbose output.
        :param logFlag: bool - Enable logging if True.
        :param saveFlag: bool - Enable saving results if True.
        '''
        
        super().__init__(maxFEs, maxIterTimes, maxTolerateTimes, tolerate, 
                         verboseFlag, verboseFreq, logFlag, saveFlag)
        
        # Set user-defined parameters
        self.setPara('pct', pct)
        self.setPara('nInit', nInit)
        self.setPara('advance_infilling', advance_infilling)
        
        # Initialize surrogate models
        if surrogates is not None:
            self.surrogates = surrogates
        else:
            self.surrogates = MultiSurrogates(n_surrogates=3, models_list=[RBF(), RBF(), RBF()])
        
        # Initialize optimizer
        if optimizer is not None:
            if not isinstance(optimizer, Algorithm):
                raise ValueError("Please append the type of optimizer!")
            self.optimizer = optimizer
        else:
            self.optimizer = NSGAII(maxFEs= 10000, verboseFlag = False, saveFlag = False, logFlag = False)
        
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem, xInit = None, yInit = None):
        '''
        Execute the MOASMO algorithm on the specified problem.

        :param problem: Problem - The problem instance to solve.
        :param xInit: 2d-np.ndarray - Initial input samples.
        :param yInit: 2d-np.ndarray - Initial output samples.
        
        :return Result: An instance of the Result class, which contains the
                        optimization results, including the best decision variables,
                        objective values, and constraint violations encountered during
                        the optimization process.
        '''
        
        # Retrieve parameter values
        pct = self.getParaVal('pct')
        nInit = self.getParaVal('nInit')
        advance_infilling = self.getParaVal('advance_infilling')
        
        nInfilling = int(pct*nInit)
        
        # Initialize termination conditions
        self.FEs = 0; self.iters = 0; self.tolerateTimes = 0
        
        # Set the problem to solve
        self.problem = problem
        
        # Create a subproblem for surrogate model optimization
        subProblem = Problem(nInput = problem.nInput, nOutput = problem.nOutput, 
                             ub = problem.ub, lb = problem.lb, objFunc = self.surrogates.predict,
                             varType = problem.varType, 
                             varSet = problem.varSet, optType = problem.optType, 
                             xLabels = problem.xLabels)
        
        # Generate initial population
        if xInit is not None:
            if yInit is not None:
                pop = Population(xInit, yInit)
            else:
                pop = Population(xInit)
                self.evaluate(pop)
            
            if nInit > len(pop):
                pop.merge(self.initialize(nInit-len(pop)))
            
        else: 
            pop = self.initialize(nInit)
        
        # Iterative optimization process
        while self.checkTermination():
            
            # Build surrogate models
            self.surrogates.fit(pop.decs, pop.objs)
            
            # Run optimization on the surrogate model
            res = self.optimizer.run(subProblem)
            
            offSpring = Population(decs = res.bestDecs, objs = res.bestObjs)
            
            if advance_infilling==False:
                
                if offSpring.nPop > nInfilling:
                    bestOff = offSpring.getBest(nInfilling)
                else:
                    bestOff = offSpring
                    
            else:
                
                if offSpring.nPop > nInfilling:
                    Known_FrontNo, _ = NDSort(pop)
                    Unknown_FrontNo, _ = NDSort(offSpring)
                    
                    Known_best_Y = pop.objs[np.where(Known_FrontNo==1)]
                    Unknown_best_Y = offSpring.objs[np.where(Unknown_FrontNo==1)]
                    Unknown_best_X = offSpring.decs[np.where(Unknown_FrontNo==1)]
                    
                    added_points_Y = []
                    added_points_X = []
                    
                    for _ in range(nInfilling):
                        
                        if len(added_points_Y)==0:
                            distances = cdist(Unknown_best_Y, Known_best_Y)
                        else:
                            distances = cdist(Unknown_best_Y, np.append(Known_best_Y, added_points_Y, axis=0))

                        max_distance_index = np.argmax(np.min(distances, axis=1))
                        
                        added_point = Unknown_best_Y[max_distance_index]
                        added_points_Y.append(added_point)
                        added_points_X.append(Unknown_best_X[max_distance_index])
                        Known_best_Y = np.append(Known_best_Y, [added_point], axis=0)
                        
                        Unknown_best_Y = np.delete(Unknown_best_Y, max_distance_index, axis=0)
                        Unknown_best_X = np.delete(Unknown_best_X, max_distance_index, axis=0)
                    
                    BestX = np.copy(np.array(added_points_X))
                    # BestY = np.copy(np.array(added_points_Y))
                    bestOff = Population(decs = BestX)
            
            # Evaluate the selected offspring
            self.evaluate(bestOff)
            
            pop.add(bestOff)
            
            self.record(pop)
                
        return self.result
          
        
                
        
            
        
            
            
        
        
        