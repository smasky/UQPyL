# Genetic Algorithm <Single>

import numpy as np
import math

from ..algorithmABC import Algorithm, Verbose, Result
from ..population import Population
from ..utility_functions.operation_GA import operationGA
from ..utility_functions.tournament_selection import tournamentSelection
class GA(Algorithm):
    '''
        Genetic Algorithm <single> <real>/<mix>
        -------------------------------
        Attributes:
        
            nPop: int, default=50
                the population size of the algorithm
            proC: float, default=1
                the probability of crossover
            disC: float, default=20
                the distribution index of crossover
            proM: float, default=1
                the probability of mutation
            disM: float, default=20
                the distribution index of mutation
                
            maxIterTimes: int, default=10000
                the maximum iteration times
            maxFEs: int, default=2000000
                the maximum function evaluations
            maxTolerateTimes: int, default=1000
                the maximum tolerate times which the best objective value does not change
            tolerate: float, default=1e-6
                the tolerate value which the best objective value does not change
        
        Methods:
            run(problem): 
                run the algorithm
                - problem: Problem
                the problem you want to solve, including the following attributes:
                
                    nInput: int
                        the input number of the problem
                    ub: 1d-np.ndarray or float
                        the upper bound of the problem
                    lb: 1d-np.ndarray or float
                        the lower bound of the problem
                    evaluate: Callable
                        the function to evaluate the input
                        
                    Optional:
                    var_type: np.array
                        the type of variables of the problem
                    var_set: list
                        the sets of discrete variables of the problem
        
        References:
            [1] D. E. Goldberg, Genetic Algorithms in Search, Optimization, and Machine Learning, 1989.
            [2] M. Mitchell, An Introduction to Genetic Algorithms, 1998.
            [3] D. Simon, Evolutionary Optimization Algorithms, 2013.
            [4] J. H. Holland, Adaptation in Natural and Artificial Systems, MIT Press, 1992.
    '''
    
    name = "GA"
    type = "EA"
    
    def __init__(self, nPop: int = 50,
                 proC: float = 1, disC: float = 20, proM: float = 1, disM: float = 20,
                 maxIterTimes: int = 1000,
                 maxFEs: int = 50000,
                 maxTolerateTimes: int = 1000, tolerate: float = 1e-6,
                 verbose: bool = True, verboseFreq: int = 10, logFlag: bool = False, saveFlag = True):
        
        super().__init__(maxFEs=maxFEs, maxIterTimes=maxIterTimes, 
                         maxTolerateTimes=maxTolerateTimes, tolerate=tolerate,
                         verbose=verbose, verboseFreq=verboseFreq, logFlag=logFlag, saveFlag=saveFlag)
        
        #user-define setting
        self.setParameters('proC', proC)
        self.setParameters('disC', disC)
        self.setParameters('proM', proM)
        self.setParameters('disM', disM)
        self.setParameters('nPop', nPop)
        
    #--------------------Public Functions---------------------#
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem):
        
        #Initialization
        #Parameter Setting
        proC, disC, proM, disM = self.getParaValue('proC', 'disC', 'proM', 'disM')
        nPop = self.getParaValue('nPop')
        
        #Problem
        self.problem = problem
        
        #Termination Condition Setting
        self.FEs = 0; self.iters = 0; self.tolerateTimes = 0
        
        #Population Generation
        pop = self.initialize(nPop)
        
        #Record
        self.record(pop) 
        
        #Iterative
        while self.checkTermination():
            
            matingPool = tournamentSelection(pop, 2, len(pop), pop.objs, pop.cons)
            
            offspring = operationGA(matingPool, problem.ub, problem.lb, proC, disC, proM, disM)
            
            self.evaluate(offspring)
            
            pop = pop.merge(offspring)
            
            pop = pop.getBest(nPop)
            
            self.record(pop)
            
        return self.result