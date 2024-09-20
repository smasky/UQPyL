# Non-dominated Sorting Genetic Algorithm II (NSGA-II) <Multi>
import numpy as np
import math
from typing import Optional

from ..utility_functions import NDSort, crowdingDistance, tournamentSelection, operationGA
from ...DoE import LHS
from ..algorithmABC import Algorithm
from ..population import Population
from ...problems import ProblemABC  as Problem
from ...utility import Verbose
class NSGAII(Algorithm):
    '''
    Non-dominated Sorting Genetic Algorithm II <Multi>
    ------------------------------------------------
    Attributes:
        problem: Problem
            the problem you want to solve, including the following attributes:
            n_input: int
                the input number of the problem
            ub: 1d-np.ndarray or float
                the upper bound of the problem
            lb: 1d-np.ndarray or float
                the lower bound of the problem
            evaluate: Callable
                the function to evaluate the input
        n_samples: int, default=50
            the number of samples for each generation
        x_init: np.ndarray, default=None
            the initial input
        y_init: np.ndarray, default=None
            the initial output
        proC: float, default=1
            the probability of crossover
        disC: float, default=20
            the distribution index of crossover
        proM: float, default=1
            the probability of mutation
        disM: float, default=20
            the distribution index of mutation
        maxFEs: int, default=50000
            the maximum number of function evaluations
        maxIters: int, default=1000
            the maximum number of iterations
    Methods:
        run: run the NSGA-II
        
    References:
        [1] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, 2002.
    '''
    
    name = "NSGAII"
    type = "MOEA"
    
    def __init__(self, proC: float=1.0, disC: float=20.0, proM: float=1.0, disM: float=20.0,
                 nInit: int =50, nPop: int =50,
                 maxFEs: int = 50000, 
                 maxIterTimes: int = 1000, 
                 maxTolerateTimes=None, tolerate=1e-6, 
                 verbose=True, verboseFreq=10, logFlag=True, saveFlag=False):
        
        super().__init__(maxFEs, maxIterTimes, maxTolerateTimes, tolerate, verbose, verboseFreq, logFlag, saveFlag)
        
        self.setParameters('proC', proC)
        self.setParameters('disC', disC)
        self.setParameters('proM', proM)
        self.setParameters('disM', disM)
        self.setParameters('nInit', nInit)
        self.setParameters('nPop', nPop)
        
    #-------------------------Public Functions------------------------#
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem, xInit=None, yInit=None):
        
        #Parameter Setting
        proC, disC, proM, disM=self.getParaValue('proC', 'disC', 'proM', 'disM')
        nInit, nPop=self.getParaValue('nInit', 'nPop')
        #Problem
        self.setProblem(problem)
        #Termination Condition Setting
        self.FEs=0; self.iters=0
        #Population Generation
        if xInit is not None:
            pop = Population(xInit, yInit) if yInit is not None else Population(xInit)
            if yInit is None:
                self.evaluate(pop)
        else:
            pop = self.initialize(nInit)
            
        pop=pop.getTop(nPop)
        
        _, frontNo, CrowdDis=self.environmentalSelection(pop, nPop)
        
        while self.checkTermination():
            
            selectIdx=tournamentSelection(2, nPop, frontNo, -CrowdDis)
            
            offspring=operationGA(pop[selectIdx], self.problem.ub, self.problem.lb, proC, disC, proM, disM)
            
            self.evaluate(offspring)
            
            pop.merge(offspring)
            
            pop, frontNo, CrowdDis=self.environmentalSelection(pop, nPop)
            
            self.record(pop)
            
        return self.result
    
    #-------------------------Private Functions--------------------------#
    def environmentalSelection(self, pop, n):
       
        frontNo, maxFNo = NDSort(pop, n)
        
        next = frontNo < maxFNo
        
        crowdDis = crowdingDistance(pop, frontNo)
        
        last = np.where(frontNo == maxFNo)[0]
        rank = np.argsort(-crowdDis[last])
        numSelected = n - np.sum(next)
        next[last[rank[:numSelected]]] = True
        
        nextPop=pop[next]
        nextFrontNo=frontNo[next]
        nextCrowdDis=np.copy(crowdDis[next])
        
        return nextPop, nextFrontNo, nextCrowdDis