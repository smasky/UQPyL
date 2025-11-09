# Non-dominated Sorting Genetic Algorithm II (NSGA-II) <Multi>
import numpy as np
from typing import Optional

from ..base import AlgorithmABC, Verbose
from ..population import Population
from ..util import NDSort, crowdingDist, tourSelect, gaOperator

import time

class NSGAII(AlgorithmABC):
    '''
    Non-dominated Sorting Genetic Algorithm II <Multi>
    ------------------------------------------------
        
    Methods:
        run: Run the NSGA-II algorithm.
        
    References:
        [1] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, 2002.
    '''
    
    name = "NSGAII"
    alg_type = "MOEA"
    
    def __init__(self, proC: float=1.0, disC: float=20.0, proM: float=1.0, disM: float=20.0,
                 nPop: int =50,
                 maxFEs: int = 50000, 
                 maxIters: int = 1000, 
                 maxTolerates = None, tolerate=1e-6, 
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = True, saveFlag: bool = True):
        '''
        Initialize the NSGA-II algorithm with user-defined parameters.
        
        :param proC: Crossover probability.
        :param disC: Crossover distribution index.
        :param proM: Mutation probability.
        :param disM: Mutation distribution index.
        :param nPop: Population size.
        :param maxFEs: Maximum number of function evaluations.
        :param maxIterTimes: Maximum number of iterations.
        :param maxTolerateTimes: Maximum number of tolerated iterations without improvement.
        :param tolerate: Tolerance for improvement.
        :param verbose: Flag to enable verbose output.
        :param verboseFreq: Frequency of verbose output.
        :param logFlag: Flag to enable logging.
        :param saveFlag: Flag to enable saving results.
        '''
        
        super().__init__(maxFEs, maxIters, maxTolerates, tolerate, 
                         verboseFlag, verboseFreq, logFlag, saveFlag)
        
        # Set user-defined parameters
        self.setParaVal('proC', proC)
        self.setParaVal('disC', disC)
        self.setParaVal('proM', proM)
        self.setParaVal('disM', disM)
        self.setParaVal('nPop', nPop)
        
    #-------------------------Public Functions------------------------#
    @Verbose.run
    def run(self, problem, seed: Optional[int] = None):
        '''
        Execute the NSGA-II algorithm on the specified problem.

        :param problem: An instance of a class derived from ProblemABC.
                        This object defines the optimization problem, including
                        the number of inputs (nInput), number of outputs (nOutput),
                        upper bounds (ub), lower bounds (lb), and evaluation methods.
        
        :return Result: An instance of the Result class, which contains the
                        optimization results, including the best decision variables,
                        objective values, and constraint violations encountered during
                        the optimization process.
        '''
        # setup algorithm
        self.setup(problem, seed)
        
        # Parameter Setting
        proC, disC, proM, disM = self.getParaVal('proC', 'disC', 'proM', 'disM')
        nPop = self.getParaVal('nPop')
        
        # Generate initial population
        pop = self.initPop(nPop)
        
        # Perform environmental selection
        _, frontNo, CrowdDis = self.environmentalSelection(pop.decs, pop.objs, pop.cons, pop.conWgt, nPop)
        
        # Iterative process
        while self.checkTermination(pop):
            # Select mating pool using tournament selection
            matingIdx = tourSelect(2, len(pop), frontNo, -CrowdDis)
            matingPool = pop[matingIdx]
          
            # Generate offspring using genetic operations
            offspringDecs = gaOperator(matingPool.decs, problem.ub, problem.lb, proC, disC, proM, disM)
            offspring = Population(offspringDecs)
         
            # Evaluate the offspring
            self.evaluate(offspring)
            
            # Merge offspring with current population
            pop.merge(offspring)
        
            # Perform environmental selection
            nextIdx, frontNo, CrowdDis = self.environmentalSelection(pop.decs, pop.objs, pop.cons, pop.conWgt, nPop)
            pop = pop[nextIdx]; pop.frontNo = frontNo; pop.crowdDis = CrowdDis

        # Return the final result
        return self.result
    
    #-------------------------Private Functions--------------------------#
    def environmentalSelection(self, popDecs, popObjs, popCons = None, conWgt = None, n = None):
        '''
        Perform environmental selection to choose the next generation.

        :param pop: Current population.
        :param n: Number of individuals to select.
        
        :return: The next population, front numbers, and crowding distances.
        '''
       
        # Non-dominated sorting
        frontNo, maxFNo = NDSort(popObjs, popCons, n)
     
        # Determine the next population
        nextIdx = frontNo < maxFNo
        
        # Handle the last front
        mask_last = (frontNo == maxFNo)
        if np.any(mask_last):
            crowdDis = np.zeros(popDecs.shape[0])
            crowdDis[mask_last] = crowdingDist(popObjs[mask_last], np.ones(np.sum(mask_last)))
        else:
            crowdDis = np.zeros(popDecs.shape[0])
        
        numSelected = n - np.sum(nextIdx)
        if numSelected > 0:
            last_indices = np.flatnonzero(mask_last)
            rank = np.argpartition(-crowdDis[last_indices], numSelected - 1)[:numSelected]
            nextIdx[last_indices[rank]] = True
        
        # Form the next population
        nextFrontNo = frontNo[nextIdx]
        nextCrowdDis = crowdDis[nextIdx]
        
        return nextIdx, nextFrontNo, nextCrowdDis