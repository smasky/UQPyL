# Genetic Algorithm <Single>
import numpy as np

from typing import Optional

from ..base import AlgorithmABC, Verbose
from ..population import Population

from ..util.ga_operator import gaOperator
from ..util.tournament import tourSelect

class GA(AlgorithmABC):
    '''
    Genetic Algorithm <single> <real>/<mix>
    -------------------------------
    
    Methods:
        run(problem): 
            Executes the genetic algorithm on a given problem.
            - problem: Problem
                The problem to solve, which includes attributes like nInput, ub, lb, and evaluate.
    
    References:
        [1] D. E. Goldberg, Genetic Algorithms in Search, Optimization, and Machine Learning, 1989.
        [2] M. Mitchell, An Introduction to Genetic Algorithms, 1998.
        [3] D. Simon, Evolutionary Optimization Algorithms, 2013.
        [4] J. H. Holland, Adaptation in Natural and Artificial Systems, MIT Press, 1992.
    '''
    
    name = "GA"
    alg_type = "EA"
    
    def __init__(self, nPop: int = 50,
                 proC: float = 1, disC: float = 20, proM: float = 1, disM: float = 20,
                 maxFEs: int = 50000,
                 maxIters: int = 1000,
                 maxTolerates: Optional[int] = None, tolerate: float = 1e-6,
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = False, saveFlag = True):
        '''
        Initialize the genetic algorithm with user-defined parameters.
        
        
        :param nPop: Population size.
        :param proC: Crossover probability.
        :param disC: Crossover distribution index.
        :param proM: Mutation probability.
        :param disM: Mutation distribution index.
        
        :param maxFEs: Maximum number of function evaluations.
        :param maxIterTimes: Maximum number of iterations.
        :param maxTolerateTimes: Maximum number of tolerated iterations without improvement.
        :param tolerate: Tolerance for improvement.
        :param verboseFlag: Flag to enable verbose output.
        :param verboseFreq: Frequency of verbose output.
        :param logFlag: Flag to enable logging.
        :param saveFlag: Flag to enable saving results.
        
        '''
        
        super().__init__(maxFEs = maxFEs, maxIters = maxIters, 
                         maxTolerates = maxTolerates, tolerate = tolerate,
                         verboseFlag = verboseFlag, verboseFreq = verboseFreq, logFlag = logFlag, saveFlag = saveFlag)
        
        # Set user-defined parameters
        self.setParaVal('proC', proC)
        self.setParaVal('disC', disC)
        self.setParaVal('proM', proM)
        self.setParaVal('disM', disM)
        self.setParaVal('nPop', nPop)
        
    #--------------------Public Functions---------------------#
    @Verbose.run
    def run(self, problem, seed: Optional[int] = None):
        '''
        Execute the genetic algorithm on the specified problem.

        :param problem: An instance of a class derived from ProblemABC.
                        This object defines the optimization problem, including
                        the number of inputs (nInput), number of outputs (nOutput),
                        upper bounds (ub), lower bounds (lb), and evaluation methods.
                        
        :param seed: Random seed for reproducibility.
        
        :return Result: An instance of the Result class, which contains the
                        optimization results, including the best decision variables,
                        objective values, and constraint violations encountered during
                        the optimization process.
        '''
        # setup algorithm
        self.setup(problem, seed)
        
        # Retrieve parameter values
        proC, disC, proM, disM = self.getParaVal('proC', 'disC', 'proM', 'disM')
        nPop = self.getParaVal('nPop')
        
        # Generate initial population
        pop = self.initPop(nPop)
       
        # Iterative process
        while self.checkTermination(pop):
            
            # Select mating pool using tournament selection
            objs = pop.objs; cons = pop.cons * -1 if pop.cons is not None else None
            matingIdx = tourSelect(2, len(pop), objs, cons)
            matingPool = pop[matingIdx]
            
            # Generate offspring using genetic operator
            offspringDecs = gaOperator(matingPool.decs, problem.ub, problem.lb, proC, disC, proM, disM)
            offspring = Population(offspringDecs)
            
            # Evaluate the offspring
            self.evaluate(offspring)
            
            # Merge offspring with current population
            pop = pop.merge(offspring)
            
            # Select the best individuals to form the new population
            pop = pop.getBest(nPop)
                    
        # Return the final result
        return self.result