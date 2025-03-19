# Genetic Algorithm <Single>

from typing import Optional

from ..algorithmABC import Algorithm, Verbose, Result
from ..population import Population
from ..utility_functions.operation_GA import operationGA
from ..utility_functions.tournament_selection import tournamentSelection

class GA(Algorithm):
    '''
    Genetic Algorithm <single> <real>/<mix>
    -------------------------------
    This class implements a single-objective genetic algorithm for optimization.
    
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
    type = "EA"
    
    def __init__(self, nPop: int = 50,
                 proC: float = 1, disC: float = 20, proM: float = 1, disM: float = 20,
                 maxIterTimes: int = 1000,
                 maxFEs: int = 50000,
                 maxTolerateTimes: Optional[int] = None, tolerate: float = 1e-6,
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = False, saveFlag = True):
        '''
        Initialize the genetic algorithm with user-defined parameters.
        
        :param nPop: Population size.
        :param proC: Crossover probability.
        :param disC: Crossover distribution index.
        :param proM: Mutation probability.
        :param disM: Mutation distribution index.
        :param maxIterTimes: Maximum number of iterations.
        :param maxFEs: Maximum number of function evaluations.
        :param maxTolerateTimes: Maximum number of tolerated iterations without improvement.
        :param tolerate: Tolerance for improvement.
        :param verboseFlag: Flag to enable verbose output.
        :param verboseFreq: Frequency of verbose output.
        :param logFlag: Flag to enable logging.
        :param saveFlag: Flag to enable saving results.
        '''
        
        super().__init__(maxFEs = maxFEs, maxIterTimes = maxIterTimes, 
                         maxTolerateTimes = maxTolerateTimes, tolerate = tolerate,
                         verboseFlag = verboseFlag, verboseFreq = verboseFreq, logFlag = logFlag, saveFlag = saveFlag)
        
        # Set user-defined parameters
        self.setPara('proC', proC)
        self.setPara('disC', disC)
        self.setPara('proM', proM)
        self.setPara('disM', disM)
        self.setPara('nPop', nPop)
        
    #--------------------Public Functions---------------------#
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem):
        '''
        Execute the genetic algorithm on the specified problem.

        :param problem: An instance of a class derived from ProblemABC.
                        This object defines the optimization problem, including
                        the number of inputs (nInput), number of outputs (nOutput),
                        upper bounds (ub), lower bounds (lb), and evaluation methods.
        
        :return Result: An instance of the Result class, which contains the
                        optimization results, including the best decision variables,
                        objective values, and constraint violations encountered during
                        the optimization process.
        '''
        
        # Initialization
        # Retrieve parameter values
        proC, disC, proM, disM = self.getParaVal('proC', 'disC', 'proM', 'disM')
        nPop = self.getParaVal('nPop')
        
        # Set the problem to solve
        self.setProblem(problem)
        
        # Initialize termination conditions
        self.FEs = 0; self.iters = 0; self.tolerateTimes = 0
        
        # Generate initial population
        pop = self.initialize(nPop)
        
        # Record initial population state
        self.record(pop)
        
        # Iterative process
        while self.checkTermination():
            # Select mating pool using tournament selection
            matingPool = tournamentSelection(pop, 2, len(pop), pop.objs, pop.cons)
            
            # Generate offspring using genetic operations
            offspring = operationGA(matingPool, problem.ub, problem.lb, proC, disC, proM, disM)
            
            # Evaluate the offspring
            self.evaluate(offspring)
            
            # Merge offspring with current population
            pop = pop.merge(offspring)
            
            # Select the best individuals to form the new population
            pop = pop.getBest(nPop)
            
            # Record the current state of the population
            self.record(pop)
        
        # Return the final result
        return self.result