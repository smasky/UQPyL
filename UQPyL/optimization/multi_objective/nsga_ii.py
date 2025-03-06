# Non-dominated Sorting Genetic Algorithm II (NSGA-II) <Multi>
import numpy as np

from ..utility_functions import NDSort, crowdingDistance, tournamentSelection
from ..utility_functions.operation_GA import operationGA

from ..algorithmABC import Algorithm
from ..population import Population
from ...utility import Verbose

class NSGAII(Algorithm):
    '''
    Non-dominated Sorting Genetic Algorithm II <Multi>
    ------------------------------------------------
    This class implements the NSGA-II algorithm for multi-objective optimization.
    
    Methods:
        run: Run the NSGA-II algorithm.
        
    References:
        [1] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, 2002.
    '''
    
    name = "NSGAII"
    type = "MOEA"
    
    def __init__(self, proC: float=1.0, disC: float=20.0, proM: float=1.0, disM: float=20.0,
                 nPop: int =50,
                 maxFEs: int = 50000, 
                 maxIterTimes: int = 1000, 
                 maxTolerateTimes = None, tolerate=1e-6, 
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
        
        super().__init__(maxFEs, maxIterTimes, maxTolerateTimes, tolerate, 
                         verboseFlag, verboseFreq, logFlag, saveFlag)
        
        # Set user-defined parameters
        self.setPara('proC', proC)
        self.setPara('disC', disC)
        self.setPara('proM', proM)
        self.setPara('disM', disM)
        self.setPara('nPop', nPop)
        
    #-------------------------Public Functions------------------------#
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem):
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
        
        # Parameter Setting
        proC, disC, proM, disM = self.getParaVal('proC', 'disC', 'proM', 'disM')
        nPop = self.getParaVal('nPop')
        
        # Set the problem to solve
        self.setProblem(problem)
        
        # Initialize termination conditions
        self.FEs = 0; self.iters = 0
        
        # Generate initial population
        pop = self.initialize(nPop)
        
        # Perform environmental selection
        _, frontNo, CrowdDis = self.environmentalSelection(pop, nPop)
        
        # Iterative process
        while self.checkTermination():
            # Select mating pool using tournament selection
            matingPool = tournamentSelection(pop, 2, len(pop), frontNo, -CrowdDis)
            
            # Generate offspring using genetic operations
            offspring = operationGA(matingPool, problem.ub, problem.lb, proC, disC, proM, disM)
            
            # Evaluate the offspring
            self.evaluate(offspring)
            
            # Merge offspring with current population
            pop.merge(offspring)
            
            # Perform environmental selection
            pop, frontNo, CrowdDis = self.environmentalSelection(pop, nPop)
            
            # Record the current state of the population
            self.record(pop)
            
        # Return the final result
        return self.result
    
    #-------------------------Private Functions--------------------------#
    def environmentalSelection(self, pop, n):
        '''
        Perform environmental selection to choose the next generation.

        :param pop: Current population.
        :param n: Number of individuals to select.
        
        :return: The next population, front numbers, and crowding distances.
        '''
       
        # Non-dominated sorting
        frontNo, maxFNo = NDSort(pop, n)
        
        # Determine the next population
        next = frontNo < maxFNo
        
        # Calculate crowding distance
        crowdDis = crowdingDistance(pop, frontNo)
        
        # Handle the last front
        last = np.where(frontNo == maxFNo)[0]
        rank = np.argsort(-crowdDis[last])
        numSelected = n - np.sum(next)
        next[last[rank[:numSelected]]] = True
        
        # Form the next population
        nextPop = pop[next]
        nextFrontNo = frontNo[next]
        nextCrowdDis = np.copy(crowdDis[next])
        
        return nextPop, nextFrontNo, nextCrowdDis