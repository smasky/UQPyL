# Artificial Bee Colony Algorithm <Single>
import numpy as np

from ..algorithmABC import Algorithm, Population
from ...utility import Verbose

class ABC(Algorithm):
    """
    Artificial Bee Colony Algorithm (ABC) <Single>
    ----------------------------------------------
    This class implements a single-objective artificial bee colony algorithm for optimization.
    
    Methods:
        run(problem): 
            Executes the ABC algorithm on a given problem.
            - problem: Problem
                The problem to solve, which includes attributes like nInput, ub, lb, and evaluate.
    
    References:
        [1] D. Karaboga, An Idea Based on Honey Bee Swarm for Numerical Optimization, 2005.
        [2] D. Karaboga and B. Basturk, A Powerful and Efficient Algorithm for Numerical Function Optimization: ABC Algorithm, 2007.
    """
    
    name = "ABC"
    type = "EA"
    
    def __init__(self, employedRate: float = 0.3,  limit: int = 50,
                 nPop: int = 50, 
                 maxFEs: int = 50000, 
                 maxIterTimes: int = 1000, 
                 maxTolerateTimes = 1000, tolerate = 1e-6, 
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the ABC algorithm with user-defined parameters.
        
        :param employedRate: The rate of employed bees in the population.
        :param limit: The limit for abandoning a food source.
        :param nPop: Population size.
        :param maxFEs: Maximum number of function evaluations.
        :param maxIterTimes: Maximum number of iterations.
        :param maxTolerateTimes: Maximum number of tolerated iterations without improvement.
        :param tolerate: Tolerance for improvement.
        :param verbose: Flag to enable verbose output.
        :param verboseFreq: Frequency of verbose output.
        :param logFlag: Flag to enable logging.
        :param saveFlag: Flag to enable saving results.
        """
        
        super().__init__(maxFEs, maxIterTimes, maxTolerateTimes, tolerate, 
                         verboseFlag, verboseFreq, logFlag, saveFlag)
        
        # Set user-defined parameters
        self.setPara('employedRate', employedRate)
        self.setPara('limit', limit)
        self.setPara('nPop', nPop)
    
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem):
        """
        Execute the ABC algorithm on the specified problem.

        :param problem: An instance of a class derived from ProblemABC.
                        This object defines the optimization problem, including
                        the number of inputs (nInput), number of outputs (nOutput),
                        upper bounds (ub), lower bounds (lb), and evaluation methods.
        
        :return Result: An instance of the Result class, which contains the
                        optimization results, including the best decision variables,
                        objective values, and constraint violations encountered during
                        the optimization process.
        """
        
        # Parameter Setting
        employedRate, limit = self.getParaVal('employedRate', 'limit')
        nPop = self.getParaVal('nPop')
        
        # Set the problem to solve
        self.setProblem(problem)
        
        # Initialize termination conditions
        self.FEs = 0; self.iters = 0; self.tolerateTimes = 0
        
        # Generate initial population
        pop = self.initialize(nPop)
            
        beeType = np.zeros(nPop, dtype=np.int32)
        limitCount = np.zeros(nPop)
        
        # Iterative process
        while self.checkTermination():
            # Set employed bees
            pop, beeType = self.setEmployedBees(beeType, pop, employedRate)
            
            # Update employed bees
            pop, limitCount = self.updateEmployedBees(pop, beeType, limitCount)
            
            # Update unemployed bees
            pop, beeType, limitCount = self.updateUnemployedBees(pop, beeType, limitCount)
            
            # Update onlooker bees
            pop, beeType, limitCount = self.updateOnlookerBees(pop, beeType, limitCount, employedRate)
            
            # Check limit times for abandonment
            beeType = self.checkLimitTimes(beeType, limitCount, limit)
            
            # Record the current state of the population
            self.record(pop)
         
        # Return the final result
        return self.result
            
    def checkLimitTimes(self, beeType: np.ndarray, limitCount: np.ndarray, limit: int):
        """
        Check if any onlooker bees have exceeded the limit and need to be abandoned.

        :param beeType: Array indicating the type of each bee.
        :param limitCount: Array counting the number of times each bee has been limited.
        :param limit: The limit for abandoning a food source.
        
        :return: Updated beeType array.
        """
        
        onlookerBees = np.where(limitCount > limit)[0]
        
        beeType[onlookerBees] = 2
        
        return beeType        
    
    def updateOnlookerBees(self, pop: Population, beeType: np.ndarray, limitCount: np.ndarray, employedRate: float):
        """
        Update the onlooker bees by generating new solutions and evaluating them.

        :param pop: Current population of solutions.
        :param beeType: Array indicating the type of each bee.
        :param limitCount: Array counting the number of times each bee has been limited.
        :param employedRate: The rate of employed bees in the population.
        
        :return: Updated population, beeType, and limitCount.
        """
        
        maxNEmployed = int(len(pop) * employedRate)
        
        if np.sum(beeType == 2) > 0:
            onlookerIdx = np.where(beeType == 2)[0]
            onlookerBees = pop[onlookerIdx]
            n, d = onlookerBees.size()
            
            onlookerBees.decs = np.random.random((n, d)) * (self.problem.ub - self.problem.lb) + self.problem.lb
            
            self.evaluate(onlookerBees)
            
            onlookerBees = onlookerBees[onlookerBees.argsort()]
            
            pop.replace(beeType == 2, onlookerBees)
            
            nEmployed = np.sum(beeType == 1)
            
            limitCount[onlookerIdx] = 0
            beeType[onlookerIdx] = 0
            
            if nEmployed < maxNEmployed:
                pop.replace(onlookerIdx, onlookerBees)
                beeType[onlookerIdx[:maxNEmployed - nEmployed]] = 1
        
        return pop, beeType, limitCount
            
    def updateUnemployedBees(self, pop: Population, beeType: np.ndarray, limitCount: np.ndarray):
        """
        Update the unemployed bees by generating new solutions based on employed bees.

        :param pop: Current population of solutions.
        :param beeType: Array indicating the type of each bee.
        :param limitCount: Array counting the number of times each bee has been limited.
        
        :return: Updated population, beeType, and limitCount.
        """
        
        n, d = pop.size()
        
        employedType = np.where(beeType == 1)[0]
        unemployedType = np.where(beeType == 0)[0]
        
        employedBees = pop[employedType]
        unemployedBees = pop[unemployedType]
        
        idx = employedBees.argsort()
        nEmployed = len(employedBees)
        p = 2 * (nEmployed + 1.0 - np.linspace(1, nEmployed, nEmployed)) / ((nEmployed + 1) * nEmployed)
        p[idx] = p / np.sum(p)
        
        globalIdx = np.random.choice(len(employedBees), len(unemployedBees), p=p)

        idx = np.arange(len(pop))
        while True:
            randIdx = np.random.permutation(idx)
            if np.all(randIdx[beeType == 0] != idx[beeType == 1][globalIdx]):
                break
        
        rnd = np.random.random((len(unemployedBees), d)) * 2 - 1
        
        popDecs = pop.decs
        employedDecs = employedBees.decs
        newDecs = employedDecs[globalIdx] + (employedDecs[globalIdx] - popDecs[randIdx[beeType == 0]]) * rnd
        
        newBees = Population(decs=newDecs)
        newBees.clip(self.problem.lb, self.problem.ub)
        
        self.evaluate(newBees)
        
        pop.replace(beeType == 0, newBees)
        
        replaceIdx = np.where(newBees.objs < employedBees[globalIdx].objs)[0]
        limitCount[unemployedType[replaceIdx]] = 0
        beeType[unemployedType[replaceIdx]] = 1
        limitCount[employedType[globalIdx][replaceIdx]] = 0
        beeType[employedType[globalIdx][replaceIdx]] = 0
        
        updateIdx = np.where(newBees.objs > employedBees[globalIdx].objs)[0]
        limitCount[employedType[globalIdx][updateIdx]] += 1
        
        return pop, beeType, limitCount
        
    def updateEmployedBees(self, pop: Population, beeType: np.ndarray, limitCount: np.ndarray):
        """
        Update the employed bees by generating new solutions and evaluating them.

        :param pop: Current population of solutions.
        :param beeType: Array indicating the type of each bee.
        :param limitCount: Array counting the number of times each bee has been limited.
        
        :return: Updated population and limitCount.
        """
        
        _, D = pop.size()
        employedBeesType = np.where(beeType == 1)[0]
        nEmployBees = np.sum(beeType == 1)
        idx = np.arange(len(pop))
        while True:
            randIdx = np.random.permutation(idx)
            if np.all(randIdx[employedBeesType] != idx[employedBeesType]):
                break
            
        rnd = np.random.random((nEmployBees, D)) * 2 - 1
        
        popDecs = pop.decs
        newDecs = popDecs[employedBeesType] + (popDecs[randIdx[employedBeesType]] - popDecs[employedBeesType]) * rnd
        
        newBees = Population(decs=newDecs)
        newBees.clip(self.problem.lb, self.problem.ub)
        
        self.evaluate(newBees)
        
        countIdx = np.where(newBees.objs >= pop[employedBeesType].objs)[0]
        limitCount[employedBeesType[countIdx]] += 1
        
        updateIdx = np.where(newBees.objs < pop[employedBeesType].objs)[0]
        pop.replace(employedBeesType[updateIdx], newBees[updateIdx])
        
        return pop, limitCount
    
    def setEmployedBees(self, beeType: np.ndarray, pop: Population, employedRate: float):
        """
        Set the employed bees in the population based on the employed rate.

        :param beeType: Array indicating the type of each bee.
        :param pop: Current population of solutions.
        :param employedRate: The rate of employed bees in the population.
        
        :return: Updated population and beeType.
        """
        
        nEmployBees = np.sum(beeType == 1)
        
        if nEmployBees == 0:
            idx = pop.argsort()
            beeType[idx[:int(len(pop) * employedRate)]] = 1
        
        return pop, beeType