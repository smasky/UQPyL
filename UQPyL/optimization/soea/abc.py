# Artificial Bee Colony Algorithm <Single>
import numpy as np

from typing import Optional

from ..base import AlgorithmABC
from ..population import Population
from ..core.constraint import betterMask

class ABC(AlgorithmABC):
    """
    Single-objective artificial bee colony algorithm.

    Examples:
        >>> abc = ABC(nPop=50, maxFEs=5000)
        >>> res = abc.run(problem, seed=1234)
        >>> print(res.bestObjs)

    References:
        [1] D. Karaboga, An idea based on honey bee swarm for numerical optimization,
            Technical Report-TR06, Erciyes University, 2005.
    """
    
    name = "ABC"
    alg_type = "EA"
    
    def __init__(self, employedRate: float = 0.3,  limit: int = 50,
                 nPop: int = 50, 
                 maxFEs: int = 50000, 
                 maxIters: int = 1000, 
                 maxTolerates = 1000, tolerate = 1e-6, 
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = False, saveFlag: bool = True,
                 saveFreq: int = 100):
        """
        Initialize the algorithm.

        :param employedRate: Fraction of employed bees.
        :param limit: Abandonment limit.
        :param nPop: Population size.
        :param maxFEs: Maximum number of function evaluations.
        :param maxIters: Maximum number of iterations.
        :param maxTolerates: Maximum tolerated non-improving iterations.
        :param tolerate: Improvement tolerance.
        :param verboseFlag: Whether to print terminal output.
        :param verboseFreq: Summary output frequency.
        :param logFlag: Whether to save full text logs.
        :param saveFlag: Whether to save sqlite results.
        :param saveFreq: Snapshot save frequency.
        """
        
        super().__init__(maxFEs, maxIters, maxTolerates, tolerate, 
                         verboseFlag, verboseFreq, logFlag, saveFlag, saveFreq)
        
        # Set user-defined parameters
        self.set('employedRate', employedRate)
        self.set('limit', limit)
        self.set('nPop', nPop)
    
    def run(self, problem, seed: Optional[int] = None):
        """
        Run the algorithm on the given problem.

        :param problem: Problem instance.
        :param seed: Random seed.
        :return OptResult: Final optimization result.
        """
        # setup algorithm
        self.setup(problem, seed)
        
        # Parameter Setting
        employedRate, limit = self.get('employedRate', 'limit')
        nPop = self.get('nPop')
        
        # Generate initial population
        pop = self.initPop(nPop)
        self.update(pop)
            
        beeType = np.zeros(nPop, dtype=np.int32)
        limitCount = np.zeros(nPop)

        # Iterative process
        while self.checkTermination(pop):
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
            self.update(pop)
            
        # Return the final result
        return self.finalize()
            
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
            
            onlookerBees.decs = self.rng.random((n, d)) * (self.problem.ub - self.problem.lb) + self.problem.lb
            
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
        
        globalIdx = self.rng.choice(len(employedBees), len(unemployedBees), p=p)

        idx = np.arange(len(pop))
        while True:
            randIdx = self.rng.permutation(idx)
            if np.all(randIdx[beeType == 0] != idx[beeType == 1][globalIdx]):
                break
        
        rnd = self.rng.random((len(unemployedBees), d)) * 2 - 1
        
        popDecs = pop.decs
        employedDecs = employedBees.decs
        newDecs = employedDecs[globalIdx] + (employedDecs[globalIdx] - popDecs[randIdx[beeType == 0]]) * rnd
        
        newBees = Population(decs=newDecs)
        newBees.clip(self.problem.lb, self.problem.ub)
        
        self.evaluate(newBees)
        
        pop.replace(beeType == 0, newBees)
        
        replaceIdx = np.where(
            betterMask(newBees.objs, newBees.cons, employedBees[globalIdx].objs, employedBees[globalIdx].cons, pop.conWgt)
        )[0]
        limitCount[unemployedType[replaceIdx]] = 0
        beeType[unemployedType[replaceIdx]] = 1
        limitCount[employedType[globalIdx][replaceIdx]] = 0
        beeType[employedType[globalIdx][replaceIdx]] = 0
        
        updateIdx = np.where(
            ~betterMask(newBees.objs, newBees.cons, employedBees[globalIdx].objs, employedBees[globalIdx].cons, pop.conWgt)
        )[0]
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
            randIdx = self.rng.permutation(idx)
            if np.all(randIdx[employedBeesType] != idx[employedBeesType]):
                break
            
        rnd = self.rng.random((nEmployBees, D)) * 2 - 1
        
        popDecs = pop.decs
        newDecs = popDecs[employedBeesType] + (popDecs[randIdx[employedBeesType]] - popDecs[employedBeesType]) * rnd
        
        newBees = Population(decs=newDecs)
        newBees.clip(self.problem.lb, self.problem.ub)
        
        self.evaluate(newBees)
        
        better = betterMask(newBees.objs, newBees.cons, pop[employedBeesType].objs, pop[employedBeesType].cons, pop.conWgt)
        countIdx = np.where(~better)[0]
        limitCount[employedBeesType[countIdx]] += 1
        
        updateIdx = np.where(better)[0]
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
