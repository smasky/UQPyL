# Cooperation search algorithm <Single>
import numpy as np

from typing import Optional

from ..base import AlgorithmABC
from ..population import Population
from ..core.constraint import betterMask

class CSA(AlgorithmABC):
    """
    Single-objective cooperative search algorithm.

    Examples:
        >>> csa = CSA(nPop=25, maxFEs=5000)
        >>> res = csa.run(problem, seed=1234)
        >>> print(res.bestObjs)

    References:
        [1] Z. Feng, W. Niu, S. Wang, J. Zhou, and Y. Cheng, Cooperation search algorithm:
            a novel metaheuristic evolutionary intelligence algorithm for numerical optimization
            and engineering optimization problems, Applied Soft Computing, vol. 98, 2021.
    """
    
    name = "CSA"
    alg_type = "EA" 
    
    def __init__(self, alpha: float = 0.10, beta: float = 0.15, M: int = 3,
                 nPop: int = 25,
                 maxIters: int=  1000,
                 maxFEs: int = 50000,
                 maxTolerates: int = 1000, tolerate: float = 1e-6, 
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = False, saveFlag: bool=True,
                 saveFreq: int = 100):
        """
        Initialize the algorithm.

        :param alpha: Team communication coefficient.
        :param beta: Reflective learning coefficient.
        :param M: Number of global best solutions to maintain.
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
        
        super().__init__(maxFEs = maxFEs, maxIters = maxIters, 
                         maxTolerates = maxTolerates, tolerate = tolerate, 
                         verboseFlag = verboseFlag, verboseFreq = verboseFreq, logFlag = logFlag, saveFlag = saveFlag,
                         saveFreq = saveFreq)
        
        # Set user-defined parameters
        self.set('alpha', alpha)
        self.set('beta', beta)
        self.set('M', M)
        self.set('nPop', nPop)
           
    #------------------Public Function------------------#
    def run(self, problem, seed: Optional[int] = None):
        """
        Run the algorithm on the given problem.

        :param problem: Problem instance.
        :param seed: Random seed.
        :return OptResult: Final optimization result.
        """
        # setup algorithm
        self.setup(problem, seed)
        
        # Retrieve parameter values
        alpha, beta, M = self.get('alpha', 'beta', 'M')
        nPop = self.get('nPop')
        
        # Generate initial population
        pop = self.initPop(nPop)
        self.update(pop)
        
        # Initial personal best and global best
        pBest = pop.copy()  # Personal Best
        gBest = pBest[pBest.argsort()[:M]]  # Global Best

        while self.checkTermination(pop):
            # Team communication operator
            uPopDecs = self._teamCommunicationOperator(pop.decs, pBest.decs, gBest.decs, alpha, beta)
            uPop = Population(uPopDecs)
            # Reflective learning operator 
            vPopDecs = self._reflectiveLearningOperator(uPop.decs)
            vPop = Population(vPopDecs)
            
            # Internal competition operator
            self.evaluate(uPop)
            self.evaluate(vPop)
            
            chooseU = betterMask(uPop.objs, uPop.cons, vPop.objs, vPop.cons, pop.conWgt).reshape(-1, 1)
            pop = Population(
                decs=np.where(chooseU, uPop.decs, vPop.decs),
                objs=np.where(chooseU, uPop.objs, vPop.objs),
                cons=None if uPop.cons is None and vPop.cons is None else np.where(chooseU, uPop.cons, vPop.cons),
                conWgt=pop.conWgt,
            )

            # Update personal best and global best
            tmp = pop[pop.argsort()[:M]]
            choosePop = betterMask(pop.objs, pop.cons, pBest.objs, pBest.cons, pop.conWgt).reshape(-1, 1)
            pBest = Population(
                decs=np.where(choosePop, pop.decs, pBest.decs),
                objs=np.where(choosePop, pop.objs, pBest.objs),
                cons=None if pop.cons is None and pBest.cons is None else np.where(choosePop, pop.cons, pBest.cons),
                conWgt=pop.conWgt,
            )
           
            gBest.add(tmp)
            gBest = gBest[gBest.argsort()[:M]]
            self.update(pop)
            
        return self.finalize()
    
    def _reflectiveLearningOperator(self, popDecs):
        """
        Apply the reflective learning operator to the population.

        :param pop: Current population of solutions.
        
        :return: Updated population after applying reflective learning.
        """
        
        N, D = popDecs.shape
        
        c = (self.problem.ub + self.problem.lb) / 2
        
        c_n = np.repeat(c, N, axis=0)
        lb_n = np.repeat(self.problem.lb, N, axis=0)
        ub_n = np.repeat(self.problem.ub, N, axis=0)
        fai_1 = self.problem.ub + self.problem.lb - popDecs
        
        gailv = np.abs(popDecs - c) / (self.problem.ub - self.problem.lb)
        # Calculate r
        t1 = self.rng.random((N, D)) * np.abs(c - fai_1) + np.where(c_n > fai_1, fai_1, c_n)
        t2 = self.rng.random((N, D)) * np.abs(fai_1 - self.problem.lb) + np.where(fai_1 > lb_n, lb_n, fai_1)
        seed = self.rng.random((N, D))
        r = np.where(gailv < seed, t1, t2)
        
        # Calculate p
        t3 = self.rng.random((N, D)) * np.abs(fai_1 - c) + np.where(c_n > fai_1, fai_1, c_n)
        t4 = self.rng.random((N, D)) * np.abs(self.problem.ub - fai_1) + np.where(fai_1 > ub_n, ub_n, fai_1)
        seed = self.rng.random((N, D))
        p = np.where(gailv < seed, t3, t4)
        
        vPopDecs = np.where(popDecs >= c_n, r, p)
        np.clip(vPopDecs, self.problem.lb, self.problem.ub, out=vPopDecs)
              
        return vPopDecs
    
    def _teamCommunicationOperator(self, popDecs, pBestDecs, gBestDecs, alpha, beta):
        """
        Apply the team communication operator to the population.

        :param pop: Current population of solutions.
        :param pBest: Personal best solutions.
        :param gBest: Global best solutions.
        :param alpha: Control parameter for team communication.
        :param beta: Control parameter for reflective learning.
        
        :return: Updated population after applying team communication.
        """
                
        N, D = popDecs.shape
        
        M, _ = gBestDecs.shape
        
        idx = self.rng.integers(0, M, (N, D))
        A = np.log(1.0 / self.rng.random((N, D))) * (gBestDecs[idx, np.arange(D)] - popDecs)
        
        B = alpha * self.rng.random((N, D)) * (np.mean(gBestDecs, axis=0) - popDecs)
        
        C = beta * self.rng.random((N, D)) * (np.mean(pBestDecs, axis=0) - popDecs)
        
        uPopDecs = popDecs + A + B + C
        
        np.clip(uPopDecs, self.problem.lb, self.problem.ub, out=uPopDecs)
        
        return uPopDecs

    def Phi(self, num1, num2):
        """
        Calculate a value based on two numbers using a random factor.

        :param num1: First number.
        :param num2: Second number.
        
        :return: Calculated value.
        """
        if num1 < num2:
            o = num1 + self.rng.random(1) * abs(num1 - num2)
        else:
            o = num2 + self.rng.random(1) * abs(num1 - num2)
        return o
