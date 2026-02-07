# Cooperation search algorithm <Single>
import numpy as np

from typing import Optional

from ..base import AlgorithmABC, Verbose
from ..population import Population

class CSA(AlgorithmABC):
    """
    Cooperative Search Algorithm (CSA) <Single>
    -------------------------------------------------
    This class implements a single-objective cooperative search algorithm for optimization.
    
    References:
        [1] Z. Feng, W. Niu, and S. Liu (2021), Cooperation search algorithm: A novel metaheuristic evolutionary intelligence algorithm for numerical optimization and engineering optimization problems, Appl. Soft. Comput., vol. 98, p. 106734, Jan.  doi: 10.1016/j.asoc.2020.106734.
    """
    
    name = "CSA"
    alg_type = "EA" 
    
    def __init__(self, alpha: float = 0.10, beta: float = 0.15, M: int = 3,
                 nPop: int = 25,
                 maxIters: int=  1000,
                 maxFEs: int = 50000,
                 maxTolerates: int = 1000, tolerate: float = 1e-6, 
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = False, saveFlag: bool=True):
        """
        
        Initialize the CSA algorithm with user-defined parameters.
        
        :param alpha: Control parameter for team communication.
        :param beta: Control parameter for reflective learning.
        :param M: Number of global best solutions to maintain.
        :param nPop: Population size.
        
        :param maxIterTimes: Maximum number of iterations.
        :param maxFEs: Maximum number of function evaluations.
        :param maxTolerateTimes: Maximum number of tolerated iterations without improvement.
        :param tolerate: Tolerance for improvement.
        :param verbose: Flag to enable verbose output.
        :param verboseFreq: Frequency of verbose output.
        :param logFlag: Flag to enable logging.
        :param saveFlag: Flag to enable saving results.
        """
        
        super().__init__(maxFEs = maxFEs, maxIters = maxIters, 
                         maxTolerates = maxTolerates, tolerate = tolerate, 
                         verboseFlag = verboseFlag, verboseFreq = verboseFreq, logFlag = logFlag, saveFlag = saveFlag)
        
        # Set user-defined parameters
        self.setParaVal('alpha', alpha)
        self.setParaVal('beta', beta)
        self.setParaVal('M', M)
        self.setParaVal('nPop', nPop)
           
    #------------------Public Function------------------#
    @Verbose.run
    def run(self, problem, seed: Optional[int] = None):
        """
        Execute the CSA algorithm on the specified problem.

        :param problem: An instance of a class derived from ProblemABC.
                        This object defines the optimization problem, including
                        the number of inputs (nInput), number of outputs (nOutput),
                        upper bounds (ub), lower bounds (lb), and evaluation methods.
        
        :return Result: An instance of the Result class, which contains the
                        optimization results, including the best decision variables,
                        objective values, and constraint violations encountered during
                        the optimization process.
        """
        # setup algorithm
        self.setup(problem, seed)
        
        # Retrieve parameter values
        alpha, beta, M = self.getParaVal('alpha', 'beta', 'M')
        nPop = self.getParaVal('nPop')
        
        # Generate initial population
        pop = self.initPop(nPop)
        
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
            
            pop = Population(decs=np.where(uPop.objs < vPop.objs, uPop.decs, vPop.decs), objs=np.minimum(uPop.objs, vPop.objs))

            # Update personal best and global best
            tmp = pop[pop.argsort()[:M]]
            pBest = Population(decs=np.where(pop.objs < pBest.objs, pop.decs, pBest.decs), objs=np.minimum(pop.objs, pBest.objs))
           
            gBest.add(tmp)
            gBest = gBest[gBest.argsort()[:M]]
            
        return self.result
    
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
        t1 = np.random.random((N, D)) * np.abs(c - fai_1) + np.where(c_n > fai_1, fai_1, c_n)
        t2 = np.random.random((N, D)) * np.abs(fai_1 - self.problem.lb) + np.where(fai_1 > lb_n, lb_n, fai_1)
        seed = np.random.random((N, D))
        r = np.where(gailv < seed, t1, t2)
        
        # Calculate p
        t3 = np.random.random((N, D)) * np.abs(fai_1 - c) + np.where(c_n > fai_1, fai_1, c_n)
        t4 = np.random.random((N, D)) * np.abs(self.problem.ub - fai_1) + np.where(fai_1 > ub_n, ub_n, fai_1)
        seed = np.random.random((N, D))
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
        
        idx = np.random.randint(0, M, (N, D))
        A = np.log(1.0 / np.random.random((N, D))) * (gBestDecs[idx, np.arange(D)] - popDecs)
        
        B = alpha * np.random.random((N, D)) * (np.mean(gBestDecs, axis=0) - popDecs)
        
        C = beta * np.random.random((N, D)) * (np.mean(pBestDecs, axis=0) - popDecs)
        
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
            o = num1 + np.random.random(1) * abs(num1 - num2)
        else:
            o = num2 + np.random.random(1) * abs(num1 - num2)
        return o