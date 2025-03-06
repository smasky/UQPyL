# Cooperation search algorithm <Single>
import numpy as np

from ..algorithmABC import Algorithm, Population
from ...utility import Verbose

class CSA(Algorithm):
    """
    Cooperative Search Algorithm (CSA) <Single>
    -------------------------------------------------
    This class implements a single-objective cooperative search algorithm for optimization.
    
    References:
        [1] Z. Feng, W. Niu, and S. Liu (2021), Cooperation search algorithm: A novel metaheuristic evolutionary intelligence algorithm for numerical optimization and engineering optimization problems, Appl. Soft. Comput., vol. 98, p. 106734, Jan.  doi: 10.1016/j.asoc.2020.106734.
    """
    
    name = "CSA"
    type = "EA" 
    
    def __init__(self, alpha: float = 0.10, beta: float = 0.15, M: int = 3,
                 nPop: int = 50,
                 maxIterTimes: int=  1000,
                 maxFEs: int = 50000,
                 maxTolerateTimes: int = 1000, tolerate: float = 1e-6, 
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
        
        super().__init__(maxFEs = maxFEs, maxIterTimes = maxIterTimes, 
                         maxTolerateTimes = maxTolerateTimes, tolerate = tolerate, 
                         verboseFlag = verboseFlag, verboseFreq = verboseFreq, logFlag = logFlag, saveFlag = saveFlag)
        
        # Set user-defined parameters
        self.setPara('alpha', alpha)
        self.setPara('beta', beta)
        self.setPara('M', M)
        self.setPara('nPop', nPop)
           
    #------------------Public Function------------------#
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem, xInit=None, yInit=None):
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
        
        # Initialization
        # Retrieve parameter values
        alpha, beta, M = self.getParaVal('alpha', 'beta', 'M')
        nPop = self.getParaVal('nPop')
        
        # Set the problem to solve
        self.problem = problem
        
        # Initialize termination conditions
        self.FEs = 0; self.iters = 0; self.tolerateTimes = 0
        
        # Generate initial population
        pop = self.initialize(nPop)
        
        # Initial personal best and global best
        pBest = pop.copy()  # Personal Best
        gBest = pBest[pBest.argsort()[:M]]  # Global Best
    
        while self.checkTermination():
            # Team communication operator
            uPop = self.teamCommunicationOperator(pop, pBest, gBest, alpha, beta)
            
            # Reflective learning operator 
            vPop = self.reflectiveLearningOperator(uPop)
            
            # Internal competition operator
            self.evaluate(uPop)
            self.evaluate(vPop)
            
            newPop = Population(decs=np.where(uPop.objs < vPop.objs, uPop.decs, vPop.decs), objs=np.minimum(uPop.objs, vPop.objs))
            self.record(newPop)

            # Update personal best and global best
            tmp = newPop[newPop.argsort()[:M]]
            pBest = Population(decs=np.where(newPop.objs < pBest.objs, newPop.decs, pBest.decs), objs=np.minimum(newPop.objs, pBest.objs))
           
            gBest.add(tmp)
            gBest = gBest[gBest.argsort()[:M]]
            
            pop = newPop.copy()
            
        return self.result
    
    def reflectiveLearningOperator(self, pop):
        """
        Apply the reflective learning operator to the population.

        :param pop: Current population of solutions.
        
        :return: Updated population after applying reflective learning.
        """
        
        N, D = pop.size()
        
        popDecs = pop.decs
        
        c = (self.problem.ub + self.problem.lb) / 2
        
        c_n = np.repeat(c, N, axis=0)
        lb_n = np.repeat(self.problem.lb, N, axis=0)
        ub_n = np.repeat(self.problem.ub, N, axis=0)
        fai_1 = self.problem.ub + self.problem.lb - pop.decs
        
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
        
        vPopDecs = np.where(pop.decs >= c_n, r, p)
        np.clip(vPopDecs, self.problem.lb, self.problem.ub, out=vPopDecs)
        
        vPop = Population(vPopDecs)
        
        return vPop
    
    def teamCommunicationOperator(self, pop, pBest, gBest, alpha, beta):
        """
        Apply the team communication operator to the population.

        :param pop: Current population of solutions.
        :param pBest: Personal best solutions.
        :param gBest: Global best solutions.
        :param alpha: Control parameter for team communication.
        :param beta: Control parameter for reflective learning.
        
        :return: Updated population after applying team communication.
        """
        
        popDecs = pop.decs
        pBestDecs = pBest.decs
        gBestDecs = gBest.decs
        
        N, D = pop.size()
        
        M, _ = gBest.size()
        
        idx = np.random.randint(0, M, (N, D))
        A = np.log(1.0 / np.random.random((N, D))) * (gBestDecs[idx, np.arange(D)] - popDecs)
        
        B = alpha * np.random.random((N, D)) * (np.mean(gBestDecs, axis=0) - popDecs)
        
        C = beta * np.random.random((N, D)) * (np.mean(pBestDecs, axis=0) - popDecs)
        
        uPopDecs = popDecs + A + B + C
        
        np.clip(uPopDecs, self.problem.lb, self.problem.ub, out=uPopDecs)
        
        return Population(uPopDecs)

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