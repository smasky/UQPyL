# M&L Shuffled Complex Evolution-UA <Single>

import numpy as np
from ..algorithmABC import Algorithm, Population, Verbose

class ML_SCE_UA(Algorithm):
    """
    M&L Shuffled Complex Evolution-UA (SCE-UA) Algorithm
    ----------------------------------------------------
    This class implements the SCE-UA algorithm for single-objective optimization.
    
    References:
        [1] Duan, Q., Sorooshian, S., & Gupta, V. K. (1992). Effective and efficient global optimization for conceptual rainfall-runoff models. Water Resources Research, 28(4), 1015-1031.
        [2] Duan, Q., Gupta, V. K., & Sorooshian, S. (1994). Optimal use of the SCE-UA global optimization method for calibrating watershed models. Journal of Hydrology, 158(3-4), 265-284.
        [3] Duan, Q., Sorooshian, S., & Gupta, V. K. (1994). A shuffled complex evolution approach for effective and efficient global minimization. Journal of optimization theory and applications, 76(3), 501-521.
        [4] Muttil N , Liong S Y (2006). Improved robustness and efficiency of the SCE-UA model-calibrating algorithm. Advances in Geosciences.
    """
    
    name = "ML-SCE-UA"
    type = "EA"
    
    def __init__(self, ngs: int = 3, npg: int = 7, nps: int = 4, nspl: int = 7, 
                 alpha: float = 1.0, beta: float = 0.5, sita: float = 0.2,
                 maxFEs: int = 50000, 
                 maxIterTimes: int = 1000, 
                 maxTolerateTimes: int = 1000, tolerate: float = 1e-6,
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = False, saveFlag: bool = True):
        """
        Initialize the SCE-UA algorithm with user-defined parameters.
        
        :param ngs: Number of complexes.
        :param npg: Number of points in each complex.
        :param nps: Number of points in each simplex.
        :param nspl: Number of evolution steps for each complex.
        :param alpha: Reflection coefficient.
        :param beta: Contraction coefficient.
        :param sita: Smoothing parameter.
        :param maxFEs: Maximum number of function evaluations.
        :param maxIterTimes: Maximum number of iterations.
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
        
        # Set algorithm parameters
        self.setPara('ngs', ngs)
        self.setPara('npg', npg)
        self.setPara('nps', nps)
        self.setPara('nspl', nspl)
        self.setPara('alpha', alpha)
        self.setPara('beta', beta)
        self.setPara('sita', sita)
        
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem, xInit=None, yInit=None):
        """
        Execute the SCE-UA algorithm on the specified problem.

        :param problem: An instance of a class derived from ProblemABC.
                        This object defines the optimization problem, including
                        the number of inputs (nInput), number of outputs (nOutput),
                        upper bounds (ub), lower bounds (lb), and evaluation methods.
        :param xInit: Initial decision variables (optional).
        :param yInit: Initial objective values (optional).
        
        :return Result: An instance of the Result class, which contains the
                        optimization results, including the best decision variables,
                        objective values, and constraint violations encountered during
                        the optimization process.
        """
        
        # Retrieve parameter values
        ngs, npg, nps, nspl = self.getParaVal('ngs', 'npg', 'nps', 'nspl')
        alpha, beta, sita = self.getParaVal('alpha', 'beta', 'sita')
        
        # Set the problem to solve
        self.setProblem(problem)
        
        # Initialize termination conditions
        self.FEs = 0; self.iters = 0; self.tolerateTimes = 0
    
        # Adjust number of complexes if necessary
        if ngs == 0:
            ngs = problem.nInput 
            if ngs > 15:
                ngs = 15
        
        # Initialize SCE parameters
        npg = 2 * ngs + 1
        nps = ngs + 1
        nspl = npg
        nInit = npg * ngs
    
        # Generate initial population
        pop = self.initialize(nInit)
        
        # Sort the population in order of increasing function values
        idx = pop.argsort()
        pop = pop[idx]
                 
        # Record initial population state
        self.record(pop)
        
        # Iterative process
        while self.checkTermination():
            for igs in range(ngs):
                # Partition the population into complexes (sub-populations)
                outerIdx = np.linspace(0, npg-1, npg, dtype=np.int64) * ngs + igs
                igsPop = pop[outerIdx]
                
                # Evolve sub-population igs for nspl steps
                for _ in range(nspl):
                    # Select simplex by sampling the complex according to a linear probability distribution
                    p = 2 * (npg + 1 - np.linspace(1, npg, npg)) / ((npg + 1) * npg)
                    innerIdx = np.random.choice(npg, nps, p=p, replace=False)
                    innerIdx = np.sort(innerIdx)
                    sPop = igsPop[innerIdx]
                    bPop = igsPop[0]
                    
                    # Execute CCE for simplex
                    sNew = self.cce(sPop, bPop, alpha, beta, sita)
                    igsPop.replace(innerIdx[-1], sNew)
                    
                # End of inner loop for competitive evolution of simplexes
                pop.replace(outerIdx, igsPop)
                
            # Sort the population again
            idx = pop.argsort()
            pop = pop[idx]
            
            # Record the current state of the population
            self.record(pop)
            
        # Return the final result
        return self.result
                     
    def cce(self, sPop, bPop, alpha, beta, sita):
        """
        Competitive Complex Evolution (CCE) for a given simplex.

        :param sPop: The current simplex population.
        :param bPop: The best population member.
        :param alpha: Reflection coefficient.
        :param beta: Contraction coefficient.
        :param sita: Smoothing parameter.
        
        :return: The new population after CCE.
        """
        
        N, D = sPop.size()
        
        sPopDecs = sPop.decs
        bPopDecs = bPop.decs
        
        sWorst = sPop[-1:]
        sWorstDecs = sWorst.decs
        sWorstObjs = sWorst.objs
        
        # Calculate the centroid of the simplex
        ce = np.mean(sPopDecs[:N], axis=0).reshape(1, -1)
        
        # Reflection step
        sNewDecs = ((sWorstDecs - ce) * alpha * -1 + ce) * (1 - sita) + bPopDecs * sita
        np.clip(sNewDecs, self.problem.lb, self.problem.ub, out=sNewDecs)
        
        sNew = Population(sNewDecs)
        self.evaluate(sNew)
        
        # Contraction step if reflection fails
        if sNew.objs[0] > sWorstObjs[0]:
            sNewDecs = (sWorstDecs + (sNewDecs - sWorstDecs) * beta) * (1 - sita) + bPopDecs * sita
            np.clip(sNewDecs, self.problem.lb, self.problem.ub, out=sNewDecs)
            
            sNew = Population(sNewDecs)
            self.evaluate(sNew)
        
        # Random point if both reflection and contraction fail
        if sNew.objs[0] > sWorstObjs[0]:
            sNewDecs = self.problem.lb + np.random.random(D) * (self.problem.ub - self.problem.lb)
            sNew = Population(sNewDecs)
            self.evaluate(sNew)
        
        # End of CCE
        return sNew