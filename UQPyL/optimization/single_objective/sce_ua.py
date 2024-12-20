#Shuffled Complex Evolution-UA
import numpy as np
import random
from ..algorithmABC import Algorithm, Population, Verbose
class SCE_UA(Algorithm):
    '''
        Shuffled Complex Evolution (SCE-UA) method <Single>
        ----------------------------------------------
        Attributes:
            problem: Problem
                    the problem you want to solve, including the following attributes:
                    n_input: int
                        the input number of the problem
                    ub: 1d-np.ndarray or float
                        the upper bound of the problem
                    lb: 1d-np.ndarray or float
                        the lower bound of the problem
                    evaluate: Callable
                        the function to evaluate the input
            ngs: int, default=0
                Number of complexes (sub-populations), the value 0 means ngs=n_input
            maxFE: int, default=50000
                Maximum number of function evaluations
            maxIter: int, default=1000
                Maximum number of iterations
        Methods:
            run:
                Run the optimization algorithm
        
        References:
            [1] Duan, Q., Sorooshian, S., & Gupta, V. K. (1992). Effective and efficient global optimization for conceptual rainfall-runoff models. Water Resources Research, 28(4), 1015-1031.
            [2] Duan, Q., Gupta, V. K., & Sorooshian, S. (1994). Optimal use of the SCE-UA global optimization method for calibrating watershed models. Journal of Hydrology, 158(3-4), 265-284.
            [3] Duan, Q., Sorooshian, S., & Gupta, V. K. (1994). A shuffled complex evolution approach for effective and efficient global minimization. Journal of optimization theory and applications, 76(3), 501-521.
                
    '''
    
    name="SCE-UA"
    type="EA"
    
    def __init__(self, ngs: int = 3, npg: int = 7, nps: int = 4, nspl: int = 7,
                alpha: float = 1.0, beta: float = 0.5,
                maxFEs: int = 50000, 
                maxIterTimes: int = 1000, 
                maxTolerateTimes: int = 1000, tolerate: float = 1e-6,
                verbose: bool = True, verboseFreq: int = 10, logFlag: bool = False, saveFlag: bool = True):
        
        super().__init__(maxFEs=maxFEs, maxIterTimes=maxIterTimes, 
                         maxTolerateTimes=maxTolerateTimes, tolerate=tolerate, 
                         verbose=verbose, verboseFreq=verboseFreq, logFlag=logFlag, saveFlag=saveFlag)
        
        #algorithm setting
        self.setParameters('ngs', ngs)
        self.setParameters('npg', npg)
        self.setParameters('nps', nps)
        self.setParameters('nspl', nspl)
        self.setParameters('alpha', alpha)
        self.setParameters('beta', beta)
        
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem):
        
        #Parameter Setting
        ngs, npg, nps, nspl = self.getParaValue('nps', 'npg', 'nps', 'nspl')
        alpha, beta = self.getParaValue('alpha', 'beta')
        
        #Problem
        self.setProblem(problem)
        
        #Termination Condition Setting
        self.FEs = 0; self.iters = 0; self.tolerateTimes = 0
        
        #Parameter
        if ngs==0:
            ngs=problem.nInput 
            if ngs > 15:
                ngs = 15
        
        # Initialize SCE parameters:
        npg  = 2*ngs + 1
        nps  = ngs + 1
        nspl = npg
        nInit  = npg * ngs
        
        #Population Generation
        pop = self.initialize(nInit)
        
        #Sort the population in order of increasing function values
        idx = pop.argsort()
        pop = pop[idx]
        
        #Record
        self.record(pop)
        
        while self.checkTermination():
            
            for igs in range(ngs):
                # Partition the population into complexes (sub-populations)
                
                outerIdx = np.linspace(0, npg-1, npg, dtype=np.int64) * ngs + igs
                igsPop = pop[outerIdx]
                
                # Evolve sub-population igs for nspl steps
                
                for _ in range(nspl):
                    
                    # Select simplex by sampling the complex according to a linear
                    # Compute Probability distribution and random choose
                    
                    p = 2*(npg+1-np.linspace(1, npg, npg))/((npg+1)*npg)
                    innerIdx = np.random.choice(npg, nps, p=p, replace=False)
                    innerIdx = np.sort(innerIdx)
                    sPop = igsPop[innerIdx]
                    
                    #Execute CCE for simplex
                    sNew = self.cce(sPop, alpha, beta)
                    igsPop.replace(innerIdx[-1], sNew)
                    
                # End of Inner Loop for Competitive Evolution of Simplexes
                pop.replace(outerIdx, igsPop)
                
            idx = pop.argsort()
            pop = pop[idx]
            # End of Loop on Complex Evolution;
            # Shuffled the complexes
            
            self.record(pop)
            
        return self.result
                     
    def cce(self, sPop, alpha, beta):
        
        N, D = sPop.size()

        sPopDecs = sPop.decs
        
        sWorstDecs = sPop.decs[-1:]
        sWorstObjs = sPop.objs[-1:]
        ce = np.mean(sPopDecs[:N], axis=0).reshape(1, -1)
        
        sNewDecs = (sWorstDecs-ce) * alpha * -1 + ce
        np.clip(sNewDecs, self.problem.lb, self.problem.ub, out=sNewDecs)
        
        sNew = Population(sNewDecs)
        self.evaluate(sNew)
        
        if sNew.objs[0] > sWorstObjs:
            sNewDecs = sWorstDecs + (sNewDecs-sWorstDecs) * beta
            np.clip(sNewDecs, self.problem.lb, self.problem.ub, out=sNewDecs)
            
            sNew = Population(sNewDecs)
            self.evaluate(sNew)
        
        # Both reflection and contraction have failed, attempt a random point
            if sNew.objs[0] > sWorstObjs:
                sNewDecs = self.problem.lb + np.random.random(D) * (self.problem.ub - self.problem.lb)
                sNew = Population(sNewDecs)
                self.evaluate(sNew)
                
        # END OF CCE
        return sNew
            
        
                
                
        
        
        
        
        
        
        
        
        
        