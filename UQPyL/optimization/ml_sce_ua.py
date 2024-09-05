#M&L Shuffled Complex Evolution-UA

import numpy as np
from .optimizer import Optimizer, Population, verboseForRun, verboseSetting

class ML_SCE_UA(Optimizer):
    """
        References:
            [1] Duan, Q., Sorooshian, S., & Gupta, V. K. (1992). Effective and efficient global optimization for conceptual rainfall-runoff models. Water Resources Research, 28(4), 1015-1031.
            [2] Duan, Q., Gupta, V. K., & Sorooshian, S. (1994). Optimal use of the SCE-UA global optimization method for calibrating watershed models. Journal of Hydrology, 158(3-4), 265-284.
            [3] Duan, Q., Sorooshian, S., & Gupta, V. K. (1994). A shuffled complex evolution approach for effective and efficient global minimization. Journal of optimization theory and applications, 76(3), 501-521.
            [4] Muttil N , Liong S Y (2006).Improved robustness and efficiency of the SCE-UA model-calibrating algorithm. Advances in Geosciences.
    """
    name="ML-SCE-UA"
    type="EA"
    def __init__(self, ngs: int= 3, npg: int=7, nps: int=4, nspl: int=7,
          maxFEs: int= 50000, 
          maxIterTimes: int= 1000, 
          maxTolerateTimes: int= 1000, tolerate: float=1e-6,
          verbose: bool=True, verboseFreq: int=10, logFlag: bool=False):
        
        super().__init__(maxFEs=maxFEs, maxIterTimes=maxIterTimes, 
                         maxTolerateTimes=maxTolerateTimes, tolerate=tolerate, 
                         verbose=verbose, verboseFreq=verboseFreq, logFlag=logFlag)
        
        #algorithm setting
        self.ngs=ngs
        self.npg=2 * ngs + 1
        self.nps=nps
        self.nspl=nspl
        
        #setting record
        self.setting["ngs"]=ngs
        self.setting["npg"]=npg
        self.setting["nps"]=nps
        self.setting["nspl"]=nspl
        
    @verboseForRun
    def run(self, problem, xInit=None, yInit=None):
        
        self.problem=problem
        self.FEs=0; self.iters=0; self.tolerateTimes=0
        
        if self.ngs==0:
            self.ngs=problem.n_input 
            if self.ngs>15:
                self.ngs=15
        
        # Initialize SCE parameters:
        npg  = 2 * self.ngs + 1
        nps  = self.ngs + 1
        nspl = self.npg
        self.nInit  = npg * self.ngs
        BD  = self.problem.ub - self.problem.lb
        
        if self.verbose or self.logFlag:
            print("When invoking the problem, the new setting is:")
            verboseSetting(self)
        
        if xInit is not None:
            if yInit is not None:
                pop=Population(xInit, yInit)
            else:
                pop=Population(xInit)
                self.evaluate(pop)
        else:
            pop=self.initialize()
        
        #Sort the population in order of increasing function values
        idx=pop.argsort()
        pop=pop[idx]
                 
        self.record(pop)
        
        while self.checkTermination():

            for igs in range(self.ngs):
                # Partition the population into complexes (sub-populations)
                outerIdx = np.linspace(0, npg-1, npg, dtype=np.int64) * self.ngs + igs
                igsPop=pop[outerIdx]
                
                # Evolve sub-population igs for nspl steps
                for _ in range(nspl):
                    # Select simplex by sampling the complex according to a linear
                    # Compute Probability distribution and random choose
                    p=2*(npg+1-np.linspace(1, npg, npg))/((npg+1)*npg)
                    innerIdx=np.random.choice(npg, nps, p=p, replace=False)
                    innerIdx = np.sort(innerIdx)
                    sPop = igsPop[innerIdx]
                    bPop= igsPop[0]
                    #Execute CCE for simplex
                    sNew= self.cce(sPop, bPop)
                    igsPop.replace(innerIdx[-1], sNew)
                    
                    # igsPop=igsPop[igsPop.argsort()]
                    
                # End of Inner Loop for Competitive Evolution of Simplexes
                pop.replace(outerIdx, igsPop)
                
            idx=pop.argsort()
            pop=pop[idx]
            # End of Loop on Complex Evolution;
            # Shuffled the complexes
            self.record(pop)
            
        return self.result
                     
    def cce(self, sPop, bPop):
        
        n, d = sPop.size()
        alpha = 1.0
        beta = 0.5
        sita = 0.2
        sWorst=sPop[-1:]
        ce=np.mean(sPop[:n].decs, axis=0).reshape(1, -1)
        
        sNew=((sWorst-ce) * alpha * -1 + ce )*(1-sita)+bPop*sita
        sNew.clip(self.problem.lb, self.problem.ub)
        
        self.evaluate(sNew)
        
        if sNew.objs[0] > sWorst.objs[0]:
            sNew=(sWorst + (sNew-sWorst) * beta)*(1-sita)+bPop*sita
            self.evaluate(sNew)
        
        # Both reflection and contraction have failed, attempt a random point
            if sNew.objs[0] > sWorst.objs[0]:
                sNew.decs = self.problem.lb + np.random.random(d) * (self.problem.ub - self.problem.lb)
                self.evaluate(sNew)
        # END OF CCE
        return sNew