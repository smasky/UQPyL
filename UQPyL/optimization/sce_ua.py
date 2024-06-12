#Shuffled Complex Evolution-UA
import numpy as np
from ..DoE import LHS

class SCE_UA():
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
            kstop: int, default=10
                Maximum number of iterations without improvement
            pcento: float, default=0.1
                Percentage change in the objective function value
            peps: float, default=0.001
                Percentage change in the range of the design variables
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
    def __init__(self, problem,
          ngs: int= 0, kstop: int= 10, 
          pcento: float = 0.1, peps: float= 0.001, 
          maxFE: int= 50000, maxIter: int= 1000):
        
        #problem setting
        self.evaluate=problem.evaluate
        self.NInput=problem.n_input
        self.lb=problem.lb
        self.ub=problem.ub
        
        #algorithm setting
        self.kstop=kstop
        self.pcento=pcento
        self.peps=peps
        if ngs==0:
            self.ngs=problem.n_input
        else:
            self.ngs=ngs
        
        #termination setting
        self.maxFE=maxFE
        self.maxIter=maxIter
        
    def run(self):
        '''
            Run the SCE-UA optimization algorithm
        '''
        # Initialize SCE parameters:
        NInput=self.NInput
        npg  = 2 * self.NInput + 1
        nps  = self.NInput + 1
        nspl = npg
        npt  = npg * self.ngs
        BD   = self.ub - self.lb
        
        #Initialize
        lhs=LHS('classic', problem=self.problem)
        if self.x_init is None:
            self.x_init=lhs(self.n_samples, self.n_input)
        if self.y_init is None:
            self.y_init=self.evaluate(self.x_init)
         
        XPop=self.x_init
        YPop=self.y_init
        FEs=npt
        #Sort the population in order of increasing function values
        idx=np.argsort(YPop, axis=0)
        YPop=YPop[idx[:,0]]
        XPop=XPop[idx[:,0],:]
        
        #Record
        BestX=np.copy(XPop[0, :])
        BestY=np.copy(YPop[0, 0])
        # WorstX=np.copy(XPop[-1, :])
        # WorstY=np.copy(YPop[0, 0])
        history_best_decs={}
        history_best_objs={}
        
        history_best_decs[FEs]=BestX
        history_best_objs[FEs]=BestY
        
        #Setup Setting
        gnrng = np.exp(np.mean(np.log((np.max(XPop,axis=0)-np.min(XPop,axis=0))/BD)))
        nloop=0
        criter=[]
        criter_change = 1e+5
        cx=np.zeros((npg, NInput))
        cf=np.zeros((npg,1))
        ngs=self.ngs
        
        while FEs<self.maxFE and gnrng>self.peps and criter_change>self.pcento and nloop<self.maxIter:
            nloop+=1
            
            for igs in range(ngs):
                # Partition the population into complexes (sub-populations)
                k1 = np.linspace(0, npg-1, npg, dtype=np.int64)
                k2 = k1 * ngs + igs
                cx[k1, :]=np.copy(XPop[k2, :])
                cf[k1, :]=np.copy(YPop[k2, :])
                
                # Evolve sub-population igs for nspl steps
                for _ in range(nspl):
                    # Select simplex by sampling the complex according to a linear
                    # probability distribution
                    lcs=np.random.choice(npg, nps)
                    lcs[0]=0
                    lcs = np.sort(lcs)
                    s = np.copy(cx[lcs,:])
                    sf = np.copy(cf[lcs, :])
                    
                    snew, fnew, FEs = self.cceua(self.evaluate, s, sf, self.lb, self.ub, FEs) #parallel TODO
                    
                    # Replace the worst point in Simplex with the new point:
                    s[nps-1,:] = snew
                    sf[nps-1, :] = fnew[0, :]
                    
                    # Replace the simplex into the complex
                    cx[lcs,:] = np.copy(s)
                    cf[lcs, :] = np.copy(sf)
                    
                    # Sort the complex
                    idx=np.argsort(cf, axis=0)
                    cf=cf[idx[:,0],:]
                    cx=cx[idx[:,0],:]
                # End of Inner Loop for Competitive Evolution of Simplexes
        
                # Replace the complex back into the population
                XPop[k2,:] = np.copy(cx[k1, :])
                YPop[k2,:] = np.copy(cf[k1, :])
                
            # End of Loop on Complex Evolution;
            # Shuffled the complexes
            idx=np.argsort(YPop, axis=0)
            YPop=YPop[idx[:,0]]
            XPop=XPop[idx[:,0],:]
            
            BestX=np.copy(XPop[0, :])
            BestY=np.copy(YPop[0, 0])

            history_best_decs[FEs]=BestX
            history_best_objs[FEs]=BestY
            
            gnrng = np.exp(np.mean(np.log((np.max(XPop,axis=0)-np.min(XPop,axis=0))/BD)))
            
            criter.append(BestY)
            if nloop >= self.kstop:
                criter_change = np.abs(criter[nloop-1] - criter[nloop-self.kstop])*100
                criter_change /= np.mean(np.abs(criter[nloop-self.kstop:nloop]))
            
        Result={}
        Result['best_dec']=BestX
        Result['best_obj']=BestY
        Result['history_best_decs']=history_best_decs
        Result['history_best_objs']=history_best_objs
        Result['FEs']=FEs
        Result['iters']=nloop
        
        return Result
            
                        
    def cceua(self, func, s, sf, lb, ub, FE):
        
        NSample, NInput = s.shape
        alpha = 1.0
        beta = 0.5
        
        sw = s[-1,:].reshape(1,-1)
        fw = sf[-1, 0]
        
        ce = np.mean(s[:NSample-1,:],axis=0).reshape(1,-1)
        snew = ce + alpha * (ce - sw)
        
        ibound = 0
        s1 = snew - lb
        if np.sum(s1 < 0) > 0:
            ibound = 1
        s1 = ub - snew
        if np.sum(s1 < 0) > 0:
            ibound = 2
        if ibound >= 1:
            snew = lb + np.random.random(NInput) * (ub - lb)
        
        fnew=func(snew)
        FE+=1
        
        # Reflection failed; now attempt a contraction point
        if fnew[0,0] > fw:
            snew = sw + beta * (ce - sw)
            fnew = func(snew)
            FE += 1
        
        # Both reflection and contraction have failed, attempt a random point
            if fnew[0,0] > fw:
                snew = lb + np.random.random(NInput) * (ub - lb)
                fnew = func(snew)
                FE += 1

        # END OF CCE
        return snew, fnew, FE
            
        
                
                
        
        
        
        
        
        
        
        
        
        