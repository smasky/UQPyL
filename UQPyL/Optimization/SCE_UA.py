#Shuffled Complex Evolution-UA
from ..Experiment_Design import LHS
import numpy as np
lhs=LHS('center_maximin')
class SCE_UA():
    '''
    '''
    def __init__(self, evaluator, NInput, LB, UB, 
          ngs = None, kstop = 10, 
          pcento = 0.1, peps = 0.001, verbose = True, simple=True):
        self.func=evaluator
        self.NInput=NInput
        self.LB=LB
        self.UB=UB
        self.ngs=ngs
        self.kstop=kstop
        self.pcento=pcento
        self.peps=peps
        self.verbose=verbose
        self.simple=simple
        
        if ngs==None:
            self.ngs=NInput
            
    def run(self, maxFE=50000, maxIter=3000):
        # Initialize SCE parameters:
        NInput=self.NInput
        npg  = 2 * self.NInput + 1
        nps  = self.NInput + 1
        nspl = npg
        npt  = npg * self.ngs
        BD   = self.UB - self.LB
        
        #Initialize 
        XPop=BD*lhs(npt, self.NInput)+self.LB
        YPop=self.func(XPop)
        FE=npt
        #Sort the population in order of increasing function values
        idx=np.argsort(YPop, axis=0)
        YPop=YPop[idx[:,0]]
        XPop=XPop[idx[:,0],:]
        
        #Record
        BestX=np.copy(XPop[0, :])
        BestY=np.copy(YPop[0, 0])
        # WorstX=np.copy(XPop[-1, :])
        # WorstY=np.copy(YPop[0, 0])
        List_BestX=[BestX]
        List_BestY=[BestY]
        List_FE=[FE]
        #Setup Setting
        gnrng = np.exp(np.mean(np.log((np.max(XPop,axis=0)-np.min(XPop,axis=0))/BD)))
        nloop=0
        criter=[]
        criter_change = 1e+5
        cx=np.zeros((npg, NInput))
        cf=np.zeros((npg,1))
        ngs=self.ngs
        
        while FE<maxFE and gnrng>self.peps and criter_change>self.pcento and nloop<maxIter:
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
                    
                    snew, fnew, FE = self.cceua(self.func, s, sf, self.LB, self.UB, FE) #parallel TODO
                    
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
            # WorstX=np.copy(XPop[-1, :])
            # WorstY=np.copy(YPop[0, 0])
            List_BestX.append(BestX)
            List_BestY.append(BestY)
            List_FE.append(FE)
            
            gnrng = np.exp(np.mean(np.log((np.max(XPop,axis=0)-np.min(XPop,axis=0))/BD)))
            
            criter.append(BestY)
            if nloop >= self.kstop:
                criter_change = np.abs(criter[nloop-1] - criter[nloop-self.kstop])*100
                criter_change /= np.mean(np.abs(criter[nloop-self.kstop:nloop]))
                
        if self.simple:
            return BestX, BestY
        else:
            return BestX, BestY, FE, nloop, List_BestX, List_BestY, List_FE
                        
                        
    def cceua(self, func, s, sf, LB, UB, FE):
        
        NSample, NInput = s.shape
        alpha = 1.0
        beta = 0.5
        
        sw = s[-1,:]
        fw = sf[-1, 0]
        
        ce = np.mean(s[:NSample-1,:],axis=0)
        snew = ce + alpha * (ce - sw)
        
        ibound = 0
        s1 = snew - LB
        if np.sum(s1 < 0) > 0:
            ibound = 1
        s1 = UB - snew
        if np.sum(s1 < 0) > 0:
            ibound = 2
        if ibound >= 1:
            snew = LB + np.random.random(NInput) * (UB - LB)
        
        fnew=func(snew)
        FE+=1
        
        # Reflection failed; now attempt a contraction point
        if fnew[0,0] > fw:
            snew = sw + beta * (ce - sw)
            fnew = func(snew)
            FE += 1
        
        # Both reflection and contraction have failed, attempt a random point
            if fnew[0,0] > fw:
                snew = LB + np.random.random(NInput) * (UB - LB)
                fnew = func(snew)
                FE += 1

        # END OF CCE
        return snew, fnew, FE
            
        
                
                
        
        
        
        
        
        
        
        
        
        