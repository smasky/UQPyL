### Multi-Objective Adaptive Surrogate Modelling-based Optimization
import numpy as np
from ..Experiment_Design import LHS
from .nsga_ii import NSGAII

lhs=LHS("center")
class MOASMO():
    def __init__(self, evaluator, surrogates, LB, UB, NInput, NOutput, Pct=0.2,NInit=50, XInit=None, YInit=None):
        self.evaluator=evaluator
        self.surrogates=surrogates
        self.LB=LB; self.UB=UB
        self.NInit=NInit
        self.NInput=NInput
        self.NOutput=NOutput
        self.XInit=XInit
        self.YInit=YInit
        self.Pct=Pct
        
    def run(self, maxFE=1000):
        pct=self.Pct
        NInit=self.NInit
        N_Resample=np.floor(NInit*pct)
        UB=self.UB; LB=self.LB
        
        if self.XInit is None:
            self.XInit=(UB-LB)*lhs(self.NInit, self.NInput)+LB
            
        if self.YInit is None:
            self.YInit=self.evaluator(self.XInit)
        
        FE=NInit
        XPop=self.XInit
        YPop=self.YInit
        
        while FE<maxFE:
            #build surrogate
            self.surrogates.fit(XPop, YPop)
            
            nsga_ii=NSGAII(self.surrogates.predict, self.NInput, self.NOutput, LB, UB, NInit)
            #main optimization
            BestX, BestY, FrontNo, CrowdDis=nsga_ii.run()

            if BestY.shape[0]>N_Resample:
                idx=CrowdDis.argsort()[::-1][:N_Resample]
                BestX=np.copy(BestX[idx])
                BestY=np.copy(BestY[idx])
            
            BestY=self.evaluator(BestY)
            XPop=np.vstack((XPop,BestX))
            YPop=np.vstack((YPop,BestY))
        
        FrontNo, _ =nsga_ii.NDSort(YPop, YPop.shape[0])
        idx=np.where(FrontNo==1)
        ND_XPop=XPop[idx]
        ND_YPop=YPop[idx]
        
        return ND_XPop, ND_YPop
          
        
                
        
            
        
            
            
        
        
        