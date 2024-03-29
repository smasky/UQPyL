#Adaptive Surrogate Modelling-based Optimization
from .sce_ua import SCE_UA
from ..Experiment_Design import LHS
import numpy as np
from tqdm import tqdm
lhs=LHS('center')
class ASMO():
    '''
    evaluator: the function to evaluate the parameters
    surrogate: the used surrogate model
    LB: the lower of the parameters
    UB: the upper of the parameters
    maxFE: the maximum of function evaluations
    '''
    def __init__(self, evaluator, surrogate, LB, UB, NInput, NInit=50, XInit=None, YInit=None):
        self.evaluator=evaluator
        self.surrogate=surrogate
        self.LB=LB; self.UB=UB
        self.NInit=NInit
        self.NInput=NInput
        self.XInit=XInit
        self.YInit=YInit
    def run(self,maxFE=1000, Tolerate=0.001, maxTolerateTime=50, oneStep=False):
        '''
        main procedure
        ''' 
        show_process=tqdm(total=maxFE)
        FE=0
        TT=0
        NInput=self.NInput
        LB=self.LB
        UB=self.UB
            
        if self.XInit is None:
            self.XInit=(UB-LB)*lhs(self.NInit, NInput)+LB
        if self.YInit is None:
            self.YInit=self.evaluator(self.XInit)
           
        
        XPop=self.XInit
        YPop=self.YInit
        
        FE=YPop.shape[0]
        show_process.update(FE)
        
        idx=np.argsort(YPop, axis=0)
        BestY=YPop[idx[0,0],0]
        BestX=XPop[idx[0,0],:]
        if (oneStep==False):
            while FE<maxFE and TT<maxTolerateTime:
                show_process.update(1)
                # Build surrogate model
                self.surrogate.fit(XPop, YPop)
                BestX_SM, _=SCE_UA(self.surrogate.predict, NInput, LB, UB).run()
                
                TempY=self.evaluator(BestX_SM)
                FE+=1
                XPop=np.vstack((XPop,BestX_SM))
                YPop=np.vstack((YPop,TempY))
                
                if TempY[0,0]<BestY:
                    BestY=np.copy(TempY)
                    BestX=np.copy(BestX_SM)
        else:
            self.surrogate.fit(XPop, YPop)
            BestX_SM, _=SCE_UA(self.surrogate.predict, NInput, LB, UB).run()
            
            TempY=self.evaluator(BestX_SM)
            FE+=1
            XPop=np.vstack((XPop,BestX_SM))
            YPop=np.vstack((YPop,TempY))
            
            if TempY[0,0]<BestY:
                BestY=np.copy(TempY)
                BestX=np.copy(BestX_SM) 
        return BestX, BestY
            
            
        
            
            
