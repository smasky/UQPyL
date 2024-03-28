#Adaptive Surrogate Modelling-based Optimization

from ..Experiment_Design import LHS
lhs=LHS('center_maximin')
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
        
        if XInit is None:
            self.XInit=(UB-LB)*lhs(NInit, NInput)+LB
        else:
            self.XInit=XInit
            
        if YInit is None:
            self.YInit=self.evaluator(XInit)
        else:
            self.YInit=YInit
    
    def run(self,maxFE=1000, Tolerate=0.001, maxTolerateTime=50):
        '''
        main procedure
        '''
        FE=0
        TT=0
        XPop=self.XInit
        YPop=self.YInit
        
        while FE>maxFE or TT>maxTolerateTime:
            # Build surrogate model
            self.surrogate.fit(XPop, YPop)
            
            
            
        
            
            
