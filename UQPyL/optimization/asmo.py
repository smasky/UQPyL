import numpy as np
from tqdm import tqdm
from typing import Optional

from .sce_ua import SCE_UA
from .pso import PSO
from ..DoE import LHS
from ..problems import Problem
from ..surrogates import Surrogate

lhs=LHS('classic')
class ASMO():
    '''
        Adaptive Surrogate Modelling-based Optimization <Single> <Surrogate>
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
            surrogate: Surrogate
                the surrogate model you want to use
            n_init: int, default=50
                Number of initial samples for surrogate modelling
    '''
    def __init__(self, problem: Problem, surrogate: Surrogate, 
                    n_init: int=50, x_init: Optional[np.ndarray]=None, y_init: Optional[np.ndarray]=None,
                    maxFE: int=500, maxTolerateTime=50):
        #base setting
        self.evaluate=problem.evaluate
        self.lb=problem.lb; self.ub=problem.ub
        self.n_input=problem.n_input
        self.maxFE=maxFE; self.maxTolerateTime=maxTolerateTime
        
        #surrogate setting
        self.surrogate=surrogate
        self.n_init=n_init
        self.x_init=x_init
        self.y_init=y_init
        
        #construct optimization problem to combine surrogate and algorithm
        self.subProblem=Problem(self.surrogate.predict, self.n_input, 1, self.ub, self.lb)
        
    def run(self, oneStep=False):
        '''
        main procedure
        ''' 
        show_process=tqdm(total=self.maxFE)
        FE=0
        TT=0
        n_input=self.n_input
        lb=self.lb
        ub=self.ub
            
        lhs=LHS('classic', problem=self.problem)
        if self.x_init is None:
            self.x_init=lhs(self.n_samples, n_input)
        if self.y_init is None:
            self.y_init=self.evaluate(self.x_init)
           
        XPop=self.x_init
        YPop=self.y_init
        
        FE=YPop.shape[0]
        show_process.update(FE)
        ###
        idx=np.argsort(YPop, axis=0)
        BestY=YPop[idx[0,0],0]
        BestX=XPop[idx[0,0],:]
        history_best_decs={}
        history_best_objs={}
        # history_BestY=[]; history_BestX=[]
        # history_BestX.append(BestX)
        # history_BestY.append(BestY)
        
        if (oneStep==False):
            while FE<self.maxFE and TT<self.maxTolerateTime:
                show_process.update(1)
                # Build surrogate model
                self.surrogate.fit(XPop, YPop)
                res=PSO(self.subProblem).run()
                BestX_SM=res['best_dec']
                
                TempY=self.evaluate(BestX_SM)
                FE+=1
                XPop=np.vstack((XPop,BestX_SM))
                YPop=np.vstack((YPop,TempY))
                
                if TempY[0,0]<BestY:
                    BestY=np.copy(TempY[0,0])
                    BestX=np.copy(BestX_SM)
                    history_best_decs[FE]=BestX
                    history_best_objs[FE]=BestY
        else:
            self.surrogate.fit(XPop, YPop)
            res=SCE_UA(self.subProblem).run()
            BestX_SM=res['best_dec']
            
            TempY=self.evaluate(BestX_SM)
            
            FE+=1
            XPop=np.vstack((XPop,BestX_SM))
            YPop=np.vstack((YPop,TempY))
            
            if TempY[0,0]<BestY:
                BestY=np.copy(TempY)
                BestX=np.copy(BestX_SM)
                history_best_decs[FE]=BestX
                history_best_objs[FE]=BestY
        
        Result={'best_dec':BestX, 'best_obj':BestY, 'history_best_decs': history_best_decs, 'history_best_objs':history_best_objs ,'FEs':FE}
        
        return Result
            
            
        
            
            
