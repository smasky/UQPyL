import numpy as np
from typing import Union

from ..problemABC import ProblemABC

###################Basic Test Function##################
#Reference: 
#CEC2010 Constrained Test Suit
###############################################################

class RosenbrockWithCon(ProblemABC):
    '''
    Types:
        Single Optimization Unimodal
        
    F5-> Rosenbrock Function:
        F= \sum \left ( 100\left ( x_{i+1} - x_i^2 \right ) ^2 - \left ( x_i-1 \right ) ^2 \right )
     
    Default setting:
        Dims->30;Ub->np.ones(1,30)*30;LB->np.ones(1,30)*-30
     
    Optimal:
        X*=1 1 1 ... 1
        F*=0
    '''
    name="RosenbrockWithConstraint"
    
    def __init__(self, nInput:int =30, 
                    ub: Union[int,float,np.ndarray] =30,
                        lb: Union[int,float,np.ndarray] =-30):
        
        super().__init__(nInput, 1, ub, lb)
        
    def objFunc(self, X: np.ndarray) -> np.ndarray:
        
        X = self._check_X_2d(X)
           
        Temp1 = 100*np.square(X[:, 1:]-np.square(X[:, :-1]))
        Temp2 = np.square( X[:, :-1] -1 )
        F = np.sum(Temp1+Temp2, axis=1)[:, np.newaxis]
        
        return F
    
    def conFunc(self, X: np.ndarray) -> np.ndarray:
        
        X = self._check_X_2d(X)
        
        CV = (X[:, 0]**2 + X[:, 1]**2 -1)[:, np.newaxis]
        
        return CV