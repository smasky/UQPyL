import abc
import numpy as np
from typing import Union, Optional, Literal
class ProblemABC(metaclass=abc.ABCMeta):

    def __init__(self, nInput:int, nOutput:int,
                 ub: Union[int, float, list, np.ndarray], lb: Union[int, float, list, np.ndarray],
                 optType: Union[str, list] = 'min', conWgt: Optional[list] = None,
                 varType: Optional[list] = None, varSet: Optional[dict] = None,
                 xLabels: Optional[list] = None, yLabels: Optional[list] = None):
        
        self.nInput = nInput
        self.nOutput = nOutput
        
        self._set_ub_lb(ub,lb)
        
        self.optType = self._check_optType(optType)
        
        self.encoding = "real"
        
        if varType==None:
            self.varType = np.zeros(self.nInput)
            self.idxF = np.arange(self.nInput)
        else:
            if len(varType) != nInput:
                raise ValueError("The length of varType is not equal to nInput.")
            self.encoding = "mix"
            self.varType = np.array(varType, dtype=np.int32)
            self.idxF = np.where(self.varType==0)[0]
            self.idxI = np.where(self.varType==1)[0]
            self.idxD = np.where(self.varType==2)[0]
            
        if varSet is None:
            self.varSet = {}
        else:
            self.varSet = {}
            for i in self.idxD:
                if isinstance(varSet[i], list):
                    self.varSet[i] = varSet[i]
                else:
                    raise ValueError("The type of sub varSet must be list.")
        
        if xLabels is None:
            self.xLabels = ['x_'+str(i) for i in range(1,nInput+1)]
        else:
            self.xLabels = xLabels

        if yLabels is None:
            self.yLabels = ['y_'+str(i) for i in range(1,nOutput+1)]
        else:
            self.yLabels
        
        if conWgt is not None:
            
            if not isinstance(conWgt, list):
                raise ValueError('The type of conWgt must be list or None.')
                
            conWgt = np.array(conWgt).reshape(1, -1)

        self.conWgt = conWgt
        
    def evaluate(self, X):
        
        #Use the user-define way
        if hasattr(self, 'evaluate_') and self.evaluate_ is not None:
            return self.evaluate_(X)
        
        #Use the default way
        ##calObjs
        objs = self.objFunc(X)
        
        ##calConstraints
        cons = self.conFunc(X)
        
        return {'objs': objs, 'cons': cons}

    def objFunc(self, X):
        
        if hasattr(self, 'objFunc_') and self.objFunc_ is not None:
            return self.objFunc_(X)
        
        return np.full( (X.shape[0], 1), np.inf )
    
    def conFunc(self, X):
        
        if hasattr(self, 'conFunc_') and self.conFunc_ is not None:
            return self.conFunc_(X)
        
        return None
        
    def getOptimum(self):
        
        pass
    
    def _check_optType(self, t):
        
        if isinstance(t, str):
            
            if t not in ['min', 'max']:
                raise ValueError("The optType must be 'min' or 'max'.")
            
            if t == 'min':
                self.opt = 1
            else:
                self.opt = -1
                
            t = [t.lower()]
        elif isinstance(t, list):
            if len(t) != self.nOutput:
                raise ValueError("The length of optType must be equal to nOutput.")
            
            for i in t:
                if i not in ['min', 'max']:
                    raise ValueError("The optType must be 'min' or 'max'.")
            
            t = [i.lower() for i in t]
            
            self.opt = np.array([1 if i == 'min' else -1 for i in t])
            
        else:
            raise ValueError("The type of optType must be str or list.")
        
        return " ".join(t)

    def _transform_discrete_var(self, X):
        
        for i in self.idxD:
            S = self.varSet[i]
            num_interval = len(S)
            bins = np.linspace(self.lb[0, i], self.ub[0, i], num_interval+1)
            indices = np.digitize(X[:, i], bins, right=False) - 1
            indices[indices == num_interval] = num_interval-1
            X[:, i] = np.array([S[i] for i in indices])
        
        return X
    
    def _transform_int_var(self, X):
        
        X[:, self.idxI] = np.round(X[:, self.idxI])

        return X
    
    def _transform_unit_X(self, X, dst=True):
        
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        
        X_scaled = (X - X_min) / (X_max - X_min)
        X_scaled = X_scaled*(self.ub-self.lb)+self.lb
        
        if self.encoding == 'mix':
            
            self._transform_int_var(X_scaled)
                
            if dst:
                X_scaled = self._transform_discrete_var(X_scaled)
        
        return X_scaled 
    
    def _set_ub_lb(self, ub: Union[int, float, list, np.ndarray], 
                        lb: Union[int, float, list, np.ndarray]):
        
        if (isinstance(ub, (int, float))):
            self.ub = np.ones((1,self.nInput))*ub
            
        elif(isinstance(ub, np.ndarray)):
            self._check_bound(ub)
            self.ub = ub[:, np.newaxis]
        
        elif(isinstance(ub, list)):
            self.ub = np.array(ub)[:, np.newaxis]
            self._check_bound(self.ub)
            
        else:
            raise ValueError("The type of ub is not supported.")
        
        if (isinstance(lb, (int, float))):
            self.lb = np.ones((1,self.nInput))*lb
            
        elif(isinstance(lb, np.ndarray)):
            self._check_bound(lb)
            self.lb = lb[:, np.newaxis]
        
        elif(isinstance(lb, list)):
            self.lb = np.array(lb)[:, np.newaxis]
            self._check_bound(self.lb)
        
        else:
            raise ValueError("The type of lb is not supported.")
    
    def _check_X_2d(self, X):
        
        X = np.atleast_2d(X)
        
        return X
    
    
    def _check_bound(self,bound: np.ndarray):
        
        bound = bound.ravel()
        if( not bound.shape[0] == self.nInput ):
            raise ValueError('The input bound is inconsistent with the nInput of the problem setting')
        
    @staticmethod
    def singleFunc(func):
        
        def wrapper(X):
            X = np.atleast_2d(X)
            evals = []
            
            for x in X:
                eval = func(x)
                evals.append(np.atleast_1d(eval))

            return np.vstack(evals)
        
        return wrapper
    
    @staticmethod
    def singleEval(func):
        
        def wrapper(X):
            X = np.atleast_2d(X)
            
            objs =[]
            cons =[]
            
            for x in X:
                res = func(x)
                objs.append(np.atleast_1d(res['objs']))
                if 'cons' in res:
                    cons.append(np.atleast_1d(res['cons']))
            
            res = {'objs': np.vstack(objs)}
            
            if len(cons) != 0:
                res['cons'] = np.vstack(cons)
                
            return res
        
        return wrapper 