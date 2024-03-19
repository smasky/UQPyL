from .surrogate_ABC import Surrogate, Scale_T
from typing import Literal,Tuple
import numpy as np
from scipy.linalg import LinAlgError, cholesky, qr
from scipy.spatial.distance import pdist

def regrpoly0(S):
    
    n_sample, _ = S.shape
    return np.ones((n_sample, 1))

def regrpoly1(S):
    
    n_sample, _ = S.shape
    return np.hstack((np.ones((n_sample, 1)), S))

def regrpoly2(S):
    
    n_sample, n_feature = S.shape
    nn = int((n_feature + 1) * (n_feature + 2) / 2)
    F = np.hstack((np.ones((n_sample, 1)), S, np.zeros((n_sample, nn - n_feature - 1))))
    j = n_feature + 1
    q = n_feature

    for k in np.arange(1, n_feature + 1):
        F[:, j + np.arange(q)] = np.tile(S[:, (k - 1):k],
                                            (1, q)) * S[:, np.arange(k - 1, n_feature)]
        j += q;q -= 1
    return F

class Kriging(Surrogate):
    def __init__(self, regression: Literal['poly0','poly1','poly2'], correlation: ['corrgauss'],
                 optimizer: Literal['boxmin', 'GA'] = 'boxmin', 
                 fitMode: Literal['likelihood', 'predictError']='likelihood',
                 normalized: Tuple[bool,bool]=(False,False),
                 Scale_type: Scale_T=('StandardScaler','StandardScaler')):
         
        self.regression=regression
        self.correlation=correlation
        self.optimizer=optimizer
        self.fitMode=fitMode
        
        if(regression=='poly0'):
            self.regrFunc=regrpoly0
        elif(regression=='poly1'):
            self.regrFunc=regrpoly1
        elif(regression=='poly2'):
            self.regrFunc=regrpoly2
            
        super().__init__(normalized,Scale_type)
        
    def fit_likelihood(self,trainX,trainY):
        
        self.trainX=trainX
        self.trainY=trainY
        
        F, D=self._initialize()
        
        
        
    def _initialize(self):
        
        n_sample, n_feature=self.trainX.shape
        
        D = np.zeros((n_sample*(n_sample-1)/2, n_feature))
        for k in range(n_feature):
            D[:, k] = pdist(self.trainX[:, [k]], metric='euclidean')
        
        F=self.regrFunc(self.trainX)
        
        return F, D

        
        
            