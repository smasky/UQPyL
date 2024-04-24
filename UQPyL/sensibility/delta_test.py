#Delta test

import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional, Tuple

from ..optimization import Binary_GA
from .sa_ABC import SA
from ..DoE import LHS, FAST_Sampler, Sampler
from ..problems import ProblemABC as Problem
from ..utility import Scaler
from ..surrogates import Surrogate
class Delta_Test(SA):
    def __init__(self, problem: Problem, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None), 
                       n_neighbors: int=2):
        '''
            Delta Test 
           --------------------------
            Parameters:
                problem: Problem
                    the problem you want to analyse
                scaler: Tuple[Scaler, Scaler], default=(None, None)
                    used for scaling X or Y
                n_neighbors: int default=2
                    the number of nearest neighbors in the subspace of S
                 
                Following parameters derived from the variable 'problem'
                n_input: the input number of the problem
                ub: the upper bound of the problem
                lb: the lower bound of the problem
                
            Methods:
                sample: Generate a sample for Delta Test analysis
                analyze: perform Delta Test analyze from the X and Y you provided.
                
            Examples:
                >>> delta_method=Delta_Test(problem)
                >>> X=delta_method.sample(500)
                >>> Y=problem.evaluate(X)
                >>> delta_method.analyze(X, Y)
                
            Reference:
                [1] E. Eirola et al, Using the Delta Test for Variable Selection, 
                                     Artificial Neural Networks, 2008.
        '''
        super().__init__(problem, scalers)

        self.n_neighbors=n_neighbors
        
    def sample(self, N: int=500, sampler: Sampler=LHS('classic')):
        '''
            Generate samples
            -------------------------------
            Parameters:
                N: int, default=500
                    N is corresponding to the use sampler 
                sampler: Sampler, default=LHS('classic')
            
            Returns:
                X: 2d-np.ndarray
                    the size is determined by the used sampler. Default: (N, n_input)            
        '''
        n_input=self.n_input
        
        X=sampler.sample(N, n_input)
        
        return X
    
    def analyze(self, X: np.ndarray=None, Y: np.ndarray=None, verbose: bool=False) -> dict:
        '''
            Perform Delta_Test
            -------------------------------------
            Parameters:
                X: np.ndarray
                    the input data
                Y: np.ndarray
                    the result data
                verbose: bool 
                    the switch to print analysis summary or not
            
            Returns:
                Si: dict
                    The type of Si is dict. It contains 'S1' and 'High Sensibility Parameters (HSP)'
        '''
        X, Y=self.__check_and_scale_xy__(X, Y)
        
        ##main process
        self.X_=X; self.Y_=Y
        optimizer=Binary_GA(self._cal_delta, self.n_input)
        best_paras, self.best_value, history_paras, _=optimizer.run()
        
        S1_score=np.sum(history_paras, axis=0)/history_paras.shape[0]
        
        HSP_inds=[index for index, value in enumerate(best_paras) if value==1]
        HSP_paras=self.labels[HSP_inds]
        
        Si={'S1': S1_score, 'HSP':HSP_paras}
        self.Si=Si
        
        if verbose:
            self.summary()
        
        return Si
    
    def summary(self):
        '''
            print analysis summary
        '''
        if self.Si==None:
            raise ValueError("Please run analyze() first!")
        print('Delta_Test')
        print('-------------------------------')
        print('The sensibility for all parameters:')
        print(' |'.join(self.x_labels))
        print(' |'.join(map(str, self.Si['S1'])))
        print('-------------------------------')
        print('The best performance parameter combinations:')
        print(' |'.join(self.Si['HSP']))
         
        #TODO output the optimal variables
    #--------------------Private Function--------------------------#
    def _default_sample(self):
        return self.sample(500)
    
    def _cal_delta(self, exclude_feature_list):
        
        #TODO expensive computation so using pybind11 or cython to accelerate 
        X=np.copy(self.X_)
        y=np.copy(self.Y_)
        exclude_feature = [index for index, value in enumerate(exclude_feature_list) if value == 0]
        
        if exclude_feature is not None:
            X = np.delete(X, exclude_feature, axis=1)
            
        distances = cdist(X, X)
        np.fill_diagonal(distances, np.inf)
        
        neighbors_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        
        deltas = []
        for i in range(len(X)):
            neighbor_deltas = (y[i] - y[neighbors_indices[i]])**2
            delta = np.mean(neighbor_deltas)
            deltas.append(delta)
        
        return np.mean(deltas)     