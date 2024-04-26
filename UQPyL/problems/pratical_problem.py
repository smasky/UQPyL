import numpy as np
from typing import Callable

from .problem_ABC import ProblemABC
class Problem(ProblemABC):
    '''
    Class for creating practical problem
    '''
    def __init__(self, func, n_input, n_output, ub, lb):
        super().__init__(n_input, n_output, ub, lb)
        self._func=func
        
    def __check_func__(self,func):
        '''
        check the available of coupling function(algorithm and model)
        '''
        #TODO
        pass
        
    def set_func(self,func):
        
        if(self.__check_func__(func)):
            self._func=func
        
    def evaluate(self, X):
        return self._func(X)
        