from typing import Callable
from .Problem_ABC import ProblemABC
import numpy as np
class PracticalProblem(ProblemABC):
    '''
    Class for creating practical problem
    '''
    def __init__(self):
        super().__init__()
        self._func=None
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
        