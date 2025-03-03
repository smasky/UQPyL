import abc
import numpy as np

from ..problems import ProblemABC as Problem

def decoratorRescale(func):
    def wrapper(self, *args, **kwargs):
        result=func(self, *args, **kwargs)
        
        if len(args)>=3:
            
            problem=args[2]
            return problem._transform_unit_X(result)
        
        if 'problem' in kwargs:
            problem=kwargs['problem']
            
            if problem:
                return problem._transform_unit_X(result)

        return result
    return wrapper
class Sampler(metaclass=abc.ABCMeta):
    
    def __init__(self):
        
        self.random_state=np.random.RandomState()
    
    #Abandon
    # def __call__(self, nt:int, nx: int) -> np.ndarray:
    #     return self._generate(nt, nx)
    
    # def sample(self, nt:int, nx:int) -> np.ndarray:
    #     return self._generate(nt, nx)
    
    # def rescale_to_problem(self, xInit:np.ndarray):
        
    #     if self.problem is not None:
    #         xInit=self.problem._unit_xInit_transform_to_bound(xInit)
            
    #     return xInit
    
    def _generate(self, nt: int, nx: int):
        '''
        nt: the number of sampled points
        nx: the dimensions of decision variables
        
        return:
            ndarry[nt,nx]
        
        '''
        pass

