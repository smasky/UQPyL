import numpy as np
from typing import Optional

from .samplerABC import Sampler, decoratorRescale
from ..problems import ProblemABC as Problem

class Morris_Sequence(Sampler):
    '''
    The sample technique for Morris analysis
    
    Parameters:
        num_levels (p): each x_i would take value on {0, 1/(p-1), 2/(p-1), ..., 1}. 
                        Morris et al[1]. recommend the num_levels to be even and range from 4 and 10.
    
    Methods:
        __call__ or sample: Generate a sample for FAST method
    
    Examples:
        >>> mor_seq=Morris_Sequence(num_levels=4)
        >>> mor_seq.sample(100, 4) or mor_seq(100, 4)
    
    Reference:
        [1] Max D. Morris (1991) Factorial Sampling Plans for Preliminary Computational Experiments, Technometrics, 33:2, 161-174
    '''
    def __init__(self, num_levels: int=4):
        
        super().__init__()
        self.num_levels=num_levels
        
    def _generate(self, nt: int, nx: int):
        '''
        Generate a shape of (nt*nx, nx) sample for FAST
        
        parameters:
        nt: int
            the number of trajectory
        nx: int
            the input dimensions of sampled points
            
        Returns:
        H: 2d-array
            An n-by-samples design matrix that has been normalized so factor values
            are uniformly spaced between zero and one.
        '''      
        xInit=np.zeros((nt*(nx+1), nx))
        
        for i in range(nt):
            xInit[i*(nx+1):(i+1)*(nx+1), :]=self._generate_trajectory(nx)
        
        return xInit
    
    @decoratorRescale
    def sample(self, nt: int, nx: Optional[int]=None, problem: Optional[Problem]=None, random_seed: Optional[int] = None):
        
        if random_seed is not None:
            self.random_state = np.random.RandomState(random_seed)
        else:
            self.random_state = np.random.RandomState()
        
        if problem is not None and nx is not None:
            if(problem.nInput!=nx):
                raise ValueError('The input dimensions of the problem and the samples must be the same')
        elif problem is None and nx is None:
            raise ValueError('Either the problem or the input dimensions must be provided')
        
        nx=problem.nInput if problem is not None else nx
        
        return self._generate(nt, nx)
        
    def _generate_trajectory(self, nx: int):
        
        delta=self.num_levels/(2*(self.num_levels-1))
        
        B=np.tril(np.ones([nx + 1, nx], dtype=int), -1)
        
        # from paper[1] page 164
        D_star = np.diag(np.random.choice([-1, 1], nx)) #step1
        J=np.ones((nx+1, nx))
        
        levels_grids=np.linspace(0, 1-delta, int(self.num_levels / 2))
        x_star=np.random.choice(levels_grids, nx).reshape(1,-1) #step2
        
        P_star=np.zeros((nx,nx))
        cols = np.random.choice(nx, nx, replace=False)
        P_star[np.arange(nx), cols]=1 #step3
        
        element_a = J[0, :] * x_star
        element_b = P_star.T
        element_c = np.matmul(2.0 * B, element_b)
        element_d = np.matmul((element_c - J), D_star)

        B_star = element_a + (delta / 2.0) * (element_d + J)
    
        return B_star
        
        
        
        
    