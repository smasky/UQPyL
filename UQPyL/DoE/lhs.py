import numpy as np
from typing import Literal, Optional
from scipy.spatial.distance import pdist

from ..problems import Problem

from .sampler_ABC import Sampler

# from ._lhs import _lhs_classic, _lhs_centered, _lhs_correlate, _lhs_maximin, _lhs_centered_maximin



def _lhs_classic(nt: int, nx: int, random_state=None) -> np.ndarray:
    # Generate the intervals
    if random_state is None:
        random_state=np.random.RandomState()
    cut = np.linspace(0, 1, nt + 1)    
    
    # Fill points uniformly in each interval
    u = random_state.rand(nt, nx)
    a = cut[:nt]
    b = cut[1:nt + 1]
    rdpoints = np.zeros_like(u)
    for j in range(nx):
        rdpoints[:, j] = u[:, j]*(b-a) + a
    
    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(nx):
        order = random_state.permutation(range(nt))
        H[:, j] = rdpoints[order, j]
    
    return H
    
################################################################################

def _lhs_centered(nt: int, nx: int, random_state=None) -> np.ndarray:
    
    if random_state is None:
        random_state=np.random.RandomState()
    
    # Generate the intervals
    cut = np.linspace(0, 1, nt + 1)    
    
    # Fill points uniformly in each interval
    u = random_state.rand(nt, nx)
    a = cut[:nt]
    b = cut[1:nt + 1]
    _center = (a + b)/2
    
    # Make the random pairings
    H = np.zeros_like(u)
    for j in range(nx):
        H[:, j] = random_state.permutation(_center)
    
    return H
    
################################################################################

def _lhs_maximin(nt: int, nx: int, iterations: int, random_state=None)-> np.ndarray:
    
    if random_state is None:
        random_state=np.random.RandomState()
        
    maxdist = 0
    
    # Maximize the minimum distance between points
    for i in range(iterations):

        H_candidate = _lhs_classic(nt, nx, random_state)

        d = pdist(H_candidate,'euclidean')
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = H_candidate.copy()
    
    return H

def _lhs_centered_maximin(nt: int, nx: int, iterations: int, random_state=None)-> np.ndarray:
    
    if random_state is None:
        random_state=np.random.RandomState()
    
    maxdist = 0
    
    # Maximize the minimum distance between points
    for i in range(iterations):

        H_candidate = _lhs_centered(nt, nx, random_state)
        d = pdist(H_candidate,'euclidean')
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = H_candidate.copy()
    
    return H
################################################################################

def _lhs_correlate(nt: int, nx: int, iterations: int, random_state=None) -> np.ndarray:
    
    if random_state is None:
        random_state=np.random.RandomState()
    
    mincorr = np.inf
    
    # Minimize the components correlation coefficients
    for _ in range(iterations):
        # Generate a random LHS
        H_candidate = _lhs_classic(nt, nx, random_state)
        R = np.corrcoef(H_candidate)
        if np.max(np.abs(R[R!=1]))<mincorr:
            mincorr = np.max(np.abs(R-np.eye(R.shape[0])))
            print('new candidate solution found with max,abs corrcoef = {}'.format(mincorr))
            H = H_candidate.copy()

    return H

Criterion=Literal['classic','center','maximin','center_maximin','correlation']
LHS_METHOD={'classic': _lhs_classic, 'center': _lhs_centered, 'maximin': _lhs_maximin,
             'center_maximin': _lhs_centered_maximin, 'correlation': _lhs_correlate}

class LHS(Sampler):
    '''
    Latin-hypercube design
    
    Parameters:
    criterion : str
        Allowable values are "classic", "center", "maximin", "center_maximin", 
        and "correlation". (Default: classic)
        
    iterations : int
        The number of iterations in the maximin, center_maximin and correlations methods
        (Default: 5).
    
    problem : problem
        if the problem is provided, the bounds of the problem will be used to generate the samples
    
    Methods:
    __call__ or sample: Generate a Latin-hypercube design
        
    Examples:
        >>>lhs=LHS('classic')
        >>>samples=lhs(5,10) or samples=lhs.sample(5,10)
    
    '''
    def __init__(self, criterion: Criterion ='classic', iterations: int=5, problem: Optional[Problem]=None):
        
        self.criterion=criterion
        self.iterations=iterations
        self.problem=problem
        
        #initial random state
        self.random_state=np.random.RandomState()
        
    def _generate(self, nt: int, nx: int) -> np.ndarray:
        '''
        Generate a Latin-hypercube design
        
        Parameters 
        nt: int
            the number of sampled points
        nx: int
            the input dimensions of sampled points
            
        Returns:
        H: 2d-array
            An n-by-samples design matrix that has been normalized so factor values
            are uniformly spaced between zero and one.
        '''
        
        if self.problem is not None:
            if(self.problem.n_input!=nx):
                raise ValueError('The input dimensions of the problem and the samples must be the same')
        
        if self.criterion not in LHS_METHOD:
            raise ValueError('The criterion must be one of {}'.format(LHS_METHOD.keys()))
        
        Sampling_method=LHS_METHOD[self.criterion]
        
        if self.criterion in ['maximin', 'center_maximin', 'correlation']:
            X=Sampling_method(nt, nx, self.iterations, self.random_state)
        else:
            X=Sampling_method(nt, nx, self.random_state)
        
        #rescale the samples
        # if self.problem is not None:
        #     X=X*(self.problem.ub-self.problem.lb)+self.problem.lb
        X=self.rescale_to_problems(X)  
        return X
    
    def sample(self, nt: int, nx:int, random_seed: Optional[int]=None) -> np.ndarray:
        '''
        Generate a Latin-hypercube design
        
        Parameters 
        nt: int
            the number of sampled points
            
        nx: int
            the input dimensions of sampled points
            
        random_seed: int
            the random seed for the random number generator
            
        Returns:
        H: 2d-array
            An n-by-samples design matrix that has been normalized so factor values
            are uniformly spaced between zero and one.
        '''
        
        if random_seed is not None:
            self.random_state = np.random.RandomState(random_seed)
        else:
            self.random_state = np.random.RandomState()
        
        return self._generate(nt, nx)