from typing import Literal, Optional
import numpy as np
from scipy.spatial.distance import pdist

from .samplerABC import Sampler, decoratorRescale
from ..problems import ProblemABC as Problem

def _lhs_classic(nt: int, nx: int, random_state=None) -> np.ndarray:
    """
    Generate a classic Latin Hypercube Sampling (LHS) design.
    
    :param nt: Number of samples.
    :param nx: Number of dimensions.
    :param random_state: Random state for reproducibility.
    :return: A 2D array of LHS samples.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    # Generate the intervals
    cut = np.linspace(0, 1, nt + 1)
    
    # Fill points uniformly in each interval
    u = random_state.rand(nt, nx)
    a = cut[:nt]
    b = cut[1:nt + 1]
    rdpoints = np.zeros_like(u)
    for j in range(nx):
        rdpoints[:, j] = u[:, j] * (b - a) + a
    
    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(nx):
        order = random_state.permutation(range(nt))
        H[:, j] = rdpoints[order, j]
    
    return H
    
################################################################################

def _lhs_centered(nt: int, nx: int, random_state=None) -> np.ndarray:
    """
    Generate a centered Latin Hypercube Sampling (LHS) design.
    
    :param nt: Number of samples.
    :param nx: Number of dimensions.
    :param random_state: Random state for reproducibility.
    :return: A 2D array of centered LHS samples.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
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
    """
    Generate a maximin Latin Hypercube Sampling (LHS) design.
    
    :param nt: Number of samples.
    :param nx: Number of dimensions.
    :param iterations: Number of iterations to maximize the minimum distance.
    :param random_state: Random state for reproducibility.
    :return: A 2D array of maximin LHS samples.
    """
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
    """
    Generate a centered maximin Latin Hypercube Sampling (LHS) design.
    
    :param nt: Number of samples.
    :param nx: Number of dimensions.
    :param iterations: Number of iterations to maximize the minimum distance.
    :param random_state: Random state for reproducibility.
    :return: A 2D array of centered maximin LHS samples.
    """
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
    """
    Generate a correlation-optimized Latin Hypercube Sampling (LHS) design.
    
    :param nt: Number of samples.
    :param nx: Number of dimensions.
    :param iterations: Number of iterations to minimize correlation.
    :param random_state: Random state for reproducibility.
    :return: A 2D array of correlation-optimized LHS samples.
    """
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
    """
    Latin-hypercube design class for generating samples.
    
    Methods:
        sample: Generate a Latin-hypercube design
    
            
    """
    def __init__(self, criterion: Criterion ='classic', iterations: int=5):
        """
        Initialize the LHS sampler with a specified criterion and number of iterations.
        
        :param criterion: The LHS criterion to use.
        :param iterations: Number of iterations for optimization methods.
        """
        self.criterion=criterion
        self.iterations=iterations
        
        #initial random state
        super().__init__()
        
    def _generate(self, nt: int, nx: int = None):
        """
        Generate a Latin-hypercube design.
        
        :param nt: Number of sampled points.
        :param nx: Input dimensions of sampled points.
        :return: A 2D array of LHS samples.
        """
        
        if self.criterion not in LHS_METHOD:
            raise ValueError('The criterion must be one of {}'.format(LHS_METHOD.keys()))
        
        Sampling_method = LHS_METHOD[self.criterion]
        
        if self.criterion in ['maximin', 'center_maximin', 'correlation']:
            xInit = Sampling_method(nt, nx, self.iterations, self.random_state)
        else:
            xInit = Sampling_method(nt, nx, self.random_state)
        
        return xInit
    
    @decoratorRescale
    def sample(self, nt: int, nx: int = None, problem: Problem = None, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a Latin-hypercube design.
        
        :param nt: Number of sampled points.
        :param nx: Input dimensions of sampled points.
        :param problem: Problem instance to use bounds for sampling.
        :param random_seed: Random seed for reproducibility.
        :return: A 2D array of LHS samples.
        """
        
        if random_seed is not None:
            self.random_state = np.random.RandomState(random_seed)
        else:
            self.random_state = np.random.RandomState()
        
        if problem is not None and nx is not None:
            if problem.nInput != nx:
                raise ValueError('The input dimensions of the problem and the samples must be the same')
        elif problem is None and nx is None:
            raise ValueError('Either the problem or the input dimensions must be provided')
        
        nx = problem.nInput if problem is not None else nx
        
        return self._generate(nt, nx)