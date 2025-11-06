from typing import Literal, Optional
import numpy as np
from scipy.spatial.distance import pdist

from .base import Sampler
from ..problem import ProblemABC as Problem

def _lhs_classic(nt: int, nx: int, rng):
    """
    Generate a classic Latin Hypercube Sampling (LHS) design.
    
    :param nt: Number of samples.
    :param nx: Number of dimensions.
    :param rng: Random state for reproducibility.
    :return: A 2D array of LHS samples.
    """

    # Generate the intervals
    cut = np.linspace(0, 1, nt + 1)
    
    # Fill points uniformly in each interval
    u = rng.random((nt, nx))
    a = cut[:nt]
    b = cut[1:nt + 1]
    rdpoints = np.zeros_like(u)
    for j in range(nx):
        rdpoints[:, j] = u[:, j] * (b - a) + a
    
    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(nx):
        order = rng.permutation(range(nt))
        H[:, j] = rdpoints[order, j]
    
    return H
    
def _lhs_centered(nt: int, nx: int, rng):
    """
    Generate a centered Latin Hypercube Sampling (LHS) design.
    
    :param nt: Number of samples.
    :param nx: Number of dimensions.
    :param rng: Random state for reproducibility.
    :return: A 2D array of centered LHS samples.
    """

    # Generate the intervals
    cut = np.linspace(0, 1, nt + 1)    
    
    # Fill points uniformly in each interval
    u = rng.random(nt, nx)
    a = cut[:nt]
    b = cut[1:nt + 1]
    _center = (a + b)/2
    
    # Make the random pairings
    H = np.zeros_like(u)
    for j in range(nx):
        H[:, j] = rng.permutation(_center)
    
    return H
    
def _lhs_maximin(nt: int, nx: int, iterations: int, rng):
    """
    Generate a maximin Latin Hypercube Sampling (LHS) design.
    
    :param nt: Number of samples.
    :param nx: Number of dimensions.
    :param iterations: Number of iterations to maximize the minimum distance.
    :param rng: Random state for reproducibility.
    :return: A 2D array of maximin LHS samples.
    """
     
    maxdist = 0
    
    # Maximize the minimum distance between points
    for i in range(iterations):

        H_candidate = _lhs_classic(nt, nx, rng)

        d = pdist(H_candidate,'euclidean')
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = H_candidate.copy()
    
    return H

def _lhs_centered_maximin(nt: int, nx: int, iterations: int, rng):
    """
    Generate a centered maximin Latin Hypercube Sampling (LHS) design.
    
    :param nt: Number of samples.
    :param nx: Number of dimensions.
    :param iterations: Number of iterations to maximize the minimum distance.
    :param rng: Random state for reproducibility.
    :return: A 2D array of centered maximin LHS samples.
    """

    maxdist = 0
    
    # Maximize the minimum distance between points
    for i in range(iterations):

        H_candidate = _lhs_centered(nt, nx, rng)
        d = pdist(H_candidate,'euclidean')
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = H_candidate.copy()
    
    return H
################################################################################

def _lhs_correlate(nt: int, nx: int, iterations: int, rng = None):
    """
    Generate a correlation-optimized Latin Hypercube Sampling (LHS) design.
    
    :param nt: Number of samples.
    :param nx: Number of dimensions.
    :param iterations: Number of iterations to minimize correlation.
    :param rng: Random state for reproducibility.
    :return: A 2D array of correlation-optimized LHS samples.
    """
    
    mincorr = np.inf
    
    # Minimize the components correlation coefficients
    for _ in range(iterations):
        # Generate a random LHS
        H_candidate = _lhs_classic(nt, nx, rng)
        R = np.corrcoef(H_candidate)
        if np.max(np.abs(R[R!=1]))<mincorr:
            mincorr = np.max(np.abs(R-np.eye(R.shape[0])))
            print('new candidate solution found with max,abs corrcoef = {}'.format(mincorr))
            H = H_candidate.copy()

    return H

Criterion = Literal['classic','center','maximin','center_maximin','correlation']
LHS_METHOD = {'classic': _lhs_classic, 'center': _lhs_centered, 'maximin': _lhs_maximin,
             'center_maximin': _lhs_centered_maximin, 'correlation': _lhs_correlate}

class LHS(Sampler):
    """
    Latin-hypercube design class for generating samples.
    
    Methods:
        sample: Generate a Latin-hypercube design
    
            
    """
    def __init__(self, criterion: Criterion ='classic', iterations = 5):
        """
        Initialize the LHS sampler with a specified criterion and number of iterations.
        
        :param criterion: The LHS criterion to use.
        :param iterations: Number of iterations for optimization methods.
        """

        self.criterion = criterion
        self.iterations = iterations
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
            xInit = Sampling_method(nt, nx, self.iterations, self.rng)
        else:
            xInit = Sampling_method(nt, nx, self.rng)
        
        return xInit
    
    def sample(self, problem: Problem, nt: int, seed: Optional[int] = None):
        """
        Generate a Latin-hypercube design.
        
        :param nt: Number of sampled points.
        :param nx: Input dimensions of sampled points.
        :param problem: Problem instance to use bounds for sampling.
        :param seed: Random seed for reproducibility.
        :return: A 2D array of LHS samples.
        """
        
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        nx = problem.nInput
        
        return problem._transform_unit_X(self._generate(nt, nx))