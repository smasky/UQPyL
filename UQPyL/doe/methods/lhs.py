from typing import Literal
import numpy as np
from scipy.spatial.distance import pdist

from ..base import Sampler

def _lhs_classic(nSamples: int, nInput: int, rng):
    """
    Generate a classic Latin Hypercube Sampling (LHS) design.
    
    :param nSamples: Number of samples.
    :param nInput: Number of input variables.
    :param rng: Random state for reproducibility.
    :return: A 2D array of LHS samples.
    """

    # Generate the intervals
    cut = np.linspace(0, 1, nSamples + 1)
    
    # Fill points uniformly in each interval
    u = rng.random((nSamples, nInput))
    a = cut[:nSamples]
    b = cut[1:nSamples + 1]
    rdpoints = np.zeros_like(u)
    for j in range(nInput):
        rdpoints[:, j] = u[:, j] * (b - a) + a
    
    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(nInput):
        order = rng.permutation(range(nSamples))
        H[:, j] = rdpoints[order, j]
    
    return H
    
def _lhs_centered(nSamples: int, nInput: int, rng):
    """
    Generate a centered Latin Hypercube Sampling (LHS) design.
    
    :param nSamples: Number of samples.
    :param nInput: Number of input variables.
    :param rng: Random state for reproducibility.
    :return: A 2D array of centered LHS samples.
    """

    # Generate the intervals
    cut = np.linspace(0, 1, nSamples + 1)    
    
    # Fill points uniformly in each interval
    u = rng.random((nSamples, nInput))
    a = cut[:nSamples]
    b = cut[1:nSamples + 1]
    _center = (a + b)/2
    
    # Make the random pairings
    H = np.zeros_like(u)
    for j in range(nInput):
        H[:, j] = rng.permutation(_center)
    
    return H
    
def _lhs_maximin(nSamples: int, nInput: int, iterations: int, rng):
    """
    Generate a maximin Latin Hypercube Sampling (LHS) design.
    
    :param nSamples: Number of samples.
    :param nInput: Number of input variables.
    :param iterations: Number of iterations to maximize the minimum distance.
    :param rng: Random state for reproducibility.
    :return: A 2D array of maximin LHS samples.
    """
     
    maxdist = 0
    
    # Maximize the minimum distance between points
    for i in range(iterations):

        H_candidate = _lhs_classic(nSamples, nInput, rng)

        d = pdist(H_candidate,'euclidean')
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = H_candidate.copy()
    
    return H

def _lhs_centered_maximin(nSamples: int, nInput: int, iterations: int, rng):
    """
    Generate a centered maximin Latin Hypercube Sampling (LHS) design.
    
    :param nSamples: Number of samples.
    :param nInput: Number of input variables.
    :param iterations: Number of iterations to maximize the minimum distance.
    :param rng: Random state for reproducibility.
    :return: A 2D array of centered maximin LHS samples.
    """

    maxdist = 0
    
    # Maximize the minimum distance between points
    for i in range(iterations):

        H_candidate = _lhs_centered(nSamples, nInput, rng)
        d = pdist(H_candidate,'euclidean')
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = H_candidate.copy()
    
    return H
################################################################################

def _lhs_correlate(nSamples: int, nInput: int, iterations: int, rng = None):
    """
    Generate a correlation-optimized Latin Hypercube Sampling (LHS) design.
    
    :param nSamples: Number of samples.
    :param nInput: Number of input variables.
    :param iterations: Number of iterations to minimize correlation.
    :param rng: Random state for reproducibility.
    :return: A 2D array of correlation-optimized LHS samples.
    """
    
    mincorr = np.inf
    
    # Minimize the components correlation coefficients
    for _ in range(iterations):
        # Generate a random LHS
        H_candidate = _lhs_classic(nSamples, nInput, rng)
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
    Latin hypercube sampler.

    Examples:
        >>> from UQPyL.problems import Sphere
        >>> problem = Sphere(nInput=3)
        >>> sampler = LHS(criterion="maximin", iterations=10)
        >>> X, meta = sampler.sampleWithMeta(problem, 20, seed=123)
        >>> print(X.shape)
        (20, 3)
        >>> print(meta["designType"])
        lhs

    References:
        [1] M. D. McKay, R. J. Beckman, and W. J. Conover, A Comparison of Three Methods
            for Selecting Values of Input Variables in the Analysis of Output from a
            Computer Code, Technometrics, 21(2):239-245, 1979,
            doi: 10.1080/00401706.1979.10489755.
        [2] M. Stein, Large Sample Properties of Simulations Using Latin Hypercube Sampling,
            Technometrics, 29(2):143-151, 1987, doi: 10.1080/00401706.1987.10488205.
    """
    def __init__(self, criterion: Criterion ='classic', iterations = 5):
        """
        Initialize the LHS sampler.

        :param criterion: LHS criterion.
        :param iterations: Iterations for optimized criteria.
        """

        self.criterion = criterion
        self.iterations = iterations
        super().__init__()

    def sample(self, problem, nSamples: int = None, seed=None, nt: int = None):
        return super().sample(problem, nSamples, seed=seed, nt=nt)

    def sampleWithMeta(self, problem, nSamples: int = None, seed=None, nt: int = None):
        return super().sampleWithMeta(problem, nSamples, seed=seed, nt=nt)
        
    def _generate(self, nSamples: int, nInput: int = None):
        """
        Generate unit-space LHS samples.

        :param nSamples: Number of samples.
        :param nInput: Number of input variables.
        :return np.ndarray: Unit-space LHS samples.
        """
        
        if self.criterion not in LHS_METHOD:
            raise ValueError('The criterion must be one of {}'.format(LHS_METHOD.keys()))
        
        Sampling_method = LHS_METHOD[self.criterion]
        
        if self.criterion in ['maximin', 'center_maximin', 'correlation']:
            xInit = Sampling_method(nSamples, nInput, self.iterations, self.rng)
        else:
            xInit = Sampling_method(nSamples, nInput, self.rng)
        
        return xInit

    def _build_meta(self, problem, nSamples: int, seed=None):
        return {
            "designType": "lhs",
            "criterion": self.criterion,
            "iterations": self.iterations,
            "seed": seed,
        }
    
