import numpy as np
from scipy.spatial.distance import pdist,squareform


def _lhs_classic(nt: int, nx: int) -> np.ndarray:
    # Generate the intervals
    cut = np.linspace(0, 1, nt + 1)    
    
    # Fill points uniformly in each interval
    u = np.random.rand(nt, nx)
    a = cut[:nt]
    b = cut[1:nt + 1]
    rdpoints = np.zeros_like(u)
    for j in range(nx):
        rdpoints[:, j] = u[:, j]*(b-a) + a
    
    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(nx):
        order = np.random.permutation(range(nt))
        H[:, j] = rdpoints[order, j]
    
    return H
    
################################################################################

def _lhs_centered(nt: int, nx: int) -> np.ndarray:
    # Generate the intervals
    cut = np.linspace(0, 1, nt + 1)    
    
    # Fill points uniformly in each interval
    u = np.random.rand(nt, nx)
    a = cut[:nt]
    b = cut[1:nt + 1]
    _center = (a + b)/2
    
    # Make the random pairings
    H = np.zeros_like(u)
    for j in range(nx):
        H[:, j] = np.random.permutation(_center)
    
    return H
    
################################################################################

def _lhs_maximin(nt: int, nx: int, iterations: int)-> np.ndarray:
    
    maxdist = 0
    
    # Maximize the minimum distance between points
    for i in range(iterations):

        H_candidate = _lhs_classic(nt, nx)

        d = pdist(H_candidate,'euclidean')
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = H_candidate.copy()
    
    return H

def _lhs_centered_maximin(nt: int, nx: int, iterations: int)-> np.ndarray:
    maxdist = 0
    
    # Maximize the minimum distance between points
    for i in range(iterations):

        H_candidate = _lhs_centered(nt, nx)
        d = pdist(H_candidate,'euclidean')
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = H_candidate.copy()
    
    return H
################################################################################

def _lhs_correlate(nt: int, nx: int, iterations: int) -> np.ndarray:
    mincorr = np.inf
    
    # Minimize the components correlation coefficients
    for i in range(iterations):
        # Generate a random LHS
        H_candidate = _lhs_classic(nt, nx)
        R = np.corrcoef(H_candidate)
        if np.max(np.abs(R[R!=1]))<mincorr:
            mincorr = np.max(np.abs(R-np.eye(R.shape[0])))
            print('new candidate solution found with max,abs corrcoef = {}'.format(mincorr))
            H = H_candidate.copy()

    return H