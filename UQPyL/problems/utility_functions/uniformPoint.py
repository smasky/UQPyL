import numpy as np
from typing import Literal
from scipy.special import comb
from itertools import combinations
from numpy import linspace, meshgrid, hstack, ceil

def _grid(N: int, M: int):
    
    gap = np.linspace(0, 1, int(np.ceil(N ** (1 / M))))

    c = [np.copy(gap) for _ in range(M)]
    c_grid = np.meshgrid(*c, indexing='ij')

    W = np.hstack([c_grid[i].flatten()[:, np.newaxis] for i in range(M)])
    N=W.shape[0]
    return W, N

def _NBI(N: int, M:int):
    
    H1 = 1
    while comb(H1 + M, M - 1) <= N:
        H1 += 1
    
    W = np.array(list(combinations(range(H1 + M - 1), M - 1))) - np.tile(np.arange(M - 1), (comb(H1 + M - 1, M - 1).astype(int), 1)) - 1
    W = (np.hstack([W, np.zeros((W.shape[0], 1)) + H1]) - np.hstack([np.zeros((W.shape[0], 1)), W])) / H1
    
    if H1 < M:
        H2 = 0
        while comb(H1 + M - 1, M - 1) + comb(H2 + M, M - 1) <= N:
            H2 += 1
        
        if H2 > 0:
            W2 = np.array(list(combinations(range(H2 + M - 1), M - 1))) - np.tile(np.arange(M - 1), (comb(H2 + M - 1, M - 1).astype(int), 1)) - 1
            W2 = (np.hstack([W2, np.zeros((W2.shape[0], 1)) + H2]) - np.hstack([np.zeros((W2.shape[0], 1)), W2])) / H2
            W = np.vstack([W, W2 / 2 + 1 / (2 * M)])
    
    W = np.maximum(W, 1e-6)
    N = W.shape[0]
    
    return W, N
    

def uniformPoint(N: int, M: int, method: Literal['NBI', 'grid']='NBI'):

    if method=='NBI':
        return _NBI(N, M)
    elif method=='grid':
        return _grid(N, M)
    