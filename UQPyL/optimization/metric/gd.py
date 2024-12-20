import numpy as np
from scipy.spatial.distance import cdist

def GD(pop, optimum):
    
    distances = cdist(pop, optimum, metric='euclidean')
    
    minDist = np.min(distances, axis=1)
    
    gd = np.mean(minDist)
    
    return gd