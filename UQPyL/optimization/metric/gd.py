import numpy as np
from scipy.spatial.distance import cdist

def GD(popObjs, optimum):
    
    distances = cdist(optimum, popObjs, metric='euclidean')
    
    minDist = np.min(distances, axis=1)
    
    gd = np.mean(minDist)
    
    return gd