import numpy as np
from scipy.spatial.distance import cdist

def IGD(popObjs, optimum):
    
    distances = cdist(optimum, popObjs, metric='euclidean')
    
    minDist = np.min(distances, axis=1)
    
    igd = np.mean(minDist)
    
    return igd