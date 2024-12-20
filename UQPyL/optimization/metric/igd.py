import numpy as np
from scipy.spatial.distance import cdist

def IGD(pop, optimum):
    
    popObjs = pop.getBest().objs
    
    distances = cdist(optimum, popObjs, metric='euclidean')
    
    minDist = np.min(distances, axis=1)
    
    igd = np.mean(minDist)
    
    return igd