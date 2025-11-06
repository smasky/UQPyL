import sys
sys.path.insert(0, '.')
import numpy as np
import xarray as xr


from UQPyL.inference import MH, MH_Gibbs, AMH, DEMC, DREAM_ZS
from UQPyL.problem import Problem

def objFunc(X):
    
    X = np.atleast_2d(X)
    _, D = X.shape

    mu1 = np.array([-5.0, 0.0])
    mu2 = np.array([ 5.0, 0.0])
    Sigma1 = np.eye(D)
    Sigma2 = np.eye(D)

    inv1 = np.linalg.inv(Sigma1)
    inv2 = np.linalg.inv(Sigma2)

    e1 = 0.5 * np.sum((X - mu1) @ inv1 * (X - mu1), axis=1)
    e2 = 0.5 * np.sum((X - mu2) @ inv2 * (X - mu2), axis=1)

    f = -np.log(0.5 * np.exp(-e1) + 0.5 * np.exp(-e2))
    
    return f[:, np.newaxis]

guass = Problem(nInput = 2, nOutput = 1, objFunc = objFunc, ub = 10, lb = -10)

# mh = MH(nChain = 10, warmUp = 10, maxIter = 1000, propDist = 'gauss', verboseFlag = True, verboseFreq = 1)

# res = mh.run(problem = guass)

# mh_gibbs = MH_Gibbs(nChain = 10, warmUp = 100, maxIter = 1000, propDist = 'gauss', verboseFlag = True, verboseFreq = 1)

# res = mh_gibbs.run(problem = guass)

# amh = AMH(nChain = 10, warmUp = 100, maxIterTimes = 1000, propDist = 'gauss', verboseFlag = True, verboseFreq = 1)

# res = amh.run(problem = guass)

# demc = DEMC(nChain = 10, warmUp = 100, maxIterTimes = 1000, verboseFlag = True, verboseFreq = 1)

# res = demc.run(problem = guass)

# print(res['stats']['acceptanceRate_mean'])

# dream_zs = DREAM_ZS(nChain = 10, warmUp = 100, maxIter = 1000, verboseFlag = True, verboseFreq = 1)

# res = dream_zs.run(problem = guass)

# print(res['stats'].info)

