import numpy as np
import xarray as xr


from UQPyL.inference import MH, MH_Gibbs, AMH, DEMC, DREAM_ZS
from UQPyL.problem import Problem

# ------------------------------- #
#      Gaussian Mixture 2D        #
# ------------------------------- #

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

gauss = Problem(nInput = 2, nOutput = 1, objFunc = objFunc, ub = 10, lb = -10)

# ------------------------------- #
#      Gaussian Mixture 4D        #
# ------------------------------- #

# def objFunc(X):

#     X = np.atleast_2d(X)
#     _, D = X.shape

#     mu1 = np.array([-5.0, 0.0, 0.0, 0.0])
#     mu2 = np.array([ 5.0, 0.0, 0.0, 0.0])

#     Sigma1 = np.eye(D)
#     Sigma2 = np.eye(D)

#     inv1 = np.linalg.inv(Sigma1)
#     inv2 = np.linalg.inv(Sigma2)

#     e1 = 0.5 * np.sum((X - mu1) @ inv1 * (X - mu1), axis=1)
#     e2 = 0.5 * np.sum((X - mu2) @ inv2 * (X - mu2), axis=1)

#     f = -np.log(0.5 * np.exp(-e1) + 0.5 * np.exp(-e2))

#     return f[:, np.newaxis]

# gauss4 = Problem(nInput = 4, nOutput = 1, objFunc = objFunc, ub = 10, lb = -10, name = "Gaussian4")

# ------------------------------------------- #
#             Metropolis-Hastings             # 
# ------------------------------------------- #

# from UQPyL.inference import MH

# mh = MH(nChains = 10, warmUp = 10, maxIters = 1000, propDist = 'gauss', 
#             verboseFlag = True, verboseFreq = 1)

# res = mh.run(problem = gauss)

# print(res['posterior'].info)
# print(res['stats'].info)
# print(res['optimization'].info)

# ------------------------------------------- #
#                  MH-Gibbs                   # 
# ------------------------------------------- #

# from UQPyL.inference import MH_Gibbs

# mh_gibbs = MH_Gibbs(nChains = 10, warmUp = 100, maxIters = 1000, propDist = 'gauss', verboseFlag = True, verboseFreq = 1)

# res = mh_gibbs.run(problem = guass)

# ------------------------------------------- #
#         Adaptive Metropolis-Hastings        # 
# ------------------------------------------- #

# from UQPyL.inference import AMH

# amh = AMH(nChains = 10, warmUp = 100, maxIters = 1000, propDist = 'gauss', verboseFlag = True, verboseFreq = 1)

# res = amh.run(problem = guass)

# ------------------------------------------- #
#     Differential Evolution Markov Chain     # 
# ------------------------------------------- #

# from UQPyL.inference import DEMC

# demc = DEMC(nChains = 10, warmUp = 100, maxIterTimes = 1000, verboseFlag = True, verboseFreq = 1)

# res = demc.run(problem = guass)

# ------------------------------------------- #
#                 DREAM-ZS                    # 
# ------------------------------------------- #

# from UQPyL.inference import DREAM_ZS

# dream_zs = DREAM_ZS(nChains = 10, warmUp = 100, maxIter = 1000, verboseFlag = True, verboseFreq = 1)

# res = dream_zs.run(problem = guass)