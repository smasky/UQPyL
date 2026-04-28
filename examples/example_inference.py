import numpy as np
import xarray as xr


from UQPyL.inference import MH, MH_Gibbs, AMH, DEMC, DREAM_ZS
from UQPyL.problem import Problem

# ------------------------------- #
#      Gaussian Mixture 2D        #
# ------------------------------- #

# def objFunc(X):
    
#     X = np.atleast_2d(X)
#     _, D = X.shape

#     mu1 = np.array([-5.0, 0.0])
#     mu2 = np.array([ 5.0, 0.0])
#     Sigma1 = np.eye(D)
#     Sigma2 = np.eye(D)

#     inv1 = np.linalg.inv(Sigma1)
#     inv2 = np.linalg.inv(Sigma2)

#     e1 = 0.5 * np.sum((X - mu1) @ inv1 * (X - mu1), axis=1)
#     e2 = 0.5 * np.sum((X - mu2) @ inv2 * (X - mu2), axis=1)

#     f = -np.log(0.5 * np.exp(-e1) + 0.5 * np.exp(-e2))
    
#     return f[:, np.newaxis]

# gauss = Problem(nInput = 2, nOutput = 1, objFunc = objFunc, ub = 10, lb = -10)


# def gauss5d_objFunc(X):
#     """
#     5 维中等相关 Gaussian distribution 的 objective function
#     输入: X (n_samples, 5)
#     输出: potential = -log p(x)（忽略常数项）

#     协方差矩阵设计：
#       - 对角方差温和差异（1, 2, 1, 2, 1）→ 尺度差仅 2 倍
#       - 相邻维度相关系数 ρ ≈ 0.5（中等相关）
#       - 隔一维相关系数 ρ ≈ 0.2
#       → MH 有一定困难但不至于崩溃，DREAM-ZS 仍有优势
#     """
#     X = np.atleast_2d(X)
#     _, D = X.shape

#     if D != 5:
#         raise ValueError("Gaussian 只支持 5 维")

#     # ====== 均值向量 ======
#     mu = np.array([1.0, -1.0, 0.5, 2.0, -0.5])

#     # ====== 中等相关协方差矩阵 ======
#     sigma = np.array([1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0])   # σ = (1, 1.41, 1, 1.41, 1)

#     corr = np.array([
#         [1.0,  0.5,  0.2,  0.0,  0.0],
#         [0.5,  1.0,  0.5,  0.2,  0.0],
#         [0.2,  0.5,  1.0,  0.5,  0.2],
#         [0.0,  0.2,  0.5,  1.0,  0.5],
#         [0.0,  0.0,  0.2,  0.5,  1.0],
#     ])

#     cov = np.diag(sigma) @ corr @ np.diag(sigma)
#     cov_inv = np.linalg.inv(cov)

#     diff = X - mu
#     f = 0.5 * np.sum(diff @ cov_inv * diff, axis=1)

#     return f[:, np.newaxis]


# 定义 Problem
# gauss5d = Problem(
#     nInput=5,
#     nOutput=1,
#     objFunc=gauss5d_objFunc,
#     ub=15.0,
#     lb=-15.0
# )


def banana_objFunc(X):
    """
    Banana-shaped distribution 的 objective function
    输入: X (n_samples, 2)，第0列=y1，第1列=y2
    输出: potential = -log p(y1,y2) （忽略常数项，MCMC 接受率中自动抵消）
    """
    X = np.atleast_2d(X)
    _, D = X.shape
    
    if D != 2:
        raise ValueError("Banana-shaped 只支持 2 维 (y1, y2)")
    
    y1 = X[:, 0]
    y2 = X[:, 1]
    
    # y1 ~ N(0, 100)   →  0.5 * y1² / 100
    # y2 | y1 ~ N(y1²/100, 1) →  0.5 * (y2 - y1²/100)²
    f = 0.5 * (y1 ** 2 / 100) + 0.5 * (y2 - y1**2 / 100) ** 2
    
    return f[:, np.newaxis]   # 必须是 (n_samples, 1) 格式

# 定义 Problem（和你的 gauss 完全一样格式）
banana = Problem(
    nInput=2,
    nOutput=1,
    objFunc=banana_objFunc,
    ub=30.0,      # y1 实际 99.7% 在 [-30,30] 内，y2 也覆盖
    lb=-30.0
)




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

# mh = MH(nChains = 4, warmUp = 100, maxIters = 10000, propDist = 'gauss', 
#             verboseFlag = True, verboseFreq = 1)

# res = mh.run(problem = banana)

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

from UQPyL.inference import AMH

amh = AMH(nChains = 4, warmUp = 100, maxIterTimes = 10000, propDist = 'gauss', verboseFlag = True, verboseFreq = 1)

res = amh.run(problem = banana)

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

# dream_zs = DREAM_ZS(nChains = 4, warmUp = 100, maxIters = 10000, verboseFlag = True, verboseFreq = 1)

# res = dream_zs.run(problem = banana)