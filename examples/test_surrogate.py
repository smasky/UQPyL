import sys
sys.path.insert(0, '.')

import numpy as np
from UQPyL.problems import Sphere

from UQPyL.utility.scalers import MinMaxScaler
from UQPyL.DoE import LHS
from UQPyL.utility.metrics import r_square

lhs = LHS()

problem = Sphere(nInput = 15)

X = lhs.sample(nt = 200, problem = problem)
Y = problem.objFunc(X)

XTest = lhs.sample(nt = 100, problem = problem)
YTest = problem.objFunc(XTest)

# ------------------------------------------- #
#                   Kriging                   # 
# ------------------------------------------- #


# from UQPyL.surrogates.kriging import KRG
# from UQPyL.surrogates.kriging.kernel import Guass

# kernel = Guass(heterogeneous = False)
# krg = KRG(scalers=(MinMaxScaler(0, 1), MinMaxScaler(0, 1)), kernel=kernel)
# krg.fit(X, Y)

# YPred = krg.predict(XTest)
 
# r2 = r_square(YTest, YPred)

# print(r2)

#-------------------------------------------#
#            Gaussian Process               #
#-------------------------------------------#

# from UQPyL.surrogates.gp import GPR
# from UQPyL.surrogates.gp.kernel import RBF, Matern
# from UQPyL.optimization.single_objective import GA, PSO
# from UQPyL.surrogates.auto_tuner import AutoTuner
# ga = GA(maxFEs = 5000)
# pso = PSO(maxFEs = 5000)
# kernel = Matern(length_scale= 10.0, nu = 1.5, optimize_nu=True, heterogeneous=True)
# gpr = GPR(kernel = kernel)

# nameList = gpr.getParaList()
# auto_tuner = AutoTuner(optimizer = pso, model = gpr)
# auto_tuner.optTune(X, Y, nameList)
# gpr.fit(X, Y)

# YPred = gpr.predict(XTest)

# r2 = r_square(YTest, YPred)

# print(r2)

# ------------------------------------------- #
#              Radial Basis Function          #
# ------------------------------------------- #

# from UQPyL.surrogates.rbf import RBF
# from UQPyL.surrogates.rbf.kernel import Cubic
# from UQPyL.surrogates.auto_tuner import AutoTuner
# from UQPyL.optimization.single_objective import PSO
# kernel = Cubic()
# rbf = RBF(kernel = kernel)  

# nameList = rbf.getParaList()
# pso = PSO(maxFEs = 5000)
# auto_tuner = AutoTuner(optimizer = pso, model = rbf)
# auto_tuner.optTune(X, Y, nameList)

# # rbf.fit(X, Y)   
# YPred = rbf.predict(XTest)

# r2 = r_square(YTest, YPred)

# print(r2)

# ------------------------------------------- #
# Linear Regression and Polynomial Regression #
# ------------------------------------------- #

# from UQPyL.surrogates.regression import LinearRegression
# from UQPyL.surrogates.regression import PolynomialRegression
# from UQPyL.surrogates.auto_tuner import AutoTuner
# from UQPyL.optimization.single_objective import PSO

# lr = LinearRegression(lossType = 'Lasso', C = 10, C_attr = {'ub': 100, 'lb': 1e-5, 'type': 'float', 'log': True})
# pr = PolynomialRegression(degree = 2, lossType = 'Lasso', C = 1e-5, C_attr = {'ub': 100, 'lb': 1e-5, 'type': 'float', 'log': True})

# nameList = lr.getParaList()
# pso = PSO(maxFEs = 5000)
# auto_tuner = AutoTuner(optimizer = pso, model = pr)
# auto_tuner.optTune(X, Y, nameList)

# lr.fit(X, Y)
# pr.fit(X, Y)

# YPred = lr.predict(XTest)
# YPred = pr.predict(XTest)

# r2 = r_square(YTest, YPred)

# print(r2)

# ------------------------------------------- #
#              Support Vector Regression      #
# ------------------------------------------- #

# from UQPyL.surrogates.svr import SVR
# from UQPyL.surrogates.auto_tuner import AutoTuner
# from UQPyL.optimization.single_objective import PSO

# svr = SVR(kernel = 'rbf')

# nameList = svr.getParaList()
# pso = PSO(maxFEs = 5000)
# auto_tuner = AutoTuner(optimizer = pso, model = svr)
# auto_tuner.optTune(X, Y, nameList)

# nameDict = {'C' : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], 'epsilon' : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}
# auto_tuner = AutoTuner(model = svr)
# auto_tuner.gridTune(X, Y, nameDict)

# svr.fit(X, Y)

# YPred = svr.predict(XTest)

# r2 = r_square(YTest, YPred)

# print(r2)

# ------------------------------------------- #
#  Multivariate Adaptive Regression Splines   #
# ------------------------------------------- #

# from UQPyL.surrogates.mars import MARS

# mars = MARS(max_degree = 2)

# mars.fit(X, Y)

# YPred = mars.predict(XTest)

# r2 = r_square(YTest, YPred)

# print(r2)


