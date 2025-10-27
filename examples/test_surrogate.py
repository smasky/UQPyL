import sys
sys.path.insert(0, '.')

import numpy as np
from UQPyL.problem import Sphere

from UQPyL.util.scaler import MinMaxScaler
from UQPyL.doe import LHS
from UQPyL.util.metric import r_square

lhs = LHS()

problem = Sphere(nInput = 15)

X = lhs.sample(nt = 800, problem = problem)
Y = problem.objFunc(X)

XTest = lhs.sample(nt = 100, problem = problem)
YTest = problem.objFunc(XTest)

# --------------Ordinary Usage--------------- #

# ------------------------------------------- #
#                   Kriging                   # 
# ------------------------------------------- #

# from UQPyL.surrogate.kriging import KRG
# from UQPyL.surrogate.kriging.kernel import Guass

# kernel = Guass(heterogeneous = False)

# krg = KRG(scalers=(MinMaxScaler(0, 1), MinMaxScaler(0, 1)), kernel=kernel)

# krg.fit(X, Y)

# YPred = krg.predict(XTest)
 
# r2 = r_square(YTest, YPred)

# print(r2)

#-------------------------------------------#
#            Gaussian Process               #
#-------------------------------------------#

# from UQPyL.surrogate.gp import GPR
# from UQPyL.surrogate.gp.kernel import RBF, Matern
# from UQPyL.surrogate.auto_tuner import AutoTuner
# from UQPyL.optimization.soea import PSO

# kernel = RBF(length_scale = 10.0, heterogeneous = False)

# gpr = GPR(kernel = kernel)

# gpr.fit(X, Y)

# YPred = gpr.predict(XTest)

# r2 = r_square(YTest, YPred)

# print(r2)

# ------------------------------------------- #
#              Radial Basis Function          #
# ------------------------------------------- #

# from UQPyL.surrogate.rbf import RBF
# from UQPyL.surrogate.rbf.kernel import Cubic

# kernel = Cubic()
# rbf = RBF(kernel = kernel)

# rbf.fit(X, Y)

# YPred = rbf.predict(XTest)
# r2 = r_square(YTest, YPred)

# print(r2)

# ------------------------------------------- #
#               Linear Regression             #
# ------------------------------------------- #

# from UQPyL.surrogate.regression import LinearRegression, PolynomialRegression

# lr = LinearRegression(lossType = 'Lasso', C = 10, C_attr = {'ub': 100, 'lb': 1e-5, 'type': 'float', 'log': True})

# lr.fit(X, Y)

# YPred = lr.predict(XTest)

# r2 = r_square(YTest, YPred)

# print(r2)

# ------------------------------------------- #
#              Polynomial Regression           #
# ------------------------------------------- #

# from UQPyL.surrogate.regression import PolynomialRegression

# pr = PolynomialRegression(degree = 2, lossType = 'Lasso', C = 1e-5, C_attr = {'ub': 100, 'lb': 1e-5, 'type': 'float', 'log': True})

# pr.fit(X, Y)

# YPred = pr.predict(XTest)

# r2 = r_square(YTest, YPred)

# print(r2)

# ------------------------------------------- #
#           Support Vector Regression         #
# ------------------------------------------- #

# from UQPyL.surrogate.svr import SVR

# svr = SVR(scalers=(MinMaxScaler(0, 1), MinMaxScaler(0, 1)), kernel = 'rbf')

# svr.fit(X, Y)

# YPred = svr.predict(XTest)

# r2 = r_square(YTest, YPred)

# print(r2)

# ------------------------------------------- #
#  Multivariate Adaptive Regression Splines   #
# ------------------------------------------- #

# from UQPyL.surrogate.mars import MARS

# mars = MARS(max_degree = 2)

# mars.fit(X, Y)

# YPred = mars.predict(XTest)

# r2 = r_square(YTest, YPred)

# print(r2)

# ---------Auto Tuning  OptTune------------ #

#-------------------------------------------#
#            Gaussian Process               #
#-------------------------------------------#

# from UQPyL.surrogate.gp import GPR
# from UQPyL.surrogate.gp.kernel import Matern
# from UQPyL.optimization.soea import PSO
# from UQPyL.surrogate.auto_tuner import AutoTuner


# kernel = Matern(length_scale = 10.0, nu = 1.5, optimize_nu = True, heterogeneous = True)
# gpr = GPR(kernel = kernel)

# nameList = gpr.getParaList()

# pso = PSO(maxFEs = 5000, verboseFlag = False, logFlag = False, saveFlag = False)

# auto_tuner = AutoTuner(optimizer = pso, model = gpr)

# auto_tuner.optTune(X, Y, nameList)

# gpr.fit(X, Y)

# YPred = gpr.predict(XTest)

# r2 = r_square(YTest, YPred)

# print(r2)


# ---------Auto Tuning  GridTune------------ #

#-------------------------------------------#
#           Radial Basis Function           #
#-------------------------------------------#

# from UQPyL.surrogate.rbf import RBF
# from UQPyL.surrogate.rbf.kernel import Cubic
# from UQPyL.surrogate.auto_tuner import AutoTuner

# kernel = Cubic()
# rbf = RBF(kernel = kernel)

# paraList = rbf.getParaList()

# paraGrid = {'C_smooth' : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], 'epsilon' : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}

# auto_tuner = AutoTuner(model = rbf)
# paraVals, bestObj = auto_tuner.gridTune(X, Y, paraGrid)

# rbf.fit(X, Y)
# YPred = rbf.predict(XTest)
# r2 = r_square(YTest, YPred)
# print(r2)