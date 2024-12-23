import sys
sys.path.append('.')
import numpy as np

from UQPyL.problems import Sphere
from UQPyL.DoE import LHS
from UQPyL.utility.metrics import r_square

problem=Sphere(nInput=10)
lhs=LHS('center')

#generate train data 
xTrain=lhs.sample(500, problem=problem)
yTrain=problem.evaluate(xTrain)

#generate test data
xTest=lhs.sample(50, problem=problem)
yTest=problem.evaluate(xTest)

#save to txt
# np.savetxt('xTest.txt', xTest); np.savetxt('yTest.txt', yTest.reshape(-1, 1))
# np.savetxt('xTrain.txt', xTrain); np.savetxt('yTrain.txt', yTrain.reshape(-1, 1))

#-------------------Kriging----------------------------#
# from UQPyL.surrogates.kriging import KRG
# from UQPyL.surrogates.kriging.kernel import Guass, Cubic, Exp
# from UQPyL.utility.scalers import MinMaxScaler, StandardScaler
# from UQPyL.optimization import GA
# from time import time

# kernel=Guass(theta=1.0, heterogeneous=True)
## kernel=Cubic(theta=1.0, heterogeneous=True)
## kernel=Exp(theta=1.0, heterogeneous=True)

# optimizer = GA(maxFEs=1000, nPop=50)

# #use Boxmin
# krg=KRG(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)), kernel=kernel, optimizer='Boxmin', n_restart_optimize=0, fitMode='likelihood')

# #use optimization
# #krg=KRG(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)), kernel=kernel, optimizer=optimizer, n_restart_optimize=0, fitMode='likelihood')

# krg.fit(xTrain, yTrain)
# yPred=krg.predict(xTest)
# value=r_square(yTest, yPred)
# print(value)

#-------------------Gaussian Process---------------------#
# from UQPyL.surrogates.gp import GPR
# from UQPyL.optimization.single_objective import GA
# from UQPyL.surrogates.gp.kernel import RBF, Matern, RationalQuadratic

# optimizer = GA(maxFEs=2000, nPop=50)
# # kernel = RBF(length_scale = 1.0, heterogeneous=True)
# # kernel = Matern(length_scale = 1.0, optimize_nu = True, heterogeneous=True)
# kernel = RationalQuadratic(length_scale=1.0, alpha=1.0, heterogeneous=True)
# gpr = GPR(kernel=kernel, optimizer=optimizer, fitMode='predictError')

# gpr.fit(xTrain, yTrain)
# yPred = gpr.predict(xTest)
# value = r_square(yTest, yPred)
# print(value)
# print(kernel.getPara("l"))


# ##
#-------------------Linear regression-----------------#
# from UQPyL.surrogates.regression import LinearRegression
# from UQPyL.utility.polynomial_features import PolynomialFeatures
# lr=LinearRegression(polyFeature=PolynomialFeatures(degree=2), lossType='Lasso', fitIntercept=True)
# lr.fit(xTrain, yTrain)
# yPred=lr.predict(xTest)
# value=r_square(yTest, yPred)
# print(value)

#-------------------Polynomial regression-----------------#
# from UQPyL.surrogates.regression import PolynomialRegression
# pr=PolynomialRegression(degree=2, lossType='Lasso', fitIntercept=True)
# pr.fit(xTrain, yTrain)
# yPred=pr.predict(xTest)
# value=r_square(yTest, yPred)
# print(value)

#-------------------Support vector regression---------------#
# from UQPyL.surrogates.svr import SVR
# from UQPyL.utility.polynomial_features import PolynomialFeatures
# svr=SVR(kernel='sigmoid')
# svr.fit(xTrain, yTrain)
# yPred=svr.predict(xTest)
# value=r_square(yTest, yPred)
# print(value)

#-------------------Radial Basis Functions----------------#
# from UQPyL.surrogates.rbf import RBF
# from UQPyL.surrogates.rbf.kernel import Cubic, Multiquadric

# rbf=RBF(kernel=Cubic())
# rbf.fit(xTrain, yTrain)
# yPred=rbf.predict(xTest)
# value=r_square(yTest, yPred)

# y=rbf.predict(np.zeros(10).reshape(1,-1))
# print(value)







