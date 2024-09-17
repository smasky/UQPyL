import sys
sys.path.append('.')
import numpy as np

from UQPyL.problems import Sphere
from UQPyL.DoE import LHS
from UQPyL.utility.metrics import r_square

problem=Sphere(nInput=10)
lhs=LHS('center', problem=problem)
#generate train data and prediction data
xTrain=lhs.sample(200, problem.nInput)
yTrain=problem.evaluate(xTrain)
#
xTest=lhs.sample(50, problem.nInput)
yTest=problem.evaluate(xTest)
# 
xTest=np.loadtxt('xTest.txt'); yTest=np.loadtxt('yTest.txt').reshape(-1, 1)
xTrain=np.loadtxt('xTrain.txt'); yTrain=np.loadtxt('yTrain.txt').reshape(-1, 1)
# np.savetxt('xTest.txt', xTest); np.savetxt('yTest.txt', yTest.reshape(-1, 1))
# np.savetxt('xTrain.txt', xTrain); np.savetxt('yTrain.txt', yTrain.reshape(-1, 1))
# #
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
# print(value)

#-------------------Gaussian Process---------------------#
# from UQPyL.surrogates.gp import GPR
# from UQPyL.optimization.single_objective import GA
# from UQPyL.optimization.mathematics import Boxmin
# from UQPyL.surrogates.gp.kernel import RBF, Matern
# gpr=GPR(kernel=RBF(length_scale=1, heterogeneous=True), optimizer=GA(maxFEs=1000, verboseFreq=1), fitMode="predictError")
# gpr.fit(xTrain, yTrain)
# yPred=gpr.predict(xTest)
# value=r_square(yTest, yPred)
# print(value)

#-------------------Kriging----------------------------#
from UQPyL.surrogates.kriging import KRG
from UQPyL.surrogates.kriging.kernel import Guass
from UQPyL.optimization.single_objective import GA
from UQPyL.optimization.mathematics import Boxmin
from UQPyL.utility.scalers import MinMaxScaler, StandardScaler
krg=KRG(scalers=(StandardScaler(0,1), StandardScaler(0,1)), kernel=Guass(theta=1.0, heterogeneous=False), optimizer=GA(maxFEs=1000), n_restart_optimize=0, fitMode='predictError')
krg.fit(xTrain, yTrain)
yPred=krg.predict(xTest)
value=r_square(yTest, yPred)
print(value)