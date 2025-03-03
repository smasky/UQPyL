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

# Kriging
# from UQPyL.surrogates.kriging import KRG
# from UQPyL.surrogates.kriging.kernel import Guass

# kernel = Guass(heterogeneous = False)
# krg = KRG(scalers=(MinMaxScaler(0, 1), MinMaxScaler(0, 1)), kernel=kernel)
# krg.fit(X, Y)

# YPred = krg.predict(XTest)
 
# r2 = r_square(YTest, YPred)

# print(r2)


#Gaussian Process
from UQPyL.surrogates.gp import GPR
from UQPyL.surrogates.gp.kernel import RBF
from UQPyL.optimization.single_objective import GA

ga = GA(maxFEs= 5000)
kernel = RBF()
gpr = GPR(kernel=kernel, optimizer = ga)
gpr.fit(X, Y)

YPred = gpr.predict(XTest)

r2 = r_square(YTest, YPred)

print(r2)
