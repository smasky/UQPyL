import sys
sys.path.append(".")
from scipy.io import loadmat
import os
os.chdir('./Test')


from UQPyL.problems.Single_Benchmarks import (Sphere,Schwefel_1_22,Schwefel_2_22,Schwefel_2_21,
                                       Rosenbrock,Step,Quartic,Schwefel_2_26,Rastrigin,
                                       Ackley,Griewank,Trid,Bent_Cigar,Discus,
                                       Weierstrass)

from UQPyL.problems.Multi_ZDT import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from UQPyL.problems.Multi_DTLZ import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7

from UQPyL.DoE import LHS
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



DOE=LHS('classic')
Train_X=DOE(100,5)

########################Sphere
Problem=Sphere(5)
Train_Y=Problem.evaluate(Train_X)
print(Train_Y)

##########################Schwefel_2_22
Problem=Schwefel_2_22(5)
Train_Y=Problem.evaluate(Train_X)
print(Train_Y)

##########################Schwefel_1_22
Problem=Schwefel_1_22(5)
Train_Y=Problem.evaluate(Train_X)
print(Train_Y)

############################Schwefel_2_21
Problem=Schwefel_2_21(5)
Train_Y=Problem.evaluate(Train_X)
print(Train_Y)

##########################Rosenbrock
Problem=Rosenbrock(5)
Train_Y=Problem.evaluate(Train_X)
print(Train_Y)

#########################Step
Problem=Step(5)
Train_Y=Problem.evaluate(Train_X)
print(Train_Y)

##########################Quartic
Problem=Quartic(5)
Train_Y=Problem.evaluate(Train_X)
print(Train_Y)

##########################Schwefel_2_26
Problem=Schwefel_2_26(5)
Train_Y=Problem.evaluate(Train_X)
print(Train_Y)

###########################Rastrigin
Problem=Rastrigin(5)
Train_Y=Problem.evaluate(Train_X)
print(Train_Y)

#########################Ackley
Problem=Ackley(5)
Train_Y=Problem.evaluate(Train_X)
print(Train_Y)

########################Griewank
Problem=Griewank(5)
Train_Y=Problem.evaluate(Train_X)
print(Train_Y)

#########################Trid
Problem=Trid(5)
Train_Y=Problem.evaluate(Train_X)
print(Train_Y)

#######################Bent_Cigar
Problem=Bent_Cigar(5)
Train_Y=Problem.evaluate(Train_X)
print(Train_Y)


##########################Discus
Problem=Discus(5)
Train_Y=Problem.evaluate(Train_X)
print(Train_Y)

##########################Weierstrass
Problem=Weierstrass(5)
Train_Y=Problem.evaluate(np.zeros_like(Train_X)+0.5)
print(Train_Y)

####################ZDT1
Problem=ZDT1(15)
Y=Problem.get_PF()
# plt.scatter(Y[:, 0], Y[:, 1])
# plt.show()

####################ZDT2
Problem=ZDT2(15)
Y=Problem.get_PF()
# plt.scatter(Y[:, 0], Y[:, 1])
# plt.show()

####################ZDT3######################
Problem=ZDT3(15)
Y=Problem.get_PF()
# plt.scatter(Y[:, 0], Y[:, 1])
# plt.show()
########################DTLZ################
Problem=DTLZ1(15)
Y=Problem.get_optimum(1000)
######################