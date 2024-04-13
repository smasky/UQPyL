import sys
sys.path.append(".")
from scipy.io import loadmat
import os
os.chdir('./Test')


from UQPyL.problems.Single_Benchmarks import (Sphere,Schwefel_1_22,Schwefel_2_22,Schwefel_2_21,
                                       Rosenbrock,Step,Quartic,Schwefel_2_26,Rastrigin,
                                       Ackley,Griewank,Trid,Bent_Cigar,Discus,
                                       Weierstrass)
                                    
from UQPyL.DoE import LHS
import numpy as np

DOE=LHS('classic')
Train_X=DOE(100,5)

########################Sphere
Problem=Sphere(5)
Train_Y=Problem.evaluation(Train_X)
print(Train_Y)

##########################Schwefel_2_22
Problem=Schwefel_2_22(5)
Train_Y=Problem.evaluation(Train_X)
print(Train_Y)

##########################Schwefel_1_22
Problem=Schwefel_1_22(5)
Train_Y=Problem.evaluation(Train_X)
print(Train_Y)

############################Schwefel_2_21
Problem=Schwefel_2_21(5)
Train_Y=Problem.evaluation(Train_X)
print(Train_Y)

##########################Rosenbrock
Problem=Rosenbrock(5)
Train_Y=Problem.evaluation(Train_X)
print(Train_Y)

#########################Step
Problem=Step(5)
Train_Y=Problem.evaluation(Train_X)
print(Train_Y)

##########################Quartic
Problem=Quartic(5)
Train_Y=Problem.evaluation(Train_X)
print(Train_Y)

##########################Schwefel_2_26
Problem=Schwefel_2_26(5)
Train_Y=Problem.evaluation(Train_X)
print(Train_Y)

###########################Rastrigin
Problem=Rastrigin(5)
Train_Y=Problem.evaluation(Train_X)
print(Train_Y)

#########################Ackley
Problem=Ackley(5)
Train_Y=Problem.evaluation(Train_X)
print(Train_Y)

########################Griewank
Problem=Griewank(5)
Train_Y=Problem.evaluation(Train_X)
print(Train_Y)

#########################Trid
Problem=Trid(5)
Train_Y=Problem.evaluation(Train_X)
print(Train_Y)

#######################Bent_Cigar
Problem=Bent_Cigar(5)
Train_Y=Problem.evaluation(Train_X)
print(Train_Y)


##########################Discus
Problem=Discus(5)
Train_Y=Problem.evaluation(Train_X)
print(Train_Y)

##########################Weierstrass
Problem=Weierstrass(5)
Train_Y=Problem.evaluation(np.zeros_like(Train_X)+0.5)
print(Train_Y)