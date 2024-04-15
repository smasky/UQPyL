import sys
import numpy as np
sys.path.append(".")
from UQPyL.problems import Sphere, ZDT1, Schwefel_2_22
from UQPyL.DoE import LHS
from UQPyL.optimization import SCE_UA, ASMO, NSGAII, MOASMO
from UQPyL.surrogates import RBF, MO_Surrogates, KRG
from UQPyL.surrogates.rbf_kernels import Cubic
from UQPyL.sensibility import Morris, FAST, RBD_FAST, Sobol, Delta_Test,  MARS_SA
from UQPyL.utility import MinMaxScaler
import matplotlib.pyplot as plt
import os

#MARS_SA
# mar_sa=MARS_SA(problem=Sphere(dim=15))
# mar_sa.analyze()


#Morrios
# problem=Sphere(n_input=15)
# rbf=RBF(kernel=Cubic())
# mor=Morris(problem, surrogate=rbf, N_within_surrogate_sampler=500)
# S1, ST=mor.analyze()
# a=1


# FAST
# problem=Sphere(n_input=15)
# rbf=RBF(kernel=Cubic())
# fast=FAST(problem, surrogate=rbf,scale=(MinMaxScaler(0,1), MinMaxScaler(0,1)), N_within_sampler=500, N_within_surrogate_sampler=1000)
# S1, ST=fast.analyze()
# a=1

#RBD_FAST
# problem= Sphere(n_input=15)
# rbf=RBF(kernel=Cubic())
# rbd=RBD_FAST(problem, surrogate=rbf, N_within_sampler=1000, N_within_surrogate_sampler=1000)
# rbd.analyze()

# Sobol
problem=Sphere(n_input=15)
rbf=RBF(kernel=Cubic())
sob=Sobol(problem, cal_second_order=True, N_within_sampler=100)
sob.analyze()

# Delta_test
# problem=Sphere(dim=15)
# delta=DELTA_TEST(problem, NSample=1000)
# delta.analyze()
