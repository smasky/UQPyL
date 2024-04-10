import sys
import numpy as np
sys.path.append(".")
from UQPyL.Problems import Sphere, ZDT1, Schwefel_2_22
from UQPyL.Experiment_Design import LHS
from UQPyL.Optimization import SCE_UA, ASMO, NSGAII, MOASMO
from UQPyL.Surrogates import RBF, MO_Surrogates, KRG
from UQPyL.Surrogates.RBF_Kernel import Cubic
from UQPyL.Sensitivity_Analysis import MORRIS, FAST, RBD_FAST, SOBOL, DELTA_TEST,  MARS_SA

import matplotlib.pyplot as plt
import os

#MARS_SA

mar_sa=MARS_SA(problem=Sphere(dim=15))
mar_sa.analyze()

#Morrios
# problem=Sphere(dim=15)
# rbf=RBF(kernel=Cubic())
# mor=Morris(problem, surrogate=rbf)
# mor.analyze()


#FAST
# problem=Sphere(dim=15)
# rbf=RBF(kernel=Cubic())
# fas=FAST(problem)
# fas.analyze()

#RBD_FAST
# problem= Sphere(dim=15)
# rbf=RBF(kernel=Cubic())
# rbd=RBD_FAST(problem, surrogate=rbf,NSample=999)
# rbd.analyze()


#SOBOL
# problem=Sphere(dim=3)
# rbf=RBF(kernel=Cubic())
# sob=SOBOL(problem, cal_second_order=True, NSample=8)
# sob.analyze()

#Delta_test
problem=Sphere(dim=15)
delta=DELTA_TEST(problem, NSample=1000)
delta.analyze()