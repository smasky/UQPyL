import sys
import numpy as np
sys.path.append(".")
from UQPyL.Problems import Sphere, ZDT1, Schwefel_2_22
from UQPyL.Experiment_Design import LHS
from UQPyL.Optimization import SCE_UA, ASMO, NSGAII, MOASMO
from UQPyL.Surrogates import RBF, MO_Surrogates, KRG
from UQPyL.Surrogates.RBF_Kernel import Cubic
from UQPyL.Sensitivity_Analysis import Morris, FAST, RBD_FAST

import matplotlib.pyplot as plt
import os

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
problem= Sphere(dim=2)
rbd=RBD_FAST(problem, NSample=1000)
rbd.analyze()