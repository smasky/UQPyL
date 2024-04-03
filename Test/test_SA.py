import sys
import numpy as np
sys.path.append(".")
from UQPyL.Problems import Sphere, ZDT1
from UQPyL.Experiment_Design import LHS
from UQPyL.Optimization import SCE_UA, ASMO, NSGAII, MOASMO
from UQPyL.Surrogates import RBF, MO_Surrogates, KRG
from UQPyL.Surrogates.RBF_Kernel import Cubic
from UQPyL.Sensitivity_Analysis import Morris
import matplotlib.pyplot as plt
import os

problem=Sphere(dim=15)
rbf=RBF(kernel=Cubic())
mor=Morris(problem, surrogate=rbf)
mor.analyze()
