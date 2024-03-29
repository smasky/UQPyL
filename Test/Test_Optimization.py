import sys
import numpy as np
sys.path.append(".")
from UQPyL.Problems import Sphere
from UQPyL.Experiment_Design import LHS
from UQPyL.Optimization import SCE_UA
import os
os.chdir('./Test')

lhs=LHS('center')
Problems=Sphere(dim=10, ub=100, lb=-100)
Algorithm=SCE_UA(Problems.evaluate, Problems.dim, Problems.lb, Problems.ub)
Algorithm.run()
