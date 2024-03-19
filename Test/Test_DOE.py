import sys
sys.path.append(".")
from scipy.io import loadmat
import os
os.chdir('./Test')


import numpy as np
from UQPyL.Experiment_Design import RANDOM, FFD, LHS

############################Random Design
DOE=RANDOM()
print(DOE(10,10))


############################Full Factor Design
DOE=FFD()
print(DOE(np.array([1,2,3])))


###########################Latin-Hypercube Design
DOE=LHS('classic')
print(DOE(5,10))

DOE=LHS('center')
print(DOE(5,10))

DOE=LHS('center_maximin')
print(DOE(5,10))

DOE=LHS('correlation')
print(DOE(5,10))

DOE=LHS('maximin')
print(DOE(5,10))
