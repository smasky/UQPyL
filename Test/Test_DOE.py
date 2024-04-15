import sys
sys.path.append(".")
from scipy.io import loadmat
import os
os.chdir('./Test')

import numpy as np
import matplotlib.pyplot as plt
from UQPyL.DoE import RANDOM, FFD, LHS, Sobol_Sequence, FAST_Sampler

##################Random Design####################
DOE=RANDOM()
print(DOE(10,10))

##################Full Factor Design####################
DOE=FFD()
print(DOE(10,3))

#################Latin-Hypercube Design###################
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

######################Sobol Sequence#################
sob=Sobol_Sequence()
print(sob.sample(8,2))
x=sob.sample(128,2)
# plt.scatter(x[:, 0], x[:, 1])
# plt.show()

#####################FAST Sampler###################
fast=FAST_Sampler()
X=fast.sample(256,2)
plt.scatter(x[:, 0], x[:, 1])
plt.show()
