'''
    Examples for using sensibility analysis methods:
    - 1. Sobol
    - 2. extended Fourier Amplitude Sensitivity Test (eFAST)
    - 3. Random Balance Designs - Fourier Amplitude Sensitivity Test (RBD-FAST)
    - 4. Morris
    - 5. Regional Sensitivity Analysis (RSA)
    - 6. Multivariate Adaptive Regression Splines-Sensibility Analysis (MARS-SA)
    - 7. Delta_test (DT)
'''

#tmp
import sys
sys.path.append(".")
from scipy.io import loadmat
print(sys.path)
import os
os.chdir('./examples')
#

from UQPyL.sensibility import Sobol
from UQPyL.sensibility import FAST
from UQPyL.sensibility import RBD_FAST
from UQPyL.sensibility import Delta_Test
from UQPyL.sensibility import Morris
from UQPyL.sensibility import RSA
from UQPyL.sensibility import MARS_SA

import numpy as np
##################Prepare the data##################
#Here we construct Ishigami-Homma problem for sensibility_analysis
from UQPyL.problems import ProblemABC
#We should inherit the superclass ProblemABC
#Like following:

class Ishigami(ProblemABC):
    def __init__(self, n_input, n_output, ub, lb, A=7.0, B=0.1):
        '''
        The variable n_input, n_output, ub, lb must be given.
        '''
        self.A=A; self.B=B
        super().__init__(n_input, n_output, ub, lb)
    
    def evaluate(self, X):
        '''
        The evaluate function must be implemented in the subclass.
        '''
        Y = np.sin(X[:, 0])\
            + self.A * np.power(np.sin(X[:, 1]), 2)\
            + self.B * np.power(X[:, 2], 4) * np.sin(X[:, 0])
        
        return Y.reshape(-1,1)

#Create an instance of the Ishigami class
problem=Ishigami(3, 1, np.pi, -1*np.pi)
#S1: x1 0.3199   S2: x1_x3 0.25
#    x2 0.4424
#    x3 0.0
#    x4 0.0

################1. Sobol#################
print("################1.Sobol################")
from UQPyL.sensibility import Sobol
problem=Ishigami(3, 1, np.pi, -1*np.pi)
sobol_method=Sobol(problem=problem, cal_second_order=True) #Using Sobol Sequence and saltelli_sequence
X=sobol_method.sample(512)
Y=problem.evaluate(X)
Si=sobol_method.analyze(X, Y, verbose=True)

################2. FAST##################
print("################2.FAST################")
from UQPyL.sensibility import FAST
fast_method=FAST(problem=problem, M=4)
X=fast_method.sample(500)
Y=problem.evaluate(X)
Si=fast_method.analyze(X, Y, verbose=True)

###############3. RBD_FAST#################
print("##############3.RBD_FAST##############")
rbd_method=RBD_FAST(problem=problem, M=4)
X=rbd_method.sample(1000)
Y=problem.evaluate(X)
Si=rbd_method.analyze(X, Y, verbose=True)

################4. Morris###################
print("#############4.Morris#############")
morris_method=Morris(problem=problem, num_levels=4) #Using Morris Sampler
X=morris_method.sample(500)
Y=problem.evaluate(X)
Si=morris_method.analyze(X, Y, verbose=True)

################5. RSA###################
print("#############5.RSA#############")
rsa_method=RSA(problem=problem, n_region=20)
X=rsa_method.sample(1000)
Y=problem.evaluate(X)
Si=rsa_method.analyze(X, Y, verbose=True)


#################6. MARS_SA################
print("#############6.MARS_SA#############")
mars_method=MARS_SA(problem=problem)
X=mars_method.sample(1000)
Y=problem.evaluate(X)
Si=mars_method.analyze(X, Y, verbose=True)

##################7. Delta_Test################
print('#############7.Delta_Test#############')
delta_method=Delta_Test(problem=problem)
X=delta_method.sample(1000)
Y=problem.evaluate(X)
Si=delta_method.analyze(X, Y, verbose=True)

