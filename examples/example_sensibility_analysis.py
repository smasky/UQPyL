'''
    Examples for using sensibility analysis methods:
    - Sobol
    - extended Fourier Amplitude Sensitivity Test (eFAST)
    - Random Balance Designs - Fourier Amplitude Sensitivity Test (RBD-FAST)
    - Delta_test (DT)
    - Morris
    - Regional Sensitivity Analysis (RSA)
    - Multivariate Adaptive Regression Splines-Sensibility Analysis (MARS-SA)
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
problem=Ishigami(4, 1, np.pi, -1*np.pi)
#x1 0.3199   S2: x1_x3 0.25
#x2 0.4424
#x3 0.0
#x4 0.0


################Sobol################
print("################Sobol################")
sobol1=Sobol(problem=problem, N_within_sampler=1024, cal_second_order=True) #Using Sobol Sequence and saltelli_sequence
Si=sobol1.analyze(verbose=True)


################FAST################
print("################FAST################")
fast1=FAST(problem=problem, N_within_sampler=1000) #Using FAST Sampler
Si=fast1.analyze(verbose=True)
a=1



# Y=fast1.Y

# from SALib.analyze import fast
# from SALib.sample import fast_sampler
# from SALib.test_functions import Ishigami

# from SALib import ProblemSpec

# sp = ProblemSpec(
#             {
#                 "names": ["x1", "x2", "x3"],
#                 "groups": None,
#                 "bounds": [[-np.pi, np.pi]] * 3,
#                 "num_vars":3,
#                 "dists":['unif']*3,
#             }
#         )

# samples=fast_sampler.sample(sp, 1000)
# # Y=problem.evaluate(samples)
# Si = fast.analyze(
#     sp, Y[:,0],conf_level=0.95, print_to_console=True
# )
# a=1



                                                
        