'''
Examples for using DOE methods:
    - 1. Full factorial design (FFD)
    - 2. Latin hypercube sampling (LHS)
    - 3. Sobol Sequence
    - 4. Random Design
    - 5. The sample method for Fourier Amplitude Sensitivity Test 
'''
#temple
import sys
sys.path.append(".")
from scipy.io import loadmat
import os
os.chdir('./examples')
import numpy as np
from UQPyL.DoE import FFD
from UQPyL.DoE import LHS
from UQPyL.DoE import Random

from UQPyL.DoE import Sobol_Sequence
from UQPyL.DoE import FAST_Sequence
############################test#############################
from UQPyL.problems import Sphere
pro=Sphere(n_input=5, ub=100, lb=-100, disc_var=np.array([0, 0, 1, 1, 1]), disc_range=[0, 0, [2,5,7], [1,3,2], [0,1]])
lhs=LHS(problem=pro)
a=lhs.sample(100, 5)
#################1. Full factorial design (FFD)#################
print("#################Full factorial design (FFD)#################")
# Create an instance of the FFD class
ffd=FFD()

# Generate a full factorial design with 2 levels for each of the 3 variables
# There are two ways:

X1=ffd.sample(3, 3) #Recommend this way
print(X1)

# X2=ffd(3, 3)
# print(X2)

#################2. Latin Hypercube Sampling (LHS)#################
print('#################Latin Hypercube Sampling (LHS)#################')
# Create an instance of the LHS class
# Noted that the default criterion is 'classic', but has 'center', 'center_maximin', 'maximin',  'correlation'

lhs=LHS('classic') #You also can use lhs=LHS('center'), lhs=LHS('maximin) ....

# Generate a latin hypercube sampling with 3 samples and 3 variables
X3=lhs.sample(3, 3) #Recommend this way
print(X3)

# Also use following way:
# X4=lhs(3, 3)
# print(X4)

#################3. Random Design#################
print("#################Random Design#################")
# Create an instance of the Random class
rad=Random()

# Generate a random design with 3 samples and 3 variables
X5=rad.sample(3,3)
print(X5)

# Also use following way:
# X6=rad(3, 3)
# print(X6)

#################4. Sobol Sequence#################
print("#################Sobol Sequence#################")

# Create an instance of the Sobol_Sequence class
sobol_seq=Sobol_Sequence(scramble=True)

# Generate a Sobol sequence with 8 samples and 3 variables
X7=sobol_seq.sample(81, 2) #Recommend this way
print(X7)


#################5. The sampling for Fourier Amplitude Sensitivity Test#################
print('#################The sample method for Fourier Amplitude Sensitivity Test#################')

# Generate a FAST sampling with 81 samples and 2 variables
fast_seq=FAST_Sequence(M=4)
X8=fast_seq.sample(81, 2)
print(X8)


# Visually show the samples of Sobol and FAST sampling
import matplotlib.pyplot as plt

x7 = X7[:, 0]; y7 = X7[:, 1]
x8 = X8[:, 0]; y8 = X8[:, 1]
fig, axs = plt.subplots(1, 2)

axs[0].scatter(x7, y7)
axs[0].set_title('Sobol Sequence')

axs[1].scatter(x8, y8)
axs[1].set_title('FAST Sampling')

plt.show()

