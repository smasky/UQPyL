import numpy as np
import math

D=10
N=400
M=4


omega = np.zeros([D])
omega[0] = math.floor((N - 1) / (2 * M))
m = math.floor(omega[0] / (2 * M))

if m >= (D - 1):
    omega[1:] = np.floor(np.linspace(1, m, D - 1))
else:
    omega[1:] = np.arange(D - 1) % m +1

a=1