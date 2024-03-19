import sys
sys.path.append("./Test/Debug")
from pybind11_eigen import *

import numpy as np
test()


# a = np.array([1,2,3])
# b = cross_matrix(a)
# print(a)
# print(np.sum(b))
# c=1