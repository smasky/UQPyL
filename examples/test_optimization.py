import sys
sys.path.insert(0, '.')

import numpy as np
from UQPyL.optimization.single_objective import GA
from UQPyL.problems.single_objective import Sphere

problem = Sphere(30)
ga = GA()

res = ga.run(problem)
