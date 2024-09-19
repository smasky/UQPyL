import sys
sys.path.append('.')

from UQPyL.problems import Sphere

problem=Sphere(nInput=10)

#-------------ASMO--------------#

# from UQPyL.optimization.single_objective import ASMO
# from UQPyL.surrogates.rbf import RBF
# from UQPyL.optimization.single_objective import GA
# asmo=ASMO(nInit=200, surrogate=RBF(), optimizer=GA(maxFEs=10000), maxFEs=500)
# res=asmo.run(problem)

#-----------EGO---------------#
# from UQPyL.optimization.single_objective import EGO
# from UQPyL.optimization.single_objective import ASMO
# ego=ASMO(nInit=50)
# res=ego.run(problem)

#-----------GA---------------#
# from UQPyL.optimization.single_objective import GA
# ga=GA(saveFlag=True)
# res=ga.run(problem)

