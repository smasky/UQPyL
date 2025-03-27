import sys
sys.path.insert(0, '.')

import numpy as np

#---------------------------------------------------------------------------------------#
#Single Objective
from UQPyL.optimization.single_objective import GA, PSO, DE, CSA, SCE_UA, ABC
from UQPyL.problems.single_objective import Sphere

# problem = Sphere(30)

#GA
# ga = GA()
# res = ga.run(problem)

#PSO
# pso = PSO()
# res = pso.run(problem)

#DE
# de = DE()
# res = de.run(problem)

#CSA
# csa = CSA()
# res = csa.run(problem)

#SCE_UA
# sce_ua = SCE_UA()
# res = sce_ua.run(problem)

#ABC
# abc = ABC()
# res = abc.run(problem)

#-----------------------------------------------------------------------------------#
#Multi Objective
from UQPyL.optimization.multi_objective import NSGAII, NSGAIII, MOEAD, RVEA
from UQPyL.problems.multi_objective import ZDT1

problem = ZDT1(30)

#NSGAII
nsgaii = NSGAII()
res = nsgaii.run(problem)

#NSGAIII
# nsgaiii = NSGAIII()
# res = nsgaiii.run(problem)

#MOEAD
# moead = MOEAD()
# res = moead.run(problem)

# RVEA
# rvea = RVEA()
# res = rvea.run(problem)