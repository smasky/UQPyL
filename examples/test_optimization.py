import sys
sys.path.insert(0, '.')

#---------------------------------------------#
# Single Objective Optimization
from UQPyL.problem.sop import Sphere

problem = Sphere(30)

# ------------------------------------------- #
#                    GA                       # 
# ------------------------------------------- #

# from UQPyL.optimization.soea import GA

# ga = GA()
# res = ga.run(problem)

# ------------------------------------------- #
#                    PSO                      # 
# ------------------------------------------- #

# from UQPyL.optimization.soea import PSO
# pso = PSO()
# res = pso.run(problem)

# ------------------------------------------- #
#                    DE                       # 
# ------------------------------------------- #

# from UQPyL.optimization.soea import DE
# de = DE()
# res = de.run(problem)

# ------------------------------------------- #
#                   CSA                       # 
# ------------------------------------------- #

# from UQPyL.optimization.soea import CSA
# csa = CSA()
# res = csa.run(problem)

# ------------------------------------------- #
#                  SCE-UA                     # 
# ------------------------------------------- #

# from UQPyL.optimization.soea import SCE_UA
# sce_ua = SCE_UA()
# res = sce_ua.run(problem)

# ------------------------------------------- #
#                  ML-SCE-UA                  # 
# ------------------------------------------- #

# from UQPyL.optimization.soea import ML_SCE_UA
# ml_sce_ua = ML_SCE_UA()
# res = ml_sce_ua.run(problem)

# ------------------------------------------- #
#                    ABC                      # 
# ------------------------------------------- #

# from UQPyL.optimization.soea import ABC
# abc = ABC()
# res = abc.run(problem)

#---------------------------------------------#
# Multi Objective Optimization

from UQPyL.problem.mop import ZDT1
problem = ZDT1(30)

# ------------------------------------------- #
#                    NSGAII                   # 
# ------------------------------------------- #

# from UQPyL.optimization.moea import NSGAII
# nsgaii = NSGAII()
# res = nsgaii.run(problem)

# ------------------------------------------- #
#                   NSGAIII                   # 
# ------------------------------------------- #

# from UQPyL.optimization.moea import NSGAIII
# nsgaiii = NSGAIII()
# res = nsgaiii.run(problem)

# ------------------------------------------- #
#                   MOEA/D                    # 
# ------------------------------------------- #

# from UQPyL.optimization.moea import MOEAD
# moead = MOEAD(aggregation='PBI')
# res = moead.run(problem)

# ------------------------------------------- #
#                     RVEA                    # 
# ------------------------------------------- #

# from UQPyL.optimization.moea import RVEA
# rvea = RVEA()
# res = rvea.run(problem)

