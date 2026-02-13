# ------------------------------------------- #
#              Single Objective               #
# ------------------------------------------- #

from UQPyL.problem.sop import Sphere

problem = Sphere(10)

# ------------------------------------------- #
#                    EGO                      # 
# ------------------------------------------- #

# from UQPyL.optimization.soea import EGO

# ego = EGO(nInit = 50, maxFEs = 1000, saveFlag = True)

# ego.run(problem)


# ------------------------------------------- #
#                    ASMO                     #
# ------------------------------------------- #

from UQPyL.optimization.soea import ASMO, GA

optimizer = GA(maxFEs = 5000)

asmo = ASMO(optimizer = optimizer)

asmo.run(problem)


# ------------------------------------------- #
#              Multi Objective                #
# ------------------------------------------- #

from UQPyL.problem.mop import ZDT1

zdt1 = ZDT1(10)

# ------------------------------------------- #
#                 MOASMO                      #
# ------------------------------------------- #

# from UQPyL.optimization.moea import MOASMO

# moasmo = MOASMO(maxFEs = 200)
# moasmo.run(zdt1)
