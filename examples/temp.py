from UQPyL.problem.sop import Sphere

problem = Sphere(4)

from UQPyL.optimization.soea import ASMO, GA

optimizer = GA(maxFEs = 5000)

asmo = ASMO(optimizer = optimizer)

asmo.run(problem)