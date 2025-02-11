from .single_objective import (Sphere, Schwefel_2_22, Schwefel_1_22, Schwefel_2_21, Rosenbrock, 
                         Step, Quartic, Schwefel_2_26, Rastrigin, Ackley, Griewank, 
                         Trid, Bent_Cigar, Discus, Weierstrass, RosenbrockWithCon)

from .multi_objective import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from .multi_objective import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6
from .problemABC import ProblemABC
from .problem import Problem

single_objective_problems=["Sphere", "Schwefel_2_22", "Schwefel_1_22", "Schwefel_2_21", "Rosenbrock",
                "Step", "Quartic", "Schwefel_2_26", "Rastrigin", "Ackley", "Griewank",
                "Trid", "Bent_Cigar", "Discus", "Weierstrass",]

multi_objective_problem=["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6",
               "DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5",
               "DTLZ6", "DTLZ7"]

__all__=[
    single_objective_problems,
    multi_objective_problem,
    "ProblemABC"
]