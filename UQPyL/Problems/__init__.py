from .Single_Benchmarks import (Sphere, Schwefel_2_22, Schwefel_1_22, Schwefel_2_21, Rosenbrock, 
                         Step, Quartic, Schwefel_2_26, Rastrigin, Ackley, Griewank, 
                         Trid, Bent_Cigar, Discus, Weierstrass)
from .Multi_ZDT import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from .Multi_DTLZ import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6
from .Pratical_problem import Problem
from .Problem_ABC import ProblemABC

single_problems=["Sphere", "Schwefel_2_22", "Schwefel_1_22", "Schwefel_2_21", "Rosenbrock",
                "Step", "Quartic", "Schwefel_2_26", "Rastrigin", "Ackley", "Griewank",
                "Trid", "Bent_Cigar", "Discus", "Weierstrass",]

multi_problem=["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6",
               "DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5",
               "DTLZ6", "DTLZ7"]

__all__=[
    single_problems,
    multi_problem,
    "Problem",
    "ProblemABC"
]