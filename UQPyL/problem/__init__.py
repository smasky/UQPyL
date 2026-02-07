from .sop import (Sphere, Schwefel_2_22, Schwefel_1_22, Schwefel_2_21, Rosenbrock, 
                         Step, Quartic, Schwefel_2_26, Rastrigin, Ackley, Griewank, 
                         Trid, Bent_Cigar, Discus, Weierstrass, RosenbrockWithCon)

from .mop import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from .mop import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from .base import ProblemABC
from .problem import Problem

singleFunc = ProblemABC.singleFunc

sop = ["Sphere", "Schwefel_2_22", "Schwefel_1_22", "Schwefel_2_21", "Rosenbrock",
                "Step", "Quartic", "Schwefel_2_26", "Rastrigin", "Ackley", "Griewank",
                "Trid", "Bent_Cigar", "Discus", "Weierstrass",]

mop = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6",
               "DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5",
               "DTLZ6", "DTLZ7"]

__all__=[
    sop,
    mop,
    "ProblemABC",
    "Problem"
]