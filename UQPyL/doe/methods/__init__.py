from .fast import FASTDesign
from .full_fact import FFD
from .lhs import LHS
from .morris import MorrisDesign
from .random import Random
from .saltelli import SaltelliDesign
from .sobol import Sobol

__all__ = [
    "LHS",
    "FFD",
    "Random",
    "Sobol",
    "SaltelliDesign",
    "FASTDesign",
    "MorrisDesign",
]
