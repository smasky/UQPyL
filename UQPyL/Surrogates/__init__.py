from . import RBF_Kernel
from . import GP_Kernel
from .linear_regression import LinearRegression
from .gaussian_process import GPR
from .radial_basis_function import RBF
from .polynomial_regression import PolynomialRegression
from .support_vector_machine import SVR
from .fully_connect_neural_network import FNN
from .kriging import Kriging as KRG
__all__=[
    "LinearRegression",
    "PolynomialRegression",
    "RBF",
    "RBF_Kernel",
    "Kriging",
    "GP_Kernel",
    "MLP",
    "SVR",
    "GPR",
    "FNN",
    "KRG",
]