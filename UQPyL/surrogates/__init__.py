from .surrogate_ABC import Surrogate, Mo_Surrogates
from .regression.linear_regression import LinearRegression
from .gp.gaussian_process import GPR
from .rbf.radial_basis_function import RBF
from .regression.polynomial_regression import PolynomialRegression
from .svr.support_vector_machine import SVR
from .fnn.fully_connect_neural_network import FNN
from .kriging import KRG
from .mars.mars import MARS

kernels=["rbf_kernels", "gp_kernels"]
surrogates=["LinearRegression", "PolynomialRegression", "RBF", "Kriging", "MLP",
            "SVR", "GPR", "FNN", "KRG", "MARS"]
utility=["Mo_Surrogates", "Surrogate"]

__all__=[
    kernels,
    surrogates,
    utility
]