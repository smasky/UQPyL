from .surrogateABC import Surrogate, Mo_Surrogates
from . import rbf
from . import regression
# from .mars.mars import MARS

kernels=["rbf_kernels", "gp_kernels"]
surrogates=["LinearRegression", "PolynomialRegression", "RBF", "Kriging", "MLP",
            "SVR", "GPR", "FNN", "KRG", "MARS"]
utility=["Mo_Surrogates", "Surrogate"]