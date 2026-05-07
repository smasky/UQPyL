from .base import SurrogateABC, MultiSurrogate
from . import rbf
from . import regression
from . import gp
from . import kriging
from .auto_tuner import AutoTuner
from .poly import PolyFeature
from .split import KFold, RandSelect
from .metric import r_square, rank_score, nse, mse, sort_score
from .scaler import Scaler, MinMaxScaler, StandardScaler

try:
    from . import mars
except ModuleNotFoundError:
    mars = None

try:
    from . import svr
except ModuleNotFoundError:
    svr = None
