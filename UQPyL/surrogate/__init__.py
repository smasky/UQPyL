from .base import SurrogateABC, MultiSurrogate
from . import rbf
from . import regression
from . import fnn
from . import gp
from . import kriging
from . import fnn
from .auto_tuner import AutoTuner

try:
    from . import mars
except ModuleNotFoundError:
    mars = None

try:
    from . import svr
except ModuleNotFoundError:
    svr = None
