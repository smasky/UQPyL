from .base import SurrogateABC, MultiSurrogate
from . import rbf
from . import regression
from . import fnn
from . import gp
from . import kriging
from . import svr
from . import fnn
from .auto_tuner import AutoTuner

# MARS depends on optional compiled extensions under `surrogate/mars/core`.
# They may not be available for all Python versions in CI (e.g. cp38-only wheels).
try:  # pragma: no cover
    from . import mars  # noqa: F401
except Exception:  # pragma: no cover
    mars = None