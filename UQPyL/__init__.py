"""UQPyL package entry."""

import importlib
from types import ModuleType

doe = importlib.import_module(__name__ + ".doe")

problem = importlib.import_module(__name__ + ".problem")
viz = importlib.import_module(__name__ + ".viz")

def _import_submodule(name: str) -> ModuleType:
    return importlib.import_module(f"{__name__}.{name}")


surrogate = _import_submodule("surrogate")
optimization = _import_submodule("optimization")
analysis = _import_submodule("analysis")
inference = _import_submodule("inference")
calibration = _import_submodule("calibration")

__version__ = "2.1.5"
__author__ = "wmtSky"

__all__=[
    "problem",
    "surrogate",
    "optimization",
    "analysis",
    "doe",
    "inference",
    "calibration",
    "viz"
]
