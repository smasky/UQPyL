"""UQPyL package entry."""

import importlib

doe = importlib.import_module(__name__ + ".doe")

problem = importlib.import_module(__name__ + ".problem")
util = importlib.import_module(__name__ + ".util")

try:
    surrogate = importlib.import_module(__name__ + ".surrogate")
except Exception:
    surrogate = None

try:
    optimization = importlib.import_module(__name__ + ".optimization")
except Exception:
    optimization = None

try:
    analysis = importlib.import_module(__name__ + ".analysis")
except Exception:
    analysis = None

try:
    inference = importlib.import_module(__name__ + ".inference")
except Exception:
    inference = None

__version__ = "2.1.5"
__author__ = "wmtSky"

__all__=[
    "problem",
    "surrogate",
    "optimization",
    "analysis",
    "doe",
    "inference",
    "util"
]
