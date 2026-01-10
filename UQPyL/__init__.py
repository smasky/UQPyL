"""
UQPyL package entry.

CI (Linux) is case-sensitive. If the repository historically contains a mixed-case
DOE folder name (e.g. `DoE`), imports like `import UQPyL.doe` or relative imports
`from ..doe import ...` will fail. To make the package robust across platforms,
we create a small alias between `UQPyL.doe` and `UQPyL.DoE` before importing other
subpackages (some of which depend on DOE at import-time).
"""

import importlib
import sys

# --- DOE/DoE compatibility shim (must run early) ---
_doe_mod = None
try:
    _doe_mod = importlib.import_module(__name__ + ".doe")
except ModuleNotFoundError:
    try:
        _doe_mod = importlib.import_module(__name__ + ".DoE")
    except ModuleNotFoundError:
        _doe_mod = None

if _doe_mod is not None:
    # Alias both names to the same module object.
    sys.modules.setdefault(__name__ + ".doe", _doe_mod)
    sys.modules.setdefault(__name__ + ".DoE", _doe_mod)

# Import subpackages after the shim so downstream imports won't crash.
from . import problem, surrogate, optimization, analysis, util, inference  # noqa: E402,F401

# Keep `doe` attribute if available
doe = _doe_mod  # noqa: E402

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