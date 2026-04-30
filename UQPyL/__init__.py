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
except Exception:
    try:
        _doe_mod = importlib.import_module(__name__ + ".DoE")
    except Exception:
        _doe_mod = None

if _doe_mod is not None:
    # Alias both names to the same module object.
    sys.modules.setdefault(__name__ + ".doe", _doe_mod)
    sys.modules.setdefault(__name__ + ".DoE", _doe_mod)

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
