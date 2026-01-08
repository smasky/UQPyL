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

# NOTE:
# Do NOT eagerly import heavy subpackages here (e.g. surrogate.mars contains
# optional compiled extensions). Eager imports can break CI environments where
# some compiled wheels are not available (e.g. cp38-only extensions on cp39).
# Instead, we provide lazy attribute access via __getattr__.

__version__ = "2.1.4"
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


def __getattr__(name):
    """
    Lazy import of top-level subpackages.
    This keeps `import UQPyL` lightweight and robust across platforms.
    """
    if name in __all__:
        if name == "doe":
            # Prefer the already-resolved module (doe/DoE shim)
            if _doe_mod is not None:
                return _doe_mod
        mod = importlib.import_module(__name__ + "." + name)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)