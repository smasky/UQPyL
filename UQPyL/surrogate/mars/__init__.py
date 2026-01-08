"""
MARS surrogate model.

This subpackage uses optional compiled extensions in `UQPyL.surrogate.mars.core`.
If the extensions are not available for the current Python version/platform,
importing MARS will fail. We keep the import optional so other parts of UQPyL
can still be used (and tested) without MARS.
"""

try:  # pragma: no cover
    from .mars import MARS  # noqa: F401
except Exception as e:  # pragma: no cover
    MARS = None
    _IMPORT_ERROR = e

    def __getattr__(name):
        if name == "MARS":
            raise ImportError(
                "UQPyL.surrogate.mars.MARS requires compiled extensions under "
                "`UQPyL/surrogate/mars/core` for your Python version/platform. "
                "Please build/install wheels for your interpreter (e.g. cp39) "
                "or use Python 3.8 where prebuilt binaries exist."
            ) from _IMPORT_ERROR
        raise AttributeError(name)