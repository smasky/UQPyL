from pathlib import Path
import uuid

from setuptools import Extension, find_packages, setup

import numpy
import pybind11

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


numpyInc = numpy.get_include()
pybind11Inc = pybind11.get_include()
cacheDir = Path(".cache")
buildStamp = uuid.uuid4().hex
eggInfoDir = cacheDir / "egg_info" / buildStamp
cythonBuildDir = cacheDir / "cython_build" / buildStamp
eggInfoDir.mkdir(parents=True, exist_ok=True)
cythonBuildDir.mkdir(parents=True, exist_ok=True)

cythonModules = [
    ("UQPyL.surrogate.mars.core._types", "UQPyL/surrogate/mars/core/_types"),
    ("UQPyL.surrogate.mars.core._util", "UQPyL/surrogate/mars/core/_util"),
    ("UQPyL.surrogate.mars.core._forward", "UQPyL/surrogate/mars/core/_forward"),
    ("UQPyL.surrogate.mars.core._record", "UQPyL/surrogate/mars/core/_record"),
    ("UQPyL.surrogate.mars.core._basis", "UQPyL/surrogate/mars/core/_basis"),
    ("UQPyL.surrogate.mars.core._pruning", "UQPyL/surrogate/mars/core/_pruning"),
    ("UQPyL.surrogate.mars.core._qr", "UQPyL/surrogate/mars/core/_qr"),
    ("UQPyL.surrogate.mars.core._knot_search", "UQPyL/surrogate/mars/core/_knot_search"),
    ("UQPyL.surrogate.regression.lasso.lasso", "UQPyL/surrogate/regression/lasso/lasso_fast"),
]

cythonExtensions = [
    Extension(
        moduleName,
        [f"{sourceBase}.pyx"],
        include_dirs=[numpyInc],
    )
    for moduleName, sourceBase in cythonModules
]

pybind11Extensions = [
    Extension(
        "UQPyL.surrogate.svr.core.libsvm_interface",
        [
            str(Path("UQPyL/surrogate/svr/core/libsvm_interface.cpp")),
            str(Path("UQPyL/surrogate/svr/core/svm.cpp")),
        ],
        include_dirs=[numpyInc, pybind11Inc],
    ),
]

if cythonize is None:
    raise RuntimeError("Cython is required to build UQPyL from source.")
else:
    extensions = cythonize(
        cythonExtensions,
        build_dir=str(cythonBuildDir),
        compiler_directives={"cdivision": True, "boundscheck": False},
    ) + pybind11Extensions

setup(
    ext_modules=extensions,
    packages=find_packages(),
    options={"egg_info": {"egg_base": str(eggInfoDir)}},
)
