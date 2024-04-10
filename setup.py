from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy
import pybind11

numpy_inc = numpy.get_include()
pybind11_inc = pybind11.get_include()
# cython扩展模块
cython_extensions = [
    Extension("UQPyL.Surrogates.Mars_._types", ["UQPyL/Surrogates/Mars_/_types.pyx"], include_dirs=[numpy_inc]),
    Extension("UQPyL.Surrogates.Mars_._util", ["UQPyL/Surrogates/Mars_/_util.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.Surrogates.Mars_._forward", ["UQPyL/Surrogates/Mars_/_forward.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.Surrogates.Mars_._record", ["UQPyL/Surrogates/Mars_/_record.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.Surrogates.Mars_._basis", ["UQPyL/Surrogates/Mars_/_basis.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.Surrogates.Mars_._pruning", ["UQPyL/Surrogates/Mars_/_pruning.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.Surrogates.Mars_._qr", ["UQPyL/Surrogates/Mars_/_qr.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.Surrogates.Mars_._knot_search", ["UQPyL/Surrogates/Mars_/_knot_search.pyx"],include_dirs=[numpy_inc]),
    Extension('UQPyL.Surrogates.Lasso_.lasso_fast', sources=['UQPyL/Surrogates/Lasso_/lasso_fast.pyx'], include_dirs=[numpy_inc])
]
#pybind11扩展模块
pybind11_extensions = [
    Extension("UQPyL.Surrogates.SVR_.libsvm_interface", ["UQPyL/Surrogates/SVR_/libsvm_interface.cpp", "UQPyL/Surrogates/SVR_/svm.cpp"], include_dirs=[numpy_inc, pybind11_inc]),
]

extensions=cythonize(cython_extensions)+pybind11_extensions
# 使用cythonize编译扩展模块


setup(
    name="UQPyL",
    author="wmt_sky",
    ext_modules=extensions,
)