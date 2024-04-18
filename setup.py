from distutils.core import setup, Extension
from Cython.Build import cythonize
from pathlib import Path

import scipy
import numpy
import pybind11
# import Cython
# print(Cython.__version__)

numpy_inc = numpy.get_include()
pybind11_inc = pybind11.get_include()
# cython扩展模块
cython_extensions = [
    # Extension("UQPyL.surrogates.mars_._types", ["UQPyL/surrogates/mars_/_types.pyx"], include_dirs=[numpy_inc], extra_compile_args=['/w'],),
    # Extension("UQPyL.surrogates.mars_._util", ["UQPyL/surrogates/mars_/_util.pyx"],include_dirs=[numpy_inc], extra_compile_args=['/w'],),
    # Extension("UQPyL.surrogates.mars_._forward", ["UQPyL/surrogates/mars_/_forward.pyx"],include_dirs=[numpy_inc], extra_compile_args=['/w'],),
    # Extension("UQPyL.surrogates.mars_._record", ["UQPyL/surrogates/mars_/_record.pyx"],include_dirs=[numpy_inc], extra_compile_args=['/w'],),
    # Extension("UQPyL.surrogates.mars_._basis", ["UQPyL/surrogates/mars_/_basis.pyx"],include_dirs=[numpy_inc], extra_compile_args=['/w'],),
    # Extension("UQPyL.surrogates.mars_._pruning", ["UQPyL/surrogates/mars_/_pruning.pyx"],include_dirs=[numpy_inc], extra_compile_args=['/w'],),
    # Extension("UQPyL.surrogates.mars_._qr", ["UQPyL/surrogates/mars_/_qr.pyx"],include_dirs=[numpy_inc], extra_compile_args=['/w'],),
    # Extension("UQPyL.surrogates.mars_._knot_search", ["UQPyL/surrogates/mars_/_knot_search.pyx"],include_dirs=[numpy_inc], extra_compile_args=['/w'],),
    Extension('UQPyL.surrogates.lasso_.lasso_fast', sources=[Path('UQPyL/surrogates/lasso_/lasso_fast.pyx')], include_dirs=[numpy_inc], extra_compile_args=['/w'],)
]
#pybind11扩展模块
pybind11_extensions = [
    Extension("UQPyL.surrogates.svr_.libsvm_interface", [Path("UQPyL/surrogates/svr_/libsvm_interface.cpp"), Path("UQPyL/surrogates/svr_/svm.cpp")], include_dirs=[numpy_inc, pybind11_inc]),
]

extensions=cythonize(cython_extensions, compiler_directives={'cdivision': True, 'boundscheck': False})+pybind11_extensions
# 使用cythonize编译扩展模块

# setup(
#     name="UQPyL",
#     description='A package for uncertainty quantification and parameter optimization with surrogate models.',
#     version="2.0.0",
#     author="wmtSky",
#     url='https://https://github.com/smasky/UQPyL',
#     ext_modules=extensions,
# )
setup(
    name="UQPyL",
    version="2.0.0",
    author="wmtSky",
    # ... 其他常规setup参数 ...
    ext_modules=extensions,  # 如果有自定义的编译行为
)