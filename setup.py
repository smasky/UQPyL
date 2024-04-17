from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy
import pybind11

numpy_inc = numpy.get_include()
pybind11_inc = pybind11.get_include()
# cython扩展模块
cython_extensions = [
    Extension("UQPyL.surrogates.mars_._types", ["UQPyL/surrogates/mars_/_types.pyx"], include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars_._util", ["UQPyL/surrogates/mars_/_util.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars_._forward", ["UQPyL/surrogates/mars_/_forward.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars_._record", ["UQPyL/surrogates/mars_/_record.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars_._basis", ["UQPyL/surrogates/mars_/_basis.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars_._pruning", ["UQPyL/surrogates/mars_/_pruning.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars_._qr", ["UQPyL/surrogates/mars_/_qr.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars_._knot_search", ["UQPyL/surrogates/mars_/_knot_search.pyx"],include_dirs=[numpy_inc]),
    Extension('UQPyL.surrogates.lasso_.lasso_fast', sources=['UQPyL/surrogates/lasso_/lasso_fast.pyx'], include_dirs=[numpy_inc])
]
#pybind11扩展模块
pybind11_extensions = [
    Extension("UQPyL.surrogates.svr_.libsvm_interface", ["UQPyL/surrogates/svr_/libsvm_interface.cpp", "UQPyL/surrogates/svr_/svm.cpp"], include_dirs=[numpy_inc, pybind11_inc]),
]

extensions=cythonize(cython_extensions)+pybind11_extensions
# 使用cythonize编译扩展模块


setup(
    name="UQPyL",
    description='A package for uncertainty quantification and parameter optimization with surrogate models.',
    version="0.2.0",
    author="wmtSky",
    url='https://github.com/your/repo',
    ext_modules=extensions,
    python_requires='>=3.8',  # 支持的Python版本
    install_requires=[  # 依赖列表
        'scipy',
        'numpy',
        'Cython'
    ]
)