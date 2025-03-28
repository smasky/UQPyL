from setuptools import setup, Extension, find_packages
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
    Extension("UQPyL.surrogates.mars.core._types", ["UQPyL/surrogates/mars/core/_types.pyx"], include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars.core._util", ["UQPyL/surrogates/mars/core/_util.pyx"], include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars.core._forward", ["UQPyL/surrogates/mars/core/_forward.pyx"], include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars.core._record", ["UQPyL/surrogates/mars/core/_record.pyx"], include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars.core._basis", ["UQPyL/surrogates/mars/core/_basis.pyx"], include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars.core._pruning", ["UQPyL/surrogates/mars/core/_pruning.pyx"], include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars.core._qr", ["UQPyL/surrogates/mars/core/_qr.pyx"], include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars.core._knot_search", ["UQPyL/surrogates/mars/core/_knot_search.pyx"], include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.regression.lasso.lasso", ["UQPyL/surrogates/regression/lasso/lasso_fast.pyx"], include_dirs=[numpy_inc])
]
#pybind11扩展模块
pybind11_extensions = [
    Extension("UQPyL.surrogates.svr.core.libsvm_interface", [str(Path("UQPyL/surrogates/svr/core/libsvm_interface.cpp")), str(Path("UQPyL/surrogates/svr/core/svm.cpp"))], include_dirs=[numpy_inc, pybind11_inc]),
]

extensions=cythonize(cython_extensions, compiler_directives={'cdivision': True, 'boundscheck': False})+pybind11_extensions

setup(
    name="UQPyL",
    author="wmtSky",
    version="2.1.0",
    author_email="wmtsky@hhu.edu.cn",
    ext_modules=extensions,  
    packages=find_packages(),
    description="A python package for parameter uncertainty quantification and optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  # 如果是Markdown格式
    classifiers=[
        # 添加适合的类目，例如
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)