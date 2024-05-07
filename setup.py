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
    Extension("UQPyL.surrogates.mars_._types", ["UQPyL/surrogates/mars_/_types.pyx"], include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars_._util", ["UQPyL/surrogates/mars_/_util.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars_._forward", ["UQPyL/surrogates/mars_/_forward.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars_._record", ["UQPyL/surrogates/mars_/_record.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars_._basis", ["UQPyL/surrogates/mars_/_basis.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars_._pruning", ["UQPyL/surrogates/mars_/_pruning.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars_._qr", ["UQPyL/surrogates/mars_/_qr.pyx"],include_dirs=[numpy_inc]),
    Extension("UQPyL.surrogates.mars_._knot_search", ["UQPyL/surrogates/mars_/_knot_search.pyx"],include_dirs=[numpy_inc]),
    Extension('UQPyL.surrogates.lasso_.lasso_fast', sources=[str(Path('UQPyL/surrogates/lasso_/lasso_fast.pyx'))], include_dirs=[numpy_inc],)
]
#pybind11扩展模块
pybind11_extensions = [
    Extension("UQPyL.surrogates.svr_.libsvm_interface", [str(Path("UQPyL/surrogates/svr_/libsvm_interface.cpp")), str(Path("UQPyL/surrogates/svr_/svm.cpp"))], include_dirs=[numpy_inc, pybind11_inc]),
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
    author="wmtSky",
    version="2.0.4",
    author_email="wmtsky@hhu.edu.cn",
    # ... 其他常规setup参数 ...
    ext_modules=extensions,  # 如果有自定义的编译行为
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