from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
# ext=[Extension(name='Lasso_.lasso_fast',
#               sources=['Lasso_/lasso_fast.pyx'],
#               include_dirs=[np.get_include()]),
#     Extension(name='Lasso_.cython_utils',
#               sources=['Lasso_/cython_utils.pyx'],
#               include_dirs=[np.get_include()]),
#     ]
ext=[Extension(name='lasso_fast',
              sources=['lasso_fast.pyx'],
              include_dirs=[np.get_include()])]
setup(ext_modules=cythonize(ext, language_level=3))