# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
# import pandas as pd

# np_include_path = "C:/Users/Admin/.conda/envs/PhD_TM/lib/site-packages/numpy/core/include"  # from numpy.get_include()
pandas_include_path = "C:/Users/Admin/.conda/envs/PhD_TM/lib/site-packages/pandas/_libs/tslibs"  # /package/include"

modules = [
    'pycart'
]

# Define the extension module
extensions = [
    Extension(
        name=ext, sources=[ext + '.pyx'],
        include_dirs=[numpy.get_include(), pandas_include_path, 'src/'],
        extra_compile_args=['-fopenmp', '-O3', '-std=c++20'],
        extra_link_args=["-fopenmp"]
        # Specify NumPy include path
    )
    for ext in modules
]

setup(ext_modules=cythonize(extensions))
