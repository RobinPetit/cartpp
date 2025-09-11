# setup.py
import platform
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

from utils import make_cpp_wrapper

make_cpp_wrapper()

modules = [
    'pycart'
]

_os = platform.system().lower()
if _os == 'linux':
    COMPILE_ARGS  = ['-fopenmp', '-O3', '-std=c++20']
    LINK_ARGS = ['-fopenmp']
elif _os == 'windows':
    COMPILE_ARGS = ['/std:c++20']
    LINK_ARGS = []

COMPILE_ARGS.extend(['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'])

# Define the extension module
extensions = [
    Extension(
        name=ext, sources=[ext + '.pyx'],
        include_dirs=[numpy.get_include(), '../src/'],
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=LINK_ARGS
    )
    for ext in modules
]

setup(ext_modules=cythonize(extensions))
