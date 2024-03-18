from setuptools import setup
import os
from setuptools import Extension
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
linalg_solve_moudle = Extension(
    name='linalg_solve_moudle',
    extra_compile_args=[" /openmp"],
    sources=['eigen.cpp'],
    include_dirs=[  'D:\Program Files\eigen3.4',
                    '.\env\Scripts',
                    'D:\Anaconda3\envs\homo\Lib\site-packages\pybind11\include'],

              )

setup(ext_modules=[linalg_solve_moudle])

#compile use "python .\setup.py build_ext --inplace"
