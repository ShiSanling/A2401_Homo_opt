from setuptools import setup
import os
from setuptools import Extension
import sys
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

if sys.platform == 'linux':
    linalg_solve_moudle = Extension(
        name='linalg_solve_moudle',
    #     extra_compile_args=[" -openmp"],
        sources=['eigen.cpp'],
        include_dirs=[  '/usr/include/eigen3',
                        '/home/topjournals/.local/include'],)
    setup(ext_modules=[linalg_solve_moudle])

elif sys.platform == 'win32':
    linalg_solve_moudle = Extension(name='linalg_solve_moudle',
                                    sources=['eigen.cpp'],
                                    include_dirs=['D:\Program Files\eigen3.4',
                                                  '.\env\Scripts',
                                                  'D:\Anaconda3\envs\homo\Lib\site-packages\pybind11\include'])

    setup(ext_modules=[linalg_solve_moudle])


#compile use "python setup.py build_ext --inplace"
"""
g++ -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I/usr/include/eigen3 -I/usr/local/include/pybind11 -I/usr/include/python3.8 -c eigen.cpp -o build/temp.linux-x86_64-3.8/eigen.o -openmp

copying build/lib.linux-x86_64-3.8/linalg_solve_moudle.cpython-38-x86_64-linux-gnu.so -> 


"""