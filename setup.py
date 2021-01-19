# -*- coding: utf-8 -*-
import os
import sys
from distutils.command.build_ext import build_ext
from setuptools import setup, Extension
from setuptools import find_packages
import numpy

##########
# helpers
##########


# Placeholder to change the default
# compilation for command ``build_ext --inplace``.
def get_cmd_classes():

    class build_ext_subclass(build_ext):
        def build_extensions(self):
            build_ext.build_extensions(self)

    return {'build_ext': build_ext_subclass}


def get_defined_args():
    if sys.platform.startswith("win"):
        # windows
        define_macros = [('USE_OPENMP', None)]
        libraries = ['kernel32']
        extra_compile_args = ['/EHsc', '/O2', '/Gy', '/openmp']
        extra_link_args = None
    elif sys.platform.startswith("darwin"):
        # mac osx
        define_macros = [('USE_OPENMP', None)]
        libraries = None
        extra_compile_args = ['-lpthread', '-stdlib=libc++',
                              '-mmacosx-version-min=10.7', '-Xpreprocessor',
                              '-fopenmp']
        extra_link_args = ["-lomp"]
    else:
        # linux
        define_macros = [('USE_OPENMP', None)]
        libraries = None
        extra_compile_args = ['-lpthread', '-fopenmp']
        # option '-mavx2' forces the compiler to use
        # AVX instructions the processor might not have
        extra_link_args = ['-lgomp']

    return {
        'define_macros': define_macros,
        'libraries': libraries,
        'extra_compile_args': extra_compile_args,
        'extra_link_args': extra_link_args,
    }


def get_extension_tutorial(name):
    pattern1 = "td3a_cpp.tutorial.%s"
    srcs = ['td3a_cpp/tutorial/%s.pyx' % name]
    args = get_defined_args()
    if name in ['dot_cython', 'experiment_cython', 'dot_cython_omp',
                'mul_cython_omp']:
        srcs.extend(['td3a_cpp/tutorial/%s_.cpp' % name])
        args['language'] = "c++"

    ext = Extension(pattern1 % name, srcs,
                    include_dirs=[numpy.get_include()],
                    **args)

    opts = dict(boundscheck=False, cdivision=True,
                wraparound=False, language_level=3,
                cdivision_warnings=True)

    ext_modules = []

    from Cython.Build import cythonize
    ext_modules.extend(cythonize([ext], compiler_directives=opts))
    return ext_modules


######################
# beginning of setup
######################


here = os.path.dirname(__file__)
if here == "":
    here = '.'
packages = find_packages(where=here)
package_dir = {k: os.path.join(here, k.replace(".", "/")) for k in packages}
package_data = {
    "td3a_cpp.tutorial": ["*.pyx", '*.cpp', '*.h'],
}

try:
    with open(os.path.join(here, "requirements.txt"), "r") as f:
        requirements = f.read().strip(' \n\r\t').split('\n')
except FileNotFoundError:
    requirements = []
if len(requirements) == 0 or requirements == ['']:
    requirements = []

try:
    with open(os.path.join(here, "readme.rst"), "r", encoding='utf-8') as f:
        long_description = "td3a_cpp:" + f.read().split('td3a_cpp:')[1]
except FileNotFoundError:
    long_description = ""

version_str = '0.0.1'
with open(os.path.join(here, 'td3a_cpp/__init__.py'), "r") as f:
    line = [_ for _ in [_.strip("\r\n ")
                        for _ in f.readlines()]
            if _.startswith("__version__")]
    if len(line) > 0:
        version_str = line[0].split('=')[1].strip('" ')

ext_modules = []
for ext in ['dot_blas_lapack', 'dot_cython',
            'experiment_cython', 'dot_cython_omp',
            'mul_cython_omp']:
    ext_modules.extend(get_extension_tutorial(ext))


setup(name='td3a_cpp',
      version=version_str,
      description="Example of a python module including cython and openmp",
      long_description=long_description,
      author='Xavier Dupré',
      author_email='xavier.dupre@gmail.com',
      url='https://github.com/sdpython/td3a_cpp',
      ext_modules=ext_modules,
      packages=packages,
      package_dir=package_dir,
      package_data=package_data,
      setup_requires=["cython", "numpy", "scipy"],
      install_requires=["cython", "numpy", "scipy"],
      cmdclass=get_cmd_classes())
