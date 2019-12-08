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


def get_cmd_classes():

    class build_ext_subclass(build_ext):
        def build_extensions(self):
            build_ext.build_extensions(self)

    return {'build_ext': build_ext_subclass}


def get_defined_args():
    if sys.platform.startswith("win"):
        # windows
        define_macros = [('USE_OPENMP', None)]
        libraries = None
        extra_compile_args = ['/EHsc', '/O2', '/Gy']
    elif sys.platform.startswith("darwin"):
        # mac osx
        define_macros = [('USE_OPENMP', None)]
        libraries = None
        extra_compile_args = ['-malign-double']
    else:
        # linux
        define_macros = [('USE_OPENMP', None)]
        libraries = None
        extra_compile_args = ['-malign-double']

    return {
        'define_macros': define_macros,
        'libraries': libraries,
        'extra_compile_args': extra_compile_args,
    }


def get_extension_tutorial(name):
    pattern1 = "td3a_cpp.tutorial.%s"
    srcs = ['td3a_cpp/tutorial/%s.pyx' % name]
    args = get_defined_args()
    if name == 'dot_cython':
        srcs.extend(['td3a_cpp/tutorial/%s_.cpp' % name])
        args['language'] = "c++"
    ext = Extension(pattern1 % name,
                    srcs,
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
packages = find_packages(where=here)
package_dir = {k: os.path.join(here, k.replace(".", "/")) for k in packages}
package_data = {
    "td3a_cpp.tutorial": ["*.pyx", '*.cpp', '*.h'],
}

with open(os.path.join(here, "requirements.txt"), "r") as f:
    requirements = f.read().strip(' \n\r\t').split('\n')
if len(requirements) == 0 or requirements == ['']:
    requirements = []

ext_modules = []
for ext in ['dot_blas_lapack', 'dot_cython']:
    ext_modules.extend(get_extension_tutorial(ext))


setup(name='td3a_cpp',
      version='0.1',
      description="Example of a python module including cython and openmp",
      long_description="Exemple d'un module python incluant cython et openmp",
      author='Xavier Dupré',
      author_email='xavier.dupre@gmail.com',
      url='https://github.com/sdpython/td3a_cpp',
      ext_modules=ext_modules,
      packages=packages,
      package_dir=package_dir,
      package_data=package_data,
      setup_requires=["cython", "numpy", "scipy"],
      requires=requirements,
      cmdclass=get_cmd_classes())
