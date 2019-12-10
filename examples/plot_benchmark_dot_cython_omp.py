"""

.. _l-example-dot-cython-omp:

Compares dot implementations (numpy, c++, sse, openmp)
======================================================

:epkg:`numpy` has a very fast implementation of
the dot product. It is difficult to be better and very easy
to be slower. This example looks into a couple of slower
implementations with cython. The tested functions are
the following:

* :func:`ddot_array_16_sse <td3a_cpp.tutorial.dot_cython.ddot_array_16_sse>`
* :func:`ddot_cython_array_omp
  <td3a_cpp.tutorial.dot_cython_omp.ddot_cython_array_omp>`
* :func:`ddot_array_openmp
  <td3a_cpp.tutorial.dot_cython_omp.ddot_array_openmp>`
* :func:`ddot_array_openmp_16
  <td3a_cpp.tutorial.dot_cython_omp.ddot_array_openmp_16>`

.. contents::
    :local:
"""

import numpy
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
from td3a_cpp.tutorial.dot_cython import (
    ddot_array_16_sse, ddot_array
)
from td3a_cpp.tutorial.dot_cython_omp import (
    ddot_cython_array_omp,
    ddot_array_openmp,
    get_omp_max_threads,
    ddot_array_openmp_16
)
from td3a_cpp.tools import measure_time_dim


def get_vectors(fct, n, h=250, dtype=numpy.float64):
    ctxs = [dict(va=numpy.random.randn(n).astype(dtype),
                 vb=numpy.random.randn(n).astype(dtype),
                 dot=fct,
                 x_name=n)
            for n in range(10, n, h)]
    return ctxs

##############################
# Number of threads
# ++++++++++++++++++
#


print(get_omp_max_threads())


##############################
# Several cython dot
# ++++++++++++++++++
#


def numpy_dot(va, vb):
    return numpy.dot(va, vb)


def ddot_omp(va, vb):
    return ddot_cython_array_omp(va, vb)


def ddot_omp_static(va, vb):
    return ddot_cython_array_omp(va, vb, schedule=1)


def ddot_omp_dyn(va, vb):
    return ddot_cython_array_omp(va, vb, schedule=2)


def ddot_omp_cpp(va, vb):
    return ddot_array_openmp(va, vb)


def ddot_omp_cpp_16(va, vb):
    return ddot_array_openmp_16(va, vb)


dfs = []
for fct in [numpy_dot,
            ddot_array,
            ddot_array_16_sse,
            ddot_omp,
            ddot_omp_static,
            ddot_omp_dyn,
            ddot_omp_cpp,
            ddot_omp_cpp_16]:
    ctxs = get_vectors(fct, 40000)

    print(fct.__name__)
    df = DataFrame(list(measure_time_dim('dot(va, vb)', ctxs, verbose=1)))
    df['fct'] = fct.__name__
    dfs.append(df)
    print(df.tail(n=3))

#############################
# Let's display the results
# +++++++++++++++++++++++++

cc = concat(dfs)
cc['N'] = cc['x_name']

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
cc[cc.N <= 1000].pivot('N', 'fct', 'average').plot(
    logy=True, ax=ax[0, 0])
cc.pivot('N', 'fct', 'average').plot(
    logy=True, ax=ax[0, 1])
cc.pivot('N', 'fct', 'average').plot(
    logy=True, logx=True, ax=ax[1, 1])
cc[((cc.fct.str.contains('omp') | (cc.fct == 'ddot_array')) &
    ~cc.fct.str.contains('dyn'))].pivot('N', 'fct', 'average').plot(
    logy=True, ax=ax[1, 0])
ax[0, 0].set_title("Comparison of cython ddot implementations")
ax[0, 1].set_title("Comparison of cython ddot implementations"
                   "\nwithout dot_product")

plt.show()
