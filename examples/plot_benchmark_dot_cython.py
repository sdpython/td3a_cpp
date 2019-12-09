"""

.. _l-example-dot-cython:

Compares dot implementations (numpy, cython, c++, sse)
======================================================

:epkg:`numpy` has a very fast implementation of
the dot product. It is difficult to be better and very easy
to be slower. This example looks into a couple of slower
implementations with cython. The tested functions are
the following:

* :func:`dot_product <td3a_cpp.tutorial.dot_cython.dot_product>`
* :func:`dot_cython_array <td3a_cpp.tutorial.dot_cython.dot_cython_array>`
* :func:`dot_cython_array_optim
  <td3a_cpp.tutorial.dot_cython.dot_cython_array_optim>`
* :func:`dot_array <td3a_cpp.tutorial.dot_cython.dot_array>`
* :func:`dot_array_16 <td3a_cpp.tutorial.dot_cython.dot_array_16>`
* :func:`dot_array_16_sse <td3a_cpp.tutorial.dot_cython.dot_array_16_sse>`

.. contents::
    :local:
"""

import numpy
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
from td3a_cpp.tutorial.dot_cython import (
    dot_product, dot_cython_array,
    dot_cython_array_optim, dot_array,
    dot_array_16, dot_array_16_sse
)
from td3a_cpp.tools import measure_time_dim


def get_vectors(fct, n, h=100):
    ctxs = [dict(va=numpy.arange(n).astype(numpy.float64),
                 vb=numpy.arange(n).astype(numpy.float64) - 5,
                 dot=fct,
                 x_name=n)
            for n in range(10, n, h)]
    return ctxs

##############################
# numpy dot
# +++++++++
#


ctxs = get_vectors(numpy.dot, 10000)
df = DataFrame(list(measure_time_dim('dot(va, vb)', ctxs, verbose=1)))
df['fct'] = 'numpy.dot'
print(df.tail(n=3))
dfs = [df]

##############################
# Several cython dot
# ++++++++++++++++++
#

for fct in [dot_product, dot_cython_array,
            dot_cython_array_optim, dot_array,
            dot_array_16, dot_array_16_sse]:
    ctxs = get_vectors(fct, 10000 if fct.__name__ != 'dot_product' else 1000)

    df = DataFrame(list(measure_time_dim('dot(va, vb)', ctxs, verbose=1)))
    df['fct'] = fct.__name__
    dfs.append(df)
    print(df.tail(n=3))

#############################
# Let's display the results
# +++++++++++++++++++++++++

cc = concat(dfs)
cc['N'] = cc['x_name']

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
cc[cc.N <= 1100].pivot('N', 'fct', 'average').plot(logy=True, ax=ax[0])
cc[cc.fct != 'dot_product'].pivot(
    'N', 'fct', 'average').plot(logy=True, ax=ax[1])
ax[0].set_title("Comparison of cython dot implementations")
ax[1].set_title("Comparison of cython dot implementations"
                "\nwithout dot_product")

###################
# :epkg:`numpy` is cleary faster.

plt.show()
