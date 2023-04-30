"""

.. _l-example-mul:

Compares matrix multiplication implementations
==============================================

:epkg:`numpy` has a very fast implementation of
matrix multiplication. There are many ways to be slower.

Compared implementations:

* :func:`dmul_cython_omp <td3a_cpp.tutorial.mul_cython_omp.dmul_cython_omp>`
  `code <https://github.com/sdpython/td3a_cpp/blob/master/
  td3a_cpp/tutorial/mul_cython_omp.pyx#L171>`_

.. contents::
    :local:
"""

import pprint
import numpy
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
from td3a_cpp.tutorial.mul_cython_omp import dmul_cython_omp
from td3a_cpp.tools import measure_time_dim

dfs = []
sets = list(range(2, 145, 20))

##############################
# numpy mul
# +++++++++
#

ctxs = [dict(va=numpy.random.randn(n, n).astype(numpy.float64),
             vb=numpy.random.randn(n, n).astype(numpy.float64),
             mul=lambda x, y: x @ y,
             x_name=n)
        for n in sets]

res = list(measure_time_dim('mul(va, vb)', ctxs, verbose=1))
dfs.append(DataFrame(res))
dfs[-1]['fct'] = 'numpy'
pprint.pprint(dfs[-1].tail(n=2))


##############################
# Simple multiplication
# +++++++++++++++++++++
#

ctxs = [dict(va=numpy.random.randn(n, n).astype(numpy.float64),
             vb=numpy.random.randn(n, n).astype(numpy.float64),
             mul=dmul_cython_omp,
             x_name=n)
        for n in sets]

res = list(measure_time_dim('mul(va, vb)', ctxs, verbose=1))
pprint.pprint(res[-1])


##############################
# Other scenarios
# +++++++++++++++
#
# 3 differents algorithms, each of them parallelized.
# See :func:`dmul_cython_omp
# <td3a_cpp.tutorial.mul_cython_omp.dmul_cython_omp>`.

for algo in range(0, 2):
    for parallel in (0, 1):
        print("algo=%d parallel=%d" % (algo, parallel))
        ctxs = [dict(va=numpy.random.randn(n, n).astype(numpy.float64),
                     vb=numpy.random.randn(n, n).astype(numpy.float64),
                     mul=lambda x, y: dmul_cython_omp(
            x, y, algo=algo, parallel=parallel),
            x_name=n)
            for n in sets]

        res = list(measure_time_dim('mul(va, vb)', ctxs, verbose=1))
        dfs.append(DataFrame(res))
        dfs[-1]['fct'] = 'a=%d-p=%d' % (algo, parallel)
        pprint.pprint(dfs[-1].tail(n=2))

########################################
# One left issue
# ++++++++++++++
#
# Will you find it in :func:`dmul_cython_omp
# <td3a_cpp.tutorial.mul_cython_omp.dmul_cython_omp>`.


va = numpy.random.randn(3, 4).astype(numpy.float64)
vb = numpy.random.randn(4, 5).astype(numpy.float64)
numpy_mul = va @ vb

try:
    for a in range(0, 50):
        wrong_mul = dmul_cython_omp(va, vb, algo=2, parallel=1)
        assert_almost_equal(numpy_mul, wrong_mul)
        print("Iteration %d is Ok" % a)
    print("All iterations are unexpectedly Ok. Don't push your luck.")
except AssertionError as e:
    print(e)


##############################
# Other scenarios but transposed
# ++++++++++++++++++++++++++++++
#
# Same differents algorithms but the second matrix
# is transposed first: ``b_trans=1``.


for algo in range(0, 2):
    for parallel in (0, 1):
        print("algo=%d parallel=%d transposed" % (algo, parallel))
        ctxs = [dict(va=numpy.random.randn(n, n).astype(numpy.float64),
                     vb=numpy.random.randn(n, n).astype(numpy.float64),
                     mul=lambda x, y: dmul_cython_omp(
            x, y, algo=algo, parallel=parallel, b_trans=1),
            x_name=n)
            for n in sets]

        res = list(measure_time_dim('mul(va, vb)', ctxs, verbose=2))
        dfs.append(DataFrame(res))
        dfs[-1]['fct'] = 'a=%d-p=%d-T' % (algo, parallel)
        pprint.pprint(dfs[-1].tail(n=2))


#############################
# Let's display the results
# +++++++++++++++++++++++++

cc = concat(dfs)
cc['N'] = cc['x_name']

fig, ax = plt.subplots(3, 2, figsize=(10, 8), sharex=True, sharey=True)
ccnp = cc.fct == 'numpy'
cct = cc.fct.str.contains('-T')
cca0 = cc.fct.str.contains('a=0')
cc[ccnp | (~cct & cca0)].pivot(
    'N', 'fct', 'average').plot(logy=True, logx=True, ax=ax[0, 0])
cc[ccnp | (~cct & ~cca0)].pivot(
    'N', 'fct', 'average').plot(logy=True, logx=True, ax=ax[0, 1])
cc[ccnp | (cct & cca0)].pivot(
    'N', 'fct', 'average').plot(logy=True, logx=True, ax=ax[1, 0])
cc[ccnp | (~cct & ~cca0)].pivot(
    'N', 'fct', 'average').plot(logy=True, logx=True, ax=ax[1, 1])
cc[ccnp | cca0].pivot(index='N', columns='fct', values='average').plot(
    logy=True, logx=True, ax=ax[2, 0])
cc[ccnp | ~cca0].pivot(index='N', columns='fct', values='average').plot(
    logy=True, logx=True, ax=ax[2, 1])
fig.suptitle("Comparison of matrix multiplication implementations")

#################################
# The results depends on the machine, its
# number of cores, the compilation settings
# of :epkg:`numpy` or this module.

plt.show()
