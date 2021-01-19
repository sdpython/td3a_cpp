"""

.. _l-example-mul:

Compares mul implementations
============================

:epkg:`numpy` has a very fast implementation of
matrix multiplication. There are many ways to be slower.

.. contents::
    :local:
"""

import pprint
import numpy
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
pprint.pprint(dfs[-1].tail(n=2))


##############################
# Other scenarios
# +++++++++++++++
#

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

##############################
# Other scenarios but transposed
# ++++++++++++++++++++++++++++++
#

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

fig, ax = plt.subplots(3, 2, figsize=(10, 8))
cc[~cc.fct.str.contains('-T')].pivot('N', 'fct', 'average').plot(
    logy=True, logx=True, ax=ax[0, 0])
cc[~cc.fct.str.contains('-T') & (cc.fct != 'numpy')].pivot(
    'N', 'fct', 'average').plot(logy=True, logx=True, ax=ax[0, 1])
cc[cc.fct.str.contains('-T') | (cc.fct == 'numpy')].pivot(
    'N', 'fct', 'average').plot(logy=True, logx=True, ax=ax[1, 0])
cc[cc.fct.str.contains('-T') & (cc.fct != 'numpy')].pivot(
    'N', 'fct', 'average').plot(logy=True, logx=True, ax=ax[1, 1])
cc[cc.fct.str.contains('a=0')].pivot('N', 'fct', 'average').plot(
    logy=True, logx=True, ax=ax[2, 1])
fig.suptitle("Comparison of multiplication implementations")

#################################
# The results depends on the machine, its
# number of cores, the compilation settings
# of :epkg:`numpy` or this module.

plt.show()
