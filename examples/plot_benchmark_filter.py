"""

.. _l-example-filter:

Compares filtering implementations (numpy, cython)
==================================================

The benchmark looks into different ways to implement
thresholding: every value of a vector superior to *mx*
is replaced by *mx*. It compares several implementation
to :epkg:`numpy`.

* :func:`cfilter_dmax <td3a_cpp.tutorial.experiment_cython.cfilter_dmax>`
* :func:`cfilter_dmax2 <td3a_cpp.tutorial.experiment_cython.cfilter_dmax2>`
* :func:`cfilter_dmax4 <td3a_cpp.tutorial.experiment_cython.cfilter_dmax4>`
* :func:`cfilter_dmax16 <td3a_cpp.tutorial.experiment_cython.cfilter_dmax16>`
* :func:`cyfilter_dmax <td3a_cpp.tutorial.experiment_cython.cyfilter_dmax>`
* :func:`filter_dmax_cython
  <td3a_cpp.tutorial.experiment_cython.filter_dmax_cython>`
* :func:`filter_dmax_cython_optim
  <td3a_cpp.tutorial.experiment_cython.filter_dmax_cython_optim>`
* :func:`pyfilter_dmax <td3a_cpp.tutorial.experiment_cython.pyfilter_dmax>`
"""

import pprint
import numpy
import matplotlib.pyplot as plt
from pandas import DataFrame
from td3a_cpp.tutorial.experiment_cython import (
    pyfilter_dmax, filter_dmax_cython,
    filter_dmax_cython_optim,
    cyfilter_dmax,
    cfilter_dmax, cfilter_dmax2,
    cfilter_dmax16, cfilter_dmax4
)
from td3a_cpp.tools import measure_time_dim


def get_vectors(fct, n, h=200, dtype=numpy.float64):
    ctxs = [dict(va=numpy.random.randn(n).astype(dtype),
                 fil=fct,
                 mx=numpy.float64(0),
                 x_name=n)
            for n in range(10, n, h)]
    return ctxs


def numpy_filter(va, mx):
    va[va > mx] = mx


all_res = []
for fct in [numpy_filter,
            pyfilter_dmax, filter_dmax_cython,
            filter_dmax_cython_optim,
            cyfilter_dmax,
            cfilter_dmax, cfilter_dmax2,
            cfilter_dmax16, cfilter_dmax4]:

    print(fct)
    ctxs = get_vectors(fct, 1000 if fct == pyfilter_dmax else 40000)
    res = list(measure_time_dim('fil(va, mx)', ctxs, verbose=1))
    for r in res:
        r['fct'] = fct.__name__
    all_res.extend(res)

pprint.pprint(all_res[:2])

#############################
# Let's display the results
# +++++++++++++++++++++++++

cc = DataFrame(all_res)
cc['N'] = cc['x_name']

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
cc[cc.N <= 1100].pivot('N', 'fct', 'average').plot(
    logy=True, ax=ax[0, 0])
cc[cc.fct != 'pyfilter_dmax'].pivot('N', 'fct', 'average').plot(
    logy=True, ax=ax[0, 1])
cc[cc.fct != 'pyfilter_dmax'].pivot('N', 'fct', 'average').plot(
    logy=True, logx=True, ax=ax[1, 1])
cc[(cc.fct.str.contains('cfilter') |
    cc.fct.str.contains('numpy'))].pivot('N', 'fct', 'average').plot(
    logy=True, ax=ax[1, 0])
ax[0, 0].set_title("Comparison of filter implementations")
ax[0, 1].set_title("Comparison of filter implementations\n"
                   "without pyfilter_dmax")

#################################
# The results depends on the machine, its
# number of cores, the compilation settings
# of :epkg:`numpy` or this module.

plt.show()
