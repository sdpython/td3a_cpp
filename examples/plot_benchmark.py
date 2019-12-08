"""

.. _l-example-dot:

Compares dot implementation
===========================

:epkg:`numpy` has a very fast implementation of
the dot product. It is difficult to be better and very easy
to be slower. This example looks into a couple of slower
implementations.

.. content::
    :local:
"""

import pprint
import numpy
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
from td3a_cpp.tutorial import pydot
from td3a_cpp.tools import measure_time_dim

##############################
# python dot: pydot
# +++++++++++++++++
#
# The first function :func:`pydot
# <td3a_cpp.tutorial.pydot>` uses
# python to implement the dot product.

ctxs = [dict(va=numpy.arange(n).astype(numpy.float64),
             vb=numpy.arange(n).astype(numpy.float64) - 5,
             pydot=pydot,
             x_name=n)
        for n in range(10, 1000, 100)]

res_pydot = list(measure_time_dim('pydot(va, vb)', ctxs, verbose=1))

pprint.pprint(res_pydot[:2])

##############################
# numpy dot
# +++++++++
#

for ctx in ctxs:
    ctx['dot'] = numpy.dot

res_dot = list(measure_time_dim('dot(va, vb)', ctxs, verbose=1))

pprint.pprint(res_dot[:2])

#############################
# Let's display the results
# +++++++++++++++++++++++++

df1 = DataFrame(res_pydot)
df1['fct'] = 'pydot'
df2 = DataFrame(res_dot)
df2['fct'] = 'numpy.dot'
cc = concat([df1, df2])
cc['N'] = cc['x_name']
piv = cc.pivot('N', 'fct', 'average')
piv.plot(logy=True)

###################
# :epkg:`numpy` is cleary faster.

plt.show()
