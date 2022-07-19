"""

.. _l-example-associativity:

Associativity and matrix multiplication
=======================================

The matrix multiplication `m1 @ m2 @ m3` can be done
in two different ways: `(m1 @ m2) @ m3` or `m1 @ (m2 @ m3)`.
Are these two orders equivalent or is there a better order?

.. contents::
    :local:
"""

import pprint
import numpy
import matplotlib.pyplot as plt
from pandas import DataFrame
from tqdm import tqdm
from td3a_cpp.tools import measure_time

##############################
# First try
# +++++++++
#

m1 = numpy.random.rand(100, 100)
m2 = numpy.random.rand(100, 10)
m3 = numpy.random.rand(10, 100)

m = m1 @ m2 @ m3

print(m.shape)

mm1 = (m1 @ m2) @ m3
mm2 = m1 @ (m2 @ m3)

print(mm1.shape, mm2.shape)

t1 = measure_time(lambda: (m1 @ m2) @ m3, context={}, number=100, repeat=100)
pprint.pprint(t1)

t2 = measure_time(lambda: m1 @ (m2 @ m3), context={}, number=100, repeat=100)
pprint.pprint(t2)

###########################################
# With different sizes
# ++++++++++++++++++++

obs = []
for i in tqdm([50, 100, 125, 150, 175, 200]):

    m1 = numpy.random.rand(i, i)
    m2 = numpy.random.rand(i, 10)
    m3 = numpy.random.rand(10, i)

    t1 = measure_time(lambda: (m1 @ m2) @ m3,
                      context={}, number=100, repeat=100)
    t1['formula'] = "(m1 @ m2) @ m3"
    t1['size'] = i
    obs.append(t1)
    t2 = measure_time(lambda: m1 @ (m2 @ m3),
                      context={}, number=100, repeat=100)
    t2['formula'] = "m1 @ (m2 @ m3)"
    t2['size'] = i
    obs.append(t2)

df = DataFrame(obs)
piv = df.pivot("size", "formula", "average")
piv

###########################################
# Graph
# +++++

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
piv.plot(logx=True, logy=True, ax=ax[0],
         title=f"{m1.shape!r} @ {m2.shape!r} @ "
               f"{m3.shape!r}".replace("200", "size"))
piv["ratio"] = piv["m1 @ (m2 @ m3)"] / piv["(m1 @ m2) @ m3"]
piv[['ratio']].plot(ax=ax[1])

plt.show()
