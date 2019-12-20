"""

.. _l-example-dot-process:

Parallelization of a dot product with processes (concurrent.futures)
====================================================================

Uses processes to parallelize a dot product is not
a very solution becausep processes do not share memory,
they need to exchange data. This parallelisation
is efficient if the ratio *exchanged data / computation time*
is low. This example uses :epkg:`concurrent.futures`.
The cost of creating new processes is also significant.
"""

import numpy
from tqdm import tqdm
from pandas import DataFrame
import matplotlib.pyplot as plt
import concurrent.futures as cf
from td3a_cpp.tools import measure_time


def parallel_numpy_dot(va, vb, max_workers=2):
    if max_workers == 2:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as e:
            m = va.shape[0] // 2
            f1 = e.submit(numpy.dot, va[:m], vb[:m])
            f2 = e.submit(numpy.dot, va[m:], vb[m:])
            return f1.result() + f2.result()
    elif max_workers == 3:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as e:
            m = va.shape[0] // 3
            m2 = va.shape[0] * 2 // 3
            f1 = e.submit(numpy.dot, va[:m], vb[:m])
            f2 = e.submit(numpy.dot, va[m:m2], vb[m:m2])
            f3 = e.submit(numpy.dot, va[m2:], vb[m2:])
            return f1.result() + f2.result() + f3.result()
    else:
        raise NotImplementedError()

###########################
# We check that it returns the same values.


va = numpy.random.randn(100).astype(numpy.float64)
vb = numpy.random.randn(100).astype(numpy.float64)
print(parallel_numpy_dot(va, vb), numpy.dot(va, vb))


###############################
# Let's benchmark.
res = []
for n in tqdm([100000, 1000000, 10000000, 100000000]):
    va = numpy.random.randn(n).astype(numpy.float64)
    vb = numpy.random.randn(n).astype(numpy.float64)

    m1 = measure_time('dot(va, vb, 2)',
                      dict(va=va, vb=vb, dot=parallel_numpy_dot))
    m2 = measure_time('dot(va, vb)',
                      dict(va=va, vb=vb, dot=numpy.dot))
    res.append({'N': n, 'numpy.dot': m2['average'],
                'futures': m1['average']})

df = DataFrame(res).set_index('N')
print(df)
df.plot(logy=True, logx=True)
plt.title("Parallel / numpy dot")

#######################################
# The parallelisation is inefficient
# unless the vectors are big.

plt.show()
