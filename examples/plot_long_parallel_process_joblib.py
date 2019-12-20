"""

.. _l-example-dot-joblib:

Parallelization of a dot product with processes (joblib)
========================================================

Uses processes to parallelize a dot product is not
a very solution becausep processes do not share memory,
they need to exchange data. This parallelisation
is efficient if the ratio *exchanged data / computation time*
is low. :epkg:`joblib` is used by :epkg:`scikit-learn`.
The cost of creating new processes is also significant.
"""

import numpy
from tqdm import tqdm
from pandas import DataFrame
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from td3a_cpp.tools import measure_time


def parallel_dot_joblib(va, vb, max_workers=2):
    dh = va.shape[0] // max_workers
    k = 2
    dhk = dh // k
    if dh != float(va.shape[0]) / max_workers:
        raise RuntimeError("size must be a multiple of max_workers.")

    r = Parallel(n_jobs=max_workers, backend="loky")(
            delayed(numpy.dot)(va[i*dhk:i*dhk+dhk], vb[i*dhk:i*dhk+dhk])
            for i in range(max_workers * k))
    return sum(r)

###########################
# We check that it returns the same values.


va = numpy.random.randn(100).astype(numpy.float64)
vb = numpy.random.randn(100).astype(numpy.float64)
print(parallel_dot_joblib(va, vb), numpy.dot(va, vb))


###############################
# Let's benchmark.


res = []
for n in tqdm([1000, 2000]):
    va = numpy.random.randn(n).astype(numpy.float64)
    vb = numpy.random.randn(n).astype(numpy.float64)

    m1 = measure_time('dot(va, vb, 2)',
                      dict(va=va, vb=vb, dot=parallel_dot_joblib),
                      repeat=1)
    m2 = measure_time('dot(va, vb)',
                      dict(va=va, vb=vb, dot=numpy.dot))
    res.append({'N': n, 'numpy.dot': m2['average'],
                'joblib': m1['average']})

df = DataFrame(res).set_index('N')
print(df)
df.plot(logy=True, logx=True)
plt.title("Parallel / numpy dot")

#######################################
# The parallelisation is inefficient.

plt.show()
