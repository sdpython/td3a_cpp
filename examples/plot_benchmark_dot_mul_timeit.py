"""

.. _l-example-mul-timeit:

Compares mul implementations with timeit
========================================

:epkg:`numpy` has a very fast implementation of
matrix multiplication. There are many ways to be slower.
The following uses :epkg:`timeit` to compare implementations.

.. contents::
    :local:

Preparation
+++++++++++
"""
import timeit
import numpy

from td3a_cpp.tutorial.td_mul_cython import (
    multiply_matrix, c_multiply_matrix,
    c_multiply_matrix_parallel,
    c_multiply_matrix_parallel_transposed as cmulparamtr)


va = numpy.random.randn(150, 100).astype(numpy.float64)
vb = numpy.random.randn(100, 100).astype(numpy.float64)
ctx = {
    'va': va, 'vb': vb, 'c_multiply_matrix': c_multiply_matrix,
    'multiply_matrix': multiply_matrix,
    'c_multiply_matrix_parallel': c_multiply_matrix_parallel,
    'c_multiply_matrix_parallel_transposed': cmulparamtr}

##########################################
# Measures
# ++++++++
#
# numpy
res0 = timeit.timeit('va @ vb', number=100, globals=ctx)
print("numpy time", res0)

###########################
# python implementation

res1 = timeit.timeit(
    'multiply_matrix(va, vb)', number=10, globals=ctx)
print('python implementation', res1)


###########################
# cython implementation

res2 = timeit.timeit(
    'c_multiply_matrix(va, vb)', number=100, globals=ctx)
print('cython implementation', res2)


###########################
# cython implementation parallelized

res3 = timeit.timeit(
    'c_multiply_matrix_parallel(va, vb)', number=100, globals=ctx)
print('cython implementation parallelized', res3)


###########################
# cython implementation parallelized, AVX + transposed

res4 = timeit.timeit(
    'c_multiply_matrix_parallel_transposed(va, vb)', number=100, globals=ctx)
print('cython implementation parallelized avx', res4)


############################
# Speed up...

print("numpy is %f faster than pure python." % (res1 / res0))
print("numpy is %f faster than cython." % (res2 / res0))
print("numpy is %f faster than parallelized cython." % (res3 / res0))
print("numpy is %f faster than avx parallelized cython." % (res4 / res0))
