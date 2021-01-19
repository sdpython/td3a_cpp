"""
TD 2021/01/19.
"""
from cython.parallel import prange
cimport cython
from cpython cimport array
import array

import numpy as pynumpy
cimport numpy as cnumpy
cnumpy.import_array()


def multiply_matrix(m1, m2):
    m3 = pynumpy.zeros((m1.shape[0], m2.shape[1]), dtype=m1.dtype)
    for i in range(0, m1.shape[0]):
        for j in range(0, m2.shape[1]):
            for k in range(0, m1.shape[1]):
                m3[i, j] += m1[i, k] * m2[k, j]
    return m3


cdef double[:, :] _c_multiply_matrix(double[:, :] m1, double[:, :] m2):
    m3 = pynumpy.zeros((m1.shape[0], m2.shape[1]), dtype=pynumpy.float64)
    for i in range(0, m1.shape[0]):
        for j in range(0, m2.shape[1]):
            for k in range(0, m1.shape[1]):
                m3[i, j] += m1[i, k] * m2[k, j]
    return m3


def c_multiply_matrix(double[:, :] m1, double[:, :] m2):
    return _c_multiply_matrix(m1, m2)
