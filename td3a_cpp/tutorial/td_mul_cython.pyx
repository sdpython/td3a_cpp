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


cdef void _c_multiply_matrix(double[:, :] m1, double[:, :] m2,
                             double[:, :] m3,
                             cython.int ni, cython.int nj, cython.int nk) nogil:
    cdef cython.int i, j, k    
    for i in prange(0, ni):
        for j in range(0, nj):
            for k in range(0, nk):
                m3[i, j] += m1[i, k] * m2[k, j]


def c_multiply_matrix(double[:, :] m1, double[:, :] m2):
    m3 = pynumpy.zeros((m1.shape[0], m2.shape[1]), dtype=pynumpy.float64)
    cdef cython.int ni = m1.shape[0]
    cdef cython.int nj = m2.shape[1]
    cdef cython.int nk = m1.shape[1]
    _c_multiply_matrix(m1, m2, m3, ni, nj, nk)
    return m3
