"""
Many implementations of the dot product.
See `Cython documentation <http://docs.cython.org/en/latest/>`_.
"""
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
from libc.stdio cimport printf
from libc.math cimport NAN

import numpy
from cython.parallel import prange, parallel
cimport numpy
cimport cython
cimport openmp
numpy.import_array()


cdef double _ddot_cython_array_omp(const double[::1] va, const double[::1] vb,
                                   int chunksize, int schedule) nogil:
    """
    dot product implemented with cython and C types
    using :epkg:`prange` (:epkg:`openmp` in :epkg:`cython`).
    
    :param va: first vector, dtype must be float64
    :param vb: second vector, dtype must be float64
    :return: dot product
    """
    cdef int n = va.shape[0]
    cdef Py_ssize_t i

    cdef double s = 0
    if schedule == 1:
        for i in prange(n, schedule='static', chunksize=chunksize):
            s += va[i] * vb[i]
    elif schedule == 2:
        for i in prange(n, schedule='dynamic', chunksize=chunksize):
            s += va[i] * vb[i]
    else:
        for i in prange(n):
            s += va[i] * vb[i]
    return s


def ddot_cython_array_omp(const double[::1] va, const double[::1] vb,
                          cython.int chunksize=32, cython.int schedule=0):
    """
    dot product implemented with cython and C types
    using :epkg:`prange` (:epkg:`openmp` in :epkg:`cython`).
    
    :param va: first vector, dtype must be float64
    :param vb: second vector, dtype must be float64
    :param chunksize: see :epkg:`prange`
    :param schedule: see :epkg:`prange`
    :return: dot product
    """
    if va.shape[0] != vb.shape[0]:        
        raise ValueError("Vectors must have same shape.")
    cdef double s
    with nogil:
        s = _ddot_cython_array_omp(va, vb, chunksize, schedule)
    return s


cdef extern from "dot_cython_omp_.h":
    cdef cython.int get_omp_max_threads_cpp() nogil
    cdef double vector_ddot_openmp(const double *p1, const double *p2, cython.int size, cython.int nthreads) nogil
    cdef double vector_ddot_openmp_16(const double *p1, const double *p2, cython.int size, cython.int nthreads) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
def get_omp_max_threads():
    """
    Returns the number of threads.
    """
    cdef cython.int i
    with nogil:
        i = get_omp_max_threads_cpp()
    return i


@cython.boundscheck(False)
@cython.wraparound(False)
def ddot_array_openmp(const double[::1] va, const double[::1] vb):
    """
    dot product using :epkg:`openmp` inside C++ code.
    
    :param va: first vector, dtype must be float64
    :param vb: second vector, dtype must be float64
    :return: dot product
    """
    cdef double r;
    with nogil:
        r = vector_ddot_openmp(&va[0], &vb[0], va.shape[0], 0)
    return r


@cython.boundscheck(False)
@cython.wraparound(False)
def ddot_array_openmp_16(const double[::1] va, const double[::1] vb):
    """
    dot product using :epkg:`openmp` inside C++ code,
    parallelizes 16x16 computation.
    
    :param va: first vector, dtype must be float64
    :param vb: second vector, dtype must be float64
    :return: dot product
    """
    cdef double r;
    with nogil:
        r = vector_ddot_openmp_16(&va[0], &vb[0], va.shape[0], 0)
    return r
