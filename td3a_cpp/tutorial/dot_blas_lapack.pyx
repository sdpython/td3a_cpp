"""
Direct calls to libraries :epkg:`BLAS` and :epkg:`LAPACK`.
"""
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
from libc.stdio cimport printf
from libc.math cimport NAN

import numpy
cimport numpy
cimport cython
numpy.import_array()
cimport scipy.linalg.cython_blas as cython_blas
# cimport scipy.linalg.cython_lapack as cython_lapack


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _cblas_ddot(int n, const double* x, int sx,
                        const double* y, int sy) nogil:    
    cdef float r
    with nogil:
        r = cython_blas.ddot(&n, x, &sx, y, &sy)
    return r


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float _cblas_sdot(int n, const float* x, int sx,
                       const float* y, int sy) nogil:
    cdef float r
    with nogil:
        r = cython_blas.sdot(&n, x, &sx, y, &sy)
    return r


@cython.boundscheck(False)
@cython.wraparound(False)
def cblas_ddot(const double[::1] x, const double[::1] y):
    """
    Computes a dot product with
    `cblas_ddot <https://software.intel.com/en-us/
    mkl-developer-reference-c-cblas-dot>`_.
    
    :param x: first vector, dtype must be float64
    :param y: second vector, dtype must be float64
    :return: dot product
    """
    if x.shape[0] != y.shape[0]:        
        raise ValueError("Vector must have same shape.")
    return _cblas_ddot(x.shape[0], &x[0], x.strides[0] / x.itemsize,
                       &y[0], y.strides[0] / y.itemsize)
    

@cython.boundscheck(False)
@cython.wraparound(False)
def cblas_sdot(const float[::1] x, const float[::1] y):
    """
    Computes a dot product with
    `cblas_sdot <https://software.intel.com/en-us/
    mkl-developer-reference-c-cblas-dot>`_.
    
    :param x: first vector, dtype must be float32
    :param y: second vector, dtype must be float32
    :return: dot product
    """
    if x.shape[0] != y.shape[0]:        
        raise ValueError("Vector must have same shape.")
    return _cblas_sdot(x.shape[0], &x[0], x.strides[0] / x.itemsize,
                       &y[0], y.strides[0] / y.itemsize)


