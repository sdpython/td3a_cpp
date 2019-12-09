"""
Many implementations of the dot product.
See `Cython documentation <http://docs.cython.org/en/latest/>`_.
"""
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
from libc.stdio cimport printf
from libc.math cimport NAN

import numpy
cimport numpy
cimport cython
numpy.import_array()

def pyfilter_dmax(va, mx):
    """
    Replaces all value superior to *mx* by *mx*.
    Python inside cython.
    
    :param va: first vector
    :param mx: maximum
    """
    for i in range(va.shape[0]):
        if va[i] > mx:
            va[i] = mx


def filter_dmax_cython(double[::1] va, double mx):
    """
    Replaces all value superior to *mx* by *mx*.
    Simple cython.
    
    :param va: first vector
    :param mx: maximum
    """
    for i in range(va.shape[0]):
        if va[i] > mx:
            va[i] = mx


@cython.boundscheck(False)
@cython.wraparound(False)
def filter_dmax_cython_optim(double[::1] va, double mx):
    """
    Replaces all value superior to *mx* by *mx*.
    Simple cython with no bound checked, no wrap around.
    
    :param va: first vector
    :param mx: maximum
    """
    for i in range(va.shape[0]):
        if va[i] > mx:
            va[i] = mx


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _cyfilter_dmax(double[::1] va, double mx) nogil:
    """
    Replaces all value superior to *mx* by *mx*.
    
    :param va: first vector
    :param mx: maximum
    """
    for i in range(va.shape[0]):
        if va[i] > mx:
            va[i] = mx


@cython.boundscheck(False)
@cython.wraparound(False)
def cyfilter_dmax(double[::1] va, double mx):
    """
    Replaces all value superior to *mx* by *mx*.
    Wraps a C function implemented in cython.
    
    :param va: first vector
    :param mx: maximum
    """
    with nogil:
        _cyfilter_dmax(va, mx)


cdef extern from "experiment_cython_.h":
    cdef void filter_dmax(double *p1, cython.int size, double mx) nogil
    cdef void filter_dmax2(double *p1, int size, double mx) nogil
    cdef void filter_dmax16(double *p1, int size, double mx) nogil
    cdef void filter_dmax4(double *p1, int size, double mx) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
def cfilter_dmax(double[::1] va, double mx):
    """
    Replaces all value superior to *mx* by *mx*.
    Wraps a C function implemented in C.
        
    :param va: first vector
    :param mx: maximum
    """
    with nogil:
        filter_dmax(&va[0], va.shape[0], mx)


@cython.boundscheck(False)
@cython.wraparound(False)
def cfilter_dmax2(double[::1] va, double mx):
    """
    Replaces all value superior to *mx* by *mx*.
    Wraps a C function implemented in C,
    but uses operator ``? :`` instead of keyword ``if``.
        
    :param va: first vector
    :param mx: maximum
    """
    with nogil:
        filter_dmax2(&va[0], va.shape[0], mx)


@cython.boundscheck(False)
@cython.wraparound(False)
def cfilter_dmax16(double[::1] va, double mx):
    """
    Replaces all value superior to *mx* by *mx*.
    Wraps a C function implemented in C,
    but uses operator ``? :`` instead of keyword ``if``.
    Goes 16 by 16.
        
    :param va: first vector
    :param mx: maximum
    """
    with nogil:
        filter_dmax16(&va[0], va.shape[0], mx)


@cython.boundscheck(False)
@cython.wraparound(False)
def cfilter_dmax4(double[::1] va, double mx):
    """
    Replaces all value superior to *mx* by *mx*.
    Wraps a C function implemented in C,
    but uses operator ``? :`` instead of keyword ``if``.
    Goes 4 by 4.
        
    :param va: first vector
    :param mx: maximum
    """
    with nogil:
        filter_dmax4(&va[0], va.shape[0], mx)
