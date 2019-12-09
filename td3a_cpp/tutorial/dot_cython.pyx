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

def dot_product(va, vb):
    """
    Python dot product but in :epkg:`cython` file.
    
    :param va: first vector
    :param vb: second vector
    :return: dot product
    """
    s = 0
    for i in range(va.shape[0]):
        s += va[i] * vb[i]
    return s


def ddot_cython_array(const double[::1] va, const double[::1] vb):
    """
    dot product implemented with C types.
    
    :param va: first vector, dtype must be float64
    :param vb: second vector, dtype must be float64
    :return: dot product
    """
    if va.shape[0] != vb.shape[0]:        
        raise ValueError("Vectors must have same shape.")
    cdef double s = 0
    for i in range(va.shape[0]):
        s += va[i] * vb[i]
    return s
    

def sdot_cython_array(const float[::1] va, const float[::1] vb):
    """
    dot product implemented with C types.
    
    :param va: first vector, dtype must be float32
    :param vb: second vector, dtype must be float32
    :return: dot product
    """
    if va.shape[0] != vb.shape[0]:        
        raise ValueError("Vectors must have same shape.")
    cdef float s = 0
    for i in range(va.shape[0]):
        s += va[i] * vb[i]
    return s
    

@cython.boundscheck(False)
@cython.wraparound(False)
def ddot_cython_array_optim(const double[::1] va, const double[::1] vb):
    """
    dot product implemented with C types with
    disabled checkings (see :epkg:`compiler directives`.
    
    :param va: first vector, dtype must be float64
    :param vb: second vector, dtype must be float64
    :return: dot product
    """
    if va.shape[0] != vb.shape[0]:        
        raise ValueError("Vectors must have same shape.")
    cdef double s = 0
    for i in range(va.shape[0]):
        s += va[i] * vb[i]
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
def sdot_cython_array_optim(const float[::1] va, const float[::1] vb):
    """
    dot product implemented with C types with
    disabled checkings (see :epkg:`compiler directives`.
    
    :param va: first vector, dtype must be float32
    :param vb: second vector, dtype must be float32
    :return: dot product
    """
    if va.shape[0] != vb.shape[0]:        
        raise ValueError("Vectors must have same shape.")
    cdef float s = 0
    for i in range(va.shape[0]):
        s += va[i] * vb[i]
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _ddot_array(const double[::1] va, const double[::1] vb) nogil:
    """
    dot product implemented with C types with
    disabled checkings (see :epkg:`compiler directives`),
    and :epkg:`nogil`.
        
    :param va: first vector, dtype must be float64
    :param vb: second vector, dtype must be float64
    :return: dot product
    """
    if va.shape[0] != vb.shape[0]:        
        raise ValueError("Vectors must have same shape.")
    cdef double s = 0
    for i in range(va.shape[0]):
        s += va[i] * vb[i]
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float _sdot_array(const float[::1] va, const float[::1] vb) nogil:
    """
    dot product implemented with C types with
    disabled checkings (see :epkg:`compiler directives`),
    and :epkg:`nogil`.
        
    :param va: first vector, dtype must be float32
    :param vb: second vector, dtype must be float32
    :return: dot product
    """
    if va.shape[0] != vb.shape[0]:        
        raise ValueError("Vectors must have same shape.")
    cdef float s = 0
    for i in range(va.shape[0]):
        s += va[i] * vb[i]
    return s

    
@cython.boundscheck(False)
@cython.wraparound(False)
def ddot_array(const double[::1] va, const double[::1] vb):
    """
    dot product implemented with C types with
    disabled checkings (see :epkg:`compiler directives`),
    and :epkg:`nogil`. It is a wrapper for a C function
    as they cannot be exposed to the python world
    (gil is disabled).
        
    :param va: first vector, dtype must be float64
    :param vb: second vector, dtype must be float64
    :return: dot product
    """
    cdef double r;
    with nogil:
        r = _ddot_array(va, vb)
    return r


@cython.boundscheck(False)
@cython.wraparound(False)
def sdot_array(const float[::1] va, const float[::1] vb):
    """
    dot product implemented with C types with
    disabled checkings (see :epkg:`compiler directives`),
    and :epkg:`nogil`. It is a wrapper for a C function
    as they cannot be exposed to the python world
    (gil is disabled).
        
    :param va: first vector, dtype must be float32
    :param vb: second vector, dtype must be float32
    :return: dot product
    """
    cdef float r;
    with nogil:
        r = _sdot_array(va, vb)
    return r


cdef extern from "dot_cython_.h":
    cdef double vector_ddot_product_pointer16(const double *p1, const double *p2, cython.int size) nogil
    cdef double vector_ddot_product_pointer16_sse(const double *p1, const double *p2, cython.int size) nogil
    cdef float vector_sdot_product_pointer16(const float *p1, const float *p2, cython.int size) nogil
    cdef float vector_sdot_product_pointer16_sse(const float *p1, const float *p2, cython.int size) nogil
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
def ddot_array_16(const double[::1] va, const double[::1] vb):
    """
    dot product implemented with C types with
    disabled checkings (see :epkg:`compiler directives`),
    and :epkg:`nogil`. It is a wrapper for a C function
    as they cannot be exposed to the python world
    (gil is disabled). Computation is done 16x16 to
    benefit from :epkg:`branching`.
        
    :param va: first vector, dtype must be float64
    :param vb: second vector, dtype must be float64
    :return: dot product
    """
    cdef double r;
    with nogil:
        r = vector_ddot_product_pointer16(&va[0], &vb[0], va.shape[0])
    return r


@cython.boundscheck(False)
@cython.wraparound(False)
def sdot_array_16(const float[::1] va, const float[::1] vb):
    """
    dot product implemented with C types with
    disabled checkings (see :epkg:`compiler directives`),
    and :epkg:`nogil`. It is a wrapper for a C function
    as they cannot be exposed to the python world
    (gil is disabled). Computation is done 16x16 to
    benefit from :epkg:`branching`.
        
    :param va: first vector, dtype must be float32
    :param vb: second vector, dtype must be float32
    :return: dot product
    """
    cdef float r;
    with nogil:
        r = vector_sdot_product_pointer16(&va[0], &vb[0], va.shape[0])
    return r


@cython.boundscheck(False)
@cython.wraparound(False)
def ddot_array_16_sse(const double[::1] va, const double[::1] vb):
    """
    dot product implemented with C types with
    disabled checkings (see :epkg:`compiler directives`),
    and :epkg:`nogil`. It is a wrapper for a C function
    as they cannot be exposed to the python world
    (gil is disabled). Computation is done 16x16 to
    benefit from :epkg:`branching` and uses :epkg:`AVX`
    instructions.
        
    :param va: first vector, dtype must be float64
    :param vb: second vector, dtype must be float64
    :return: dot product
    """
    cdef double r;
    with nogil:
        r = vector_ddot_product_pointer16_sse(&va[0], &vb[0], va.shape[0])
    return r
    

@cython.boundscheck(False)
@cython.wraparound(False)
def sdot_array_16_sse(const float[::1] va, const float[::1] vb):
    """
    dot product implemented with C types with
    disabled checkings (see :epkg:`compiler directives`),
    and :epkg:`nogil`. It is a wrapper for a C function
    as they cannot be exposed to the python world
    (gil is disabled). Computation is done 16x16 to
    benefit from :epkg:`branching` and uses :epkg:`AVX`
    instructions.
        
    :param va: first vector, dtype must be float32
    :param vb: second vector, dtype must be float32
    :return: dot product
    """
    cdef float r;
    with nogil:
        r = vector_sdot_product_pointer16_sse(&va[0], &vb[0], va.shape[0])
    return r
    
