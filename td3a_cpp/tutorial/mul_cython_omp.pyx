"""
Many implementations of the dot product.
See `Cython documentation <http://docs.cython.org/en/latest/>`_.
"""
from cython.parallel import prange
cimport numpy as cnumpy
import numpy as pynumpy
cimport cython
from cpython cimport array
import array
cimport openmp
cnumpy.import_array()


cdef int _dmul_cython_omp(const double* va, const double* vb, double* res,
                          Py_ssize_t ni, Py_ssize_t nj, Py_ssize_t nk,
                          cython.int algo, cython.int parallel) nogil:
    """
    matrix multiplication product implemented with cython and C types
    using :epkg:`prange` (:epkg:`openmp` in :epkg:`cython`).

    :param va: first matrix, dtype must be float64
    :param vb: second matrix, dtype must be float64
    :param res: result of the matrix multiplication
    :param algo: algorithm (see below)
    :param parallel: kind of parallelization (see below)
    :return: matrix multiplication
    """
    cdef double s
    cdef Py_ssize_t p, i, j, k
    if parallel == 0:
        if algo == 0:
            for i in range(0, ni):
                for j in range(0, nj):
                    s = 0
                    for k in range(0, nk):
                        s += va[i * nk + k] * vb[k * nj + j]
                    res[i * nj + j] = s
            return 1

        if algo == 1:
            for j in range(0, nj):
                for i in range(0, ni):
                    s = 0
                    for k in range(0, nk):
                        s += va[i * nk + k] * vb[k * nj + j]
                    res[i * nj + j] = s
            return 1

        if algo == 2:
            for k in range(0, nk):
                for i in range(0, ni):
                    for j in range(0, nj):
                        res[i * nj + j] += va[i * nk + k] * vb[k * nj + j]
            return 1

    if parallel == 1:
        if algo == 0:
            for p in prange(0, ni):
                for j in range(0, nj):
                    res[p * nj + j] = 0
                    for k in range(0, nk):
                        res[p * nj + j] += va[p * nk + k] * vb[k * nj + j]
            return 1

        if algo == 1:
            for p in prange(0, nj):
                for i in range(0, ni):
                    res[i * nj + p] = 0
                    for k in range(0, nk):
                        res[i * nj + p] += va[i * nk + k] * vb[k * nj + p]
            return 1

        if algo == 2:
            for p in prange(0, nk):
                for i in range(0, ni):
                    for j in range(0, nj):
                        res[i * nj + j] += va[i * nk + p] * vb[p * nj + j]
            return 1
    
    return 0


cdef extern from "mul_cython_omp_.h":
    cdef double vector_ddot_product_pointer16_sse(const double *p1, const double *p2, cython.int size) nogil


cdef int _dmul_cython_omp_t(const double* va, const double* vb, double* res,
                            Py_ssize_t ni, Py_ssize_t nj, Py_ssize_t nk,
                            cython.int algo, cython.int parallel) nogil:
    """
    matrix multiplication product implemented with cython and C types
    using :epkg:`prange` (:epkg:`openmp` in :epkg:`cython`).

    :param va: first matrix, dtype must be float64
    :param vb: second matrix, dtype must be float64
    :param res: result of the matrix multiplication
    :param algo: algorithm (see below)
    :param parallel: kind of parallelization (see below)
    :return: matrix multiplication
    """
    cdef double s
    cdef Py_ssize_t p, i, j, k
    if parallel == 0:
        if algo == 0:
            for i in range(0, ni):
                for j in range(0, nj):
                    res[i * nj + j] = vector_ddot_product_pointer16_sse(&va[i * nk], &vb[j * nk], nk)
            return 1

        if algo == 1:
            for j in range(0, nj):
                for i in range(0, ni):
                    res[i * nj + j] = vector_ddot_product_pointer16_sse(&va[i * nk], &vb[j * nk], nk)
            return 1

        if algo == 2:
            for k in range(0, nk):
                for i in range(0, ni):
                    for j in range(0, nj):
                        res[i * nj + j] += va[i * nk + k] * vb[j * nk + k]
            return 1

    if parallel == 1:
        if algo == 0:
            for p in prange(0, ni):
                for j in range(0, nj):
                    res[p * nj + j] = vector_ddot_product_pointer16_sse(&va[p * nk], &vb[j * nk], nk)
                        
            return 1

        if algo == 1:
            for p in prange(0, nj):
                for i in range(0, ni):
                    res[i * nj + p] = vector_ddot_product_pointer16_sse(&va[i * nk], &vb[p * nk], nk)
                        
            return 1

        if algo == 2:
            for p in prange(0, nk):
                for i in range(0, ni):
                    for j in range(0, nj):
                        res[i * nj + j] += va[i * nk + p] * vb[j * nk + p]
            return 1
    
    return 0


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def dmul_cython_omp(const double[:, :] va, const double[:, :] vb,
                    cython.int algo=0, cython.int parallel=0,
                    cython.int b_trans=0):
    """
    matrix multiplication product implemented with cython and C types
    using :epkg:`prange` (:epkg:`openmp` in :epkg:`cython`).

    :param va: first matrix, dtype must be float64
    :param vb: second matrix, dtype must be float64
    :param algo: algorithm (see below)
    :param parallel: kind of parallelization (see below)
    :return: matrix multiplication
    """    
    cdef double[:, :] pres
    cdef cython.int r
    cdef const double *pva
    cdef const double *pvb
    cdef double *ppres
    if b_trans:
        if va.shape[1] != vb.shape[1]:
            raise ValueError(
                "Shape mismatch, cannot multiply %r and %r" % (va.shape, vb.shape))
        res = pynumpy.zeros((va.shape[0], vb.shape[0]), dtype=pynumpy.double)    
        pres = res
        pva = &va[0, 0]
        pvb = &vb[0, 0]
        ppres = &pres[0, 0]
        with nogil:
            r = _dmul_cython_omp_t(pva, pvb, ppres,
                                   va.shape[0], vb.shape[0], va.shape[1],
                                   algo, parallel)
        if r != 1:
            raise RuntimeError(
                "Unknown value for algo=%r or parallel=%d r=%d" % (algo, parallel, r))
        return res
    else:
        if va.shape[1] != vb.shape[0]:
            raise ValueError(
                "Shape mismatch, cannot multiply %r and %r" % (va.shape, vb.shape))
        res = pynumpy.zeros((va.shape[0], vb.shape[1]), dtype=pynumpy.double)    
        pres = res
        pva = &va[0, 0]
        pvb = &vb[0, 0]
        ppres = &pres[0, 0]
        with nogil:
            r = _dmul_cython_omp(pva, pvb, ppres,
                                 va.shape[0], vb.shape[1], va.shape[1],
                                 algo, parallel)
        if r != 1:
            raise RuntimeError(
                "Unknown value for algo=%r or parallel=%d r=%r" % (algo, parallel, r))
        return res

