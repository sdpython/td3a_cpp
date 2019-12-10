"""
Unit tests for ``random_strategy``.
"""
import unittest
import numpy
from td3a_cpp.tutorial.dot_cython_omp import (
    ddot_cython_array_omp,
    ddot_array_openmp,
    get_omp_max_threads,
    ddot_array_openmp_16,
)


class TestTutorialDotOmp(unittest.TestCase):

    def test_ddot_cython_array_omp(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = numpy.dot(va, vb)
        res2 = ddot_cython_array_omp(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-12)
        res2 = ddot_cython_array_omp(va, vb, schedule=1)
        self.assertTrue(abs(res1 - res2) <= 1e-12)
        res2 = ddot_cython_array_omp(va, vb, schedule=2)
        self.assertTrue(abs(res1 - res2) <= 1e-12)

    def test_ddot_array_openmp(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = numpy.dot(va, vb)
        res2 = ddot_array_openmp(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-12)

    def test_ddot_array_openmp_16(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = numpy.dot(va, vb)
        res2 = ddot_array_openmp_16(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-12)

    def test_get_omp_max_threads(self):
        res2 = get_omp_max_threads()
        self.assertTrue(res2 > 0)


if __name__ == '__main__':
    unittest.main()
