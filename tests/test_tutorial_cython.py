"""
Unit tests for ``random_strategy``.
"""
import unittest
import numpy
from td3a_cpp.tutorial import pydot, cblas_ddot, cblas_sdot
from td3a_cpp.tutorial.dot_cython import (
    dot_product, dot_cython_array,
    dot_cython_array_optim, dot_array,
    dot_array_16, dot_array_16_sse
)


class TestTutorialDot(unittest.TestCase):

    def test_dot_product(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = numpy.dot(va, vb)
        res2 = dot_product(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-12)

    def test_dot_cython_array(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = numpy.dot(va, vb)
        res2 = dot_cython_array(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-12)

    def test_dot_cython_array_optim(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = numpy.dot(va, vb)
        res2 = dot_cython_array_optim(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-12)

    def test_dot_array(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = numpy.dot(va, vb)
        res2 = dot_array(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-12)

    def test_dot_array_16(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = numpy.dot(va, vb)
        res2 = dot_array_16(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-12)

    def test_dot_array_16_sse(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = numpy.dot(va, vb)
        res2 = dot_array_16_sse(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-12)

    def test_ddot(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = cblas_ddot(va, vb)
        res2 = numpy.dot(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-4)

    def test_sdot(self):
        va = numpy.random.randn(100).astype(numpy.float32)
        vb = numpy.random.randn(100).astype(numpy.float32)
        res1 = cblas_sdot(va, vb)
        res2 = numpy.dot(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-13)

    def test_pydot(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = pydot(va, vb)
        res2 = numpy.dot(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-13)


if __name__ == '__main__':
    unittest.main()
