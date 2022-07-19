"""
Unit tests for ``random_strategy``.
"""
import unittest
import numpy
from td3a_cpp.tutorial import pydot, cblas_ddot, cblas_sdot
from td3a_cpp.tutorial.dot_cython import (
    dot_product, ddot_cython_array,
    ddot_cython_array_optim, ddot_array,
    ddot_array_16, ddot_array_16_sse
)
from td3a_cpp.tutorial.dot_cython import (
    sdot_cython_array,
    sdot_cython_array_optim, sdot_array,
    sdot_array_16, sdot_array_16_sse
)


class TestTutorialDot(unittest.TestCase):

    def test_dot_product(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = numpy.dot(va, vb)
        res2 = dot_product(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-12)

    def test_ddot_cython_array(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = numpy.dot(va, vb)
        res2 = ddot_cython_array(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-12)

    def test_ddot_cython_array_optim(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = numpy.dot(va, vb)
        res2 = ddot_cython_array_optim(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-12)

    def test_ddot_array(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = numpy.dot(va, vb)
        res2 = ddot_array(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-12)

    def test_ddot_array_16(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = numpy.dot(va, vb)
        res2 = ddot_array_16(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-12)

    def test_ddot_array_16_sse(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = numpy.dot(va, vb)
        res2 = ddot_array_16_sse(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-12)

    def test_sdot_cython_array(self):
        va = numpy.random.randn(100).astype(numpy.float32)
        vb = numpy.random.randn(100).astype(numpy.float32)
        res1 = numpy.dot(va, vb)
        res2 = sdot_cython_array(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-5)

    def test_sdot_cython_array_optim(self):
        va = numpy.random.randn(100).astype(numpy.float32)
        vb = numpy.random.randn(100).astype(numpy.float32)
        res1 = numpy.dot(va, vb)
        res2 = sdot_cython_array_optim(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-5)

    def test_sdot_array(self):
        va = numpy.random.randn(100).astype(numpy.float32)
        vb = numpy.random.randn(100).astype(numpy.float32)
        res1 = numpy.dot(va, vb)
        res2 = sdot_array(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-5)

    def test_sdot_array_16(self):
        va = numpy.random.randn(100).astype(numpy.float32)
        vb = numpy.random.randn(100).astype(numpy.float32)
        res1 = numpy.dot(va, vb)
        res2 = sdot_array_16(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-5)

    def test_sdot_array_16_sse(self):
        va = numpy.random.randn(100).astype(numpy.float32)
        vb = numpy.random.randn(100).astype(numpy.float32)
        res1 = numpy.dot(va, vb)
        res2 = sdot_array_16_sse(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-5)

    def test_ddot_blas(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = cblas_ddot(va, vb)
        res2 = numpy.dot(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-4)

    def test_sdot_blas(self):
        va = numpy.random.randn(100).astype(numpy.float32)
        vb = numpy.random.randn(100).astype(numpy.float32)
        res1 = cblas_sdot(va, vb)
        res2 = numpy.dot(va, vb)
        if abs(res1 - res2) > 1e-5:
            raise AssertionError(f"{res1!r} != {res2!r}")

    def test_pydot(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        res1 = pydot(va, vb)
        res2 = numpy.dot(va, vb)
        self.assertTrue(abs(res1 - res2) <= 1e-13)


if __name__ == '__main__':
    unittest.main()
