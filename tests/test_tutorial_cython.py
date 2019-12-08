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

    def test_all_dot(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = numpy.random.randn(100).astype(numpy.float64)
        fcts = [dot_product, dot_cython_array,
                dot_cython_array_optim, dot_array,
                dot_array_16, dot_array_16_sse]
        res = []
        for fct in fcts:
            r = fct(va, vb)
            res.append(r)
        for r in res[1:]:
            self.assertTrue(abs(res[0] - r) <= 1e-12)

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
