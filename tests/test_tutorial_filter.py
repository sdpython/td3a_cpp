"""
Unit tests for ``random_strategy``.
"""
import unittest
import numpy
from numpy.testing import assert_equal
from td3a_cpp.tutorial.experiment_cython import (
    pyfilter_dmax, filter_dmax_cython,
    filter_dmax_cython_optim,
    cyfilter_dmax,
    cfilter_dmax, cfilter_dmax2, cfilter_dmax16,
    cfilter_dmax4
)


class TestTutorialFilter(unittest.TestCase):

    def test_pyfilter_dmax(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = va.copy()
        pyfilter_dmax(va, 0)
        vb[vb > 0] = 0
        assert_equal(va, vb)

    def test_filter_dmax_cython(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = va.copy()
        filter_dmax_cython(va, 0)
        vb[vb > 0] = 0
        assert_equal(va, vb)

    def test_filter_dmax_cython_optim(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = va.copy()
        filter_dmax_cython_optim(va, 0)
        vb[vb > 0] = 0
        assert_equal(va, vb)

    def test_filter_cyfilter_dmax(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = va.copy()
        cyfilter_dmax(va, 0)
        vb[vb > 0] = 0
        assert_equal(va, vb)

    def test_filter_cfilter_dmax(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = va.copy()
        cfilter_dmax(va, 0)
        vb[vb > 0] = 0
        assert_equal(va, vb)

    def test_filter_cfilter_dmax2(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = va.copy()
        cfilter_dmax2(va, 0)
        vb[vb > 0] = 0
        assert_equal(va, vb)

    def test_filter_cfilter_dmax16(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = va.copy()
        cfilter_dmax16(va, 0)
        vb[vb > 0] = 0
        assert_equal(va, vb)

    def test_filter_cfilter_dmax4(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = va.copy()
        cfilter_dmax4(va, 0)
        vb[vb > 0] = 0
        assert_equal(va, vb)

    def test_cfilter_dmax(self):
        va = numpy.random.randn(100).astype(numpy.float64)
        vb = va.copy()
        cfilter_dmax(va, 0)
        vb[vb > 0] = 0
        assert_equal(va, vb)


if __name__ == '__main__':
    unittest.main()
