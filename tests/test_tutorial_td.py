"""
"""
import unittest
import numpy
from numpy.testing import assert_almost_equal
from td3a_cpp.tutorial.td_mul_cython import (
    multiply_matrix, c_multiply_matrix)


class TestTutorialTD(unittest.TestCase):

    def test_matrix_multiply_matrix(self):
        va = numpy.random.randn(3, 4).astype(numpy.float64)
        vb = numpy.random.randn(4, 5).astype(numpy.float64)
        res1 = va @ vb
        res2 = multiply_matrix(va, vb)
        assert_almost_equal(res1, res2)

    def test_matrix_cmultiply_matrix(self):
        va = numpy.random.randn(3, 4).astype(numpy.float64)
        vb = numpy.random.randn(4, 5).astype(numpy.float64)
        res1 = va @ vb
        res2 = c_multiply_matrix(va, vb)
        assert_almost_equal(res1, res2)


if __name__ == '__main__':
    unittest.main()
