"""
"""
import unittest
import timeit
import numpy
from numpy.testing import assert_almost_equal
from td3a_cpp.tutorial.td_mul_cython import (
    multiply_matrix, c_multiply_matrix)
from td3a_cpp.tutorial.mul_cython_omp import dmul_cython_omp


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

    def test_timeit(self):
        va = numpy.random.randn(300, 400).astype(numpy.float64)
        vb = numpy.random.randn(400, 500).astype(numpy.float64)
        res1 = va @ vb
        res2 = c_multiply_matrix(va, vb)
        ctx = {'va': va, 'vb': vb, 'c_multiply_matrix': c_multiply_matrix}
        res1 = timeit.timeit('va @ vb', number=10, globals=ctx)
        res2 = timeit.timeit('c_multiply_matrix(va, vb)', number=10, globals=ctx)
        self.assertLess(res1, res2)  # numpy is much faster.
        ratio = res2 / res1
        self.assertGreater(ratio, 1)  # ratio = number of times numpy is faster
        # print(ratio)


if __name__ == '__main__':
    unittest.main()
