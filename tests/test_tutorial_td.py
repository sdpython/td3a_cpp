"""
"""
import unittest
import timeit
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

    def test_timeit(self):
        va = numpy.random.randn(300, 400).astype(numpy.float64)
        vb = numpy.random.randn(400, 500).astype(numpy.float64)
        ctx = {'va': va, 'vb': vb, 'c_multiply_matrix': c_multiply_matrix,
               'multiply_matrix': multiply_matrix}
        res1 = timeit.timeit('va @ vb', number=10, globals=ctx)
        res2 = timeit.timeit(
            'c_multiply_matrix(va, vb)', number=10, globals=ctx)
        res3 = timeit.timeit(
            'multiply_matrix(va, vb)', number=10, globals=ctx)
        self.assertLess(res1, res2)  # numpy is much faster.
        ratio1 = res2 / res1
        # ratio1 = number of times numpy is faster
        self.assertGreater(ratio1, 1)
        ratio2 = res3 / res1
        self.assertGreater(ratio2, 1)


if __name__ == '__main__':
    unittest.main()
