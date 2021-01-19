"""
Unit tests for ``random_strategy``.
"""
import unittest
import numpy
from numpy.testing import assert_almost_equal
from td3a_cpp.tutorial.mul_cython_omp import dmul_cython_omp


class TestTutorialMul(unittest.TestCase):

    def test_matrix_mul(self):
        va = numpy.random.randn(3, 4).astype(numpy.float64)
        vb = numpy.random.randn(4, 5).astype(numpy.float64)
        res1 = va @ vb
        res2 = dmul_cython_omp(va, vb)
        assert_almost_equal(res1, res2)

    def test_matrix_mul_fail(self):
        va = numpy.random.randn(3, 4).astype(numpy.float64)
        vb = numpy.random.randn(4, 5).astype(numpy.float64)
        with self.assertRaises(RuntimeError):
            dmul_cython_omp(va, vb, algo=4)

    def test_matrix_mul_algo(self):
        va = numpy.random.randn(3, 4).astype(numpy.float64)
        vb = numpy.random.randn(4, 5).astype(numpy.float64)
        res1 = va @ vb
        for algo in range(0, 3):
            with self.subTest(algo=algo):
                res2 = dmul_cython_omp(va, vb, algo=algo)
                assert_almost_equal(res1, res2)

    def test_matrix_mul_algo_para(self):
        va = numpy.random.randn(3, 4).astype(numpy.float64)
        vb = numpy.random.randn(4, 5).astype(numpy.float64)
        res1 = va @ vb
        for algo in range(0, 2):
            with self.subTest(algo=algo):
                res2 = dmul_cython_omp(va, vb, algo=algo, parallel=1)
                assert_almost_equal(res1, res2)

    def test_matrix_mul_algo_t(self):
        va = numpy.random.randn(3, 4).astype(numpy.float64)
        vb = numpy.random.randn(5, 4).astype(numpy.float64)
        res1 = va @ vb.T
        for algo in range(0, 3):
            with self.subTest(algo=algo):
                res2 = dmul_cython_omp(va, vb, algo=algo,
                                       b_trans=1)
                assert_almost_equal(res1, res2)

    def test_matrix_mul_algo_t_big(self):
        va = numpy.random.randn(300, 400).astype(numpy.float64)
        vb = numpy.random.randn(500, 400).astype(numpy.float64)
        res1 = va @ vb.T
        for algo in range(0, 3):
            with self.subTest(algo=algo):
                res2 = dmul_cython_omp(va, vb, algo=algo,
                                       b_trans=1)
                assert_almost_equal(res1, res2)

    def test_matrix_mul_algo_t_big_odd(self):
        va = numpy.random.randn(30, 41).astype(numpy.float64)
        vb = numpy.random.randn(50, 41).astype(numpy.float64)
        res1 = va @ vb.T
        for algo in range(0, 3):
            with self.subTest(algo=algo):
                res2 = dmul_cython_omp(va, vb, algo=algo,
                                       b_trans=1)
                assert_almost_equal(res1, res2)

    def test_matrix_mul_algo_para_t(self):
        va = numpy.random.randn(3, 4).astype(numpy.float64)
        vb = numpy.random.randn(5, 4).astype(numpy.float64)
        res1 = va @ vb.T
        for algo in range(0, 2):
            with self.subTest(algo=algo):
                res2 = dmul_cython_omp(va, vb, algo=algo, parallel=1,
                                       b_trans=1)
                assert_almost_equal(res1, res2)

    def test_matrix_mul_algo_para_t_big(self):
        va = numpy.random.randn(300, 400).astype(numpy.float64)
        vb = numpy.random.randn(500, 400).astype(numpy.float64)
        res1 = va @ vb.T
        for algo in range(0, 2):
            with self.subTest(algo=algo):
                res2 = dmul_cython_omp(va, vb, algo=algo, parallel=1,
                                       b_trans=1)
                assert_almost_equal(res1, res2)


if __name__ == '__main__':
    unittest.main()
