"""
Unit tests for ``random_strategy``.
"""
import unittest
import numpy
from td3a_cpp.tutorial import cblas_ddot, cblas_sdot


class TestTutorialBlas(unittest.TestCase):

    def test_ddot(self):
        va = numpy.arange(10).astype(numpy.float64)
        vb = numpy.arange(10).astype(numpy.float64) - 5
        res1 = cblas_ddot(va, vb)
        res2 = numpy.dot(va, vb)
        self.assertEqual(res1, res2)

    def test_sdot(self):
        va = numpy.arange(10).astype(numpy.float32)
        vb = numpy.arange(10).astype(numpy.float32) - 5
        res1 = cblas_sdot(va, vb)
        res2 = numpy.dot(va, vb)
        self.assertEqual(res1, res2)


if __name__ == '__main__':
    unittest.main()
