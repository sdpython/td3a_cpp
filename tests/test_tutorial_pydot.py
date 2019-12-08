"""
Unit tests for ``random_strategy``.
"""
import unittest
import numpy
from td3a_cpp.tutorial import pydot


class TestTutorialPyDot(unittest.TestCase):

    def test_pydot(self):
        va = numpy.arange(10)
        vb = numpy.arange(10) - 5
        res1 = pydot(va, vb)
        res2 = numpy.dot(va, vb)
        self.assertEqual(res1, res2)


if __name__ == '__main__':
    unittest.main()
