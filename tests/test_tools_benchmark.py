"""
Unit tests for ``random_strategy``.
"""
import unittest
import numpy
from td3a_cpp.tutorial import pydot
from td3a_cpp.tools import measure_time, measure_time_dim


class TestToolsBenchmark(unittest.TestCase):

    def test_measure_time(self):
        va = numpy.arange(10)
        vb = numpy.arange(10) - 5
        m = measure_time('pydot(va, vb)', dict(va=va, vb=vb, pydot=pydot))
        self.assertIn('average', m)
        self.assertIn('deviation', m)
        
    def test_measure_time_dim(self):
        ctx = [dict(x_name=i, va=numpy.arange(i), vb=numpy.arange(10) - 5, pydot=pydot)
               for i in range(10, 10000, 100)]
        
        ms = list(measure_time_dim('pydot(va, vb)', ctx))
        self.assertIsInstance(ms, list)
        self.assertEqual(len(ms), 100)


if __name__ == '__main__':
    unittest.main()
