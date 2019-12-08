"""
Unit tests for ``random_strategy``.
"""
import unittest
import numpy
from td3a_cpp.tutorial.dot_cython import (
    dot_product, dot_cython_array,
    dot_cython_array_optim, dot_array,
    dot_array_16, dot_array_16_sse
)


class TestTutorialBlas(unittest.TestCase):

    def test_all_dot(self):
        va = numpy.arange(10).astype(numpy.float64)
        vb = numpy.arange(10).astype(numpy.float64) - 5
        fcts = [dot_product, dot_cython_array,
                dot_cython_array_optim, dot_array,
                dot_array_16, dot_array_16_sse]
        res = []
        for fct in fcts:
            r = fct(va, vb)
            res.append(r)
        for r in res[1:]:
            self.assertEqual(res[0], r)


if __name__ == '__main__':
    unittest.main()
