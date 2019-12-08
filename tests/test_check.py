"""
Unit tests for ``random_strategy``.
"""
import unittest
from td3a_cpp import check


class TestCheck(unittest.TestCase):

    def test_check(self):
        res = check(verbose=0)
        self.assertIsInstance(res, list)


if __name__ == '__main__':
    unittest.main()
