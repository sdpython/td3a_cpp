"""
Implements of the :epkg:`dot` function in python.
"""


def pydot(va, vb):
    """
    Implements the dot product between two vectors.

    :param va: first vector
    :param vb: second vector
    :return: dot product
    """
    return sum((a * b) for a, b in zip(va, vb))
