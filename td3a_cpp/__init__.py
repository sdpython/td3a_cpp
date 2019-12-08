# coding: utf-8
"""
Template to develop a python module using
:epkg:`cython` and :epkg:`openmp`.
"""

__version__ = "0.1"
__author__ = "Xavier Dupré"


def check(verbose=1):
    """
    Runs a couple of functions to check the module is working.

    :param verbose: 0 to hide the standout output
    :return: list of dictionaries, result of each test
    """
    import pprint
    import numpy
    from .tutorial import pydot, cblas_ddot
    from .tools import measure_time

    va = numpy.arange(0, 1000).astype(numpy.float64)
    vb = numpy.arange(0, 1000).astype(numpy.float64) - 5
    fcts = [
        ('pydot', pydot),
        ('numpy.dot', numpy.dot),
        ('ddot', cblas_ddot),
    ]

    rows = []
    for name, fct in fcts:
        ctx = {'va': va, 'vb': vb, 'fdot': fct}
        res = measure_time('fdot(va, vb)', ctx)
        res['name'] = name
        if verbose > 0:
            pprint.pprint(res)
        rows.append(res)
    return rows
