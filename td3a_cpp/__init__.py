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
    from .tutorial.dot_cython import (
        dot_product, dot_cython_array,
        dot_cython_array_optim, dot_array,
        dot_array_16, dot_array_16_sse
    )
    from .tools import measure_time

    va = numpy.arange(0, 10000).astype(numpy.float64)
    vb = numpy.arange(0, 10000).astype(numpy.float64) - 5
    fcts = [
        ('pydot', pydot, 1),
        ('numpy.dot', numpy.dot),
        ('ddot', cblas_ddot),
        ('dot_product', dot_product),
        ('dot_cython_array', dot_cython_array),
        ('dot_cython_array_optim', dot_cython_array_optim),
        ('dot_array', dot_array),
        ('dot_array_16', dot_array_16),
        ('dot_array_16_sse', dot_array_16_sse),
    ]

    rows = []
    for tu in fcts:
        name, fct = tu[:2]
        ctx = {'va': va, 'vb': vb, 'fdot': fct}
        if len(tu) == 3:
            res = measure_time('fdot(va, vb)', ctx, repeat=tu[2])
        else:
            res = measure_time('fdot(va, vb)', ctx)
        res['name'] = name
        if verbose > 0:
            pprint.pprint(res)
        rows.append(res)
    return rows
