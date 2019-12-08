"""
@file
@brief Measures speed.
"""
import sys
from timeit import Timer
import numpy


def measure_time(stmt, context, repeat=10, number=50, div_by_number=True):
    """
    Measures a statement and returns the results as a dictionary.

    @param      stmt            string
    @param      context         variable to know in a dictionary
    @param      repeat          average over *repeat* experiment
    @param      number          number of executions in one row
    @param      div_by_number   divide by the number of executions
    @return                     dictionary

    .. runpython::
        :showcode:

        from math import cos
        import pprint
        from td3a_cpp.tools import measure_time

        res = measure_time("cos(x)", context=dict(cos=cos, x=5.))
        pprint.pprint(res)

    See `Timer.repeat <https://docs.python.org/3/library/timeit.html?timeit.Timer.repeat>`_
    for a better understanding of parameter *repeat* and *number*.
    The function returns a duration corresponding to
    *number* times the execution of the main statement.
    """
    tim = Timer(stmt, globals=context)
    res = numpy.array(tim.repeat(repeat=repeat, number=number))
    if div_by_number:
        res /= number
    mean = numpy.mean(res)
    dev = numpy.mean(res ** 2)
    dev = (dev - mean**2) ** 0.5
    mes = dict(average=mean, deviation=dev, min_exec=numpy.min(res),
               max_exec=numpy.max(res), repeat=repeat, number=number)
    if 'values' in context:
        if hasattr(context['values'], 'shape'):
            mes['size'] = context['values'].shape[0]
        else:
            mes['size'] = len(context['values'])
    else:
        mes['context_size'] = sys.getsizeof(context)
    return mes


def measure_time_dim(stmt, contexts, repeat=10, number=50, div_by_number=True):
    """
    Measures a statement multiple time with function :func:`measure_time_dim`.

    @param      stmt            string
    @param      contexts        variable to know in a dictionary,
                                every context must include field 'x_name',
                                which is copied in the result
    @param      repeat          average over *repeat* experiment
    @param      number          number of executions in one row
    @param      div_by_number   divide by the number of executions
    @return                     yield dictionary

    .. runpython::
        :showcode:

        import pprint
        import numpy        
        from td3a_cpp.tools import measure_time

        res = list(measure_time_dim(
            "cos(x)",
            context=[dict(cos=numpy.cos, x=numpy.arange(10), x_name=10),
                     dict(cos=numpy.cos, x=numpy.arange(100), x_name=100)))
        pprint.pprint(res)

    See `Timer.repeat <https://docs.python.org/3/library/timeit.html?timeit.Timer.repeat>`_
    for a better understanding of parameter *repeat* and *number*.
    The function returns a duration corresponding to
    *number* times the execution of the main statement.
    """
    for context in contexts:
        if 'x_name' not in context:
            raise ValueError("The context must contain field 'x_name', "
                             "usually the X coordinate to draw the benchmark.")
        res = measure_time(stmt, context, repeat=repeat,
                           number=number, div_by_number=div_by_number)
        res['x_name'] = context['x_name']
        yield res
