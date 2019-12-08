"""

.. _l-example-dot-profile:

Profile a function
==================

:epkg:`pyinstrument` is a profiler for :epkg:`python` functions.
It tells how long the program stays in every function.
Because numerical functions are usually very short,
it is usually helpful to call them multiple times
before the cause becomes significant.


.. content::
    :local:
"""

from sys import executable
import sys
import numpy
from pyinstrument import Profiler
from pyquickhelper.loghelper import run_cmd
from td3a_cpp.tutorial import pydot, cblas_ddot


va = numpy.arange(100000).astype(numpy.float64)
vb = numpy.arange(100000).astype(numpy.float64) - 5


def f1_python(va, vb, n=10):
    # pydot is really too slow, let's run 10 times only
    for i in range(n):
        pydot(va, vb)


def f2_numpy(va, vb, n=100000):
    for i in range(n):
        numpy.dot(va, vb)


def f3_blas(va, vb, n=100000):
    for i in range(n):
        cblas_ddot(va, vb)


if '--pyspy' in sys.argv:
    # When called with option --pyspy
    f1_python(va, vb)
    f2_numpy(va, vb)
    f3_blas(va, vb)
    profiler = None
else:
    profiler = Profiler()
    profiler.start()

    f1_python(va, vb)
    f2_numpy(va, vb)
    f3_blas(va, vb)

    profiler.stop()

    print(profiler.output_text(unicode=False, color=False))


#######################################
# numpy is much faster and does not always appear
# if pydot is run the same number of times.
# The program is spied on a given number of times
# per seconds, each time the system records
# which function the program is executing.
# At the end, the profiler is able to approximatvely tell
# how long the program stayed in every function.
# If a function does not appear, it means it was never executed
# or too fast to be caught.
# An HTML report can be generated.

if profiler is not None:
    with open("dot_pyinstrument.html", "w", encoding="utf-8") as f:
        f.write(profiler.output_html())

########################
# It looks like this:
#
# .. raw:: html
#       :file: _dot_pyinstrument.html
#
#
# :epkg:`pyinstrument` does not measure native function (C++)
# very well. For this module :epkg:`py-spy` is more efficient
# but it only works in command line as the package itself
# is written in :epkg:`RUST`
# (see `Taking ML to production with Rust: a 25x speedup
# <https://www.lpalmieri.com/posts/
# 2019-12-01-taking-ml-to-production-with-rust-a-25x-speedup/>`_).

if profiler is not None:
    cmd = ("py-spy record --native --function --rate=10 "
           "-o dotpyspy.svg -- {0} plot_profile.py --pyspy").format(executable)
    run_cmd(cmd, wait=True, fLOG=print)

# It looks like this:
#
# .. raw:: html
#       :file: _dotpyspy.html
#
# We see that :func:`cblas_ddot` and `numpy.dot` uses
# the same C function but the wrapping is not the same
# and numpy is more efficient.
