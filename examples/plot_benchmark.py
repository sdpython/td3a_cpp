"""

.. _l-example-dot:

Compares dot implementation
===========================

:epkg:`numpy` has a very fast implementation of
the dot product. It is difficult to be better and very easy
to be slower. This example looks into a couple of slower
implementations.

.. content::
    :local:
"""
import numpy
import matplotlib.pyplot as plt
from td3a_cpp.tutorial import pydot

##############################
# python dot: pydot
# +++++++++++++++++
#
# The first function :func:`pydot
# <td3a_cpp.tutorial.pydot>` uses
# python to implement the dot product.



###########################
# One seems better but 50 tries
# does not seem to be enough to be fully sure.
plt.show()
