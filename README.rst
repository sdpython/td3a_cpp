
.. image:: https://circleci.com/gh/sdpython/td3a_cpp/tree/master.svg?style=svg
    :target: https://circleci.com/gh/sdpython/td3a_cpp/tree/master

td3a_cpp
========

.. image:: https://raw.githubusercontent.com/sdpython/td3a_cpp/master/doc/_static/logo.png
    :width: 50

Simple template to implement an algorithm with *cython* and *openmp*.
It implements simple examples to demonstrate the speed up
obtained by using *cython*. The module must be compiled
to be used inplace:

::

    python setup.py build_ext --inplace

Generate the setup in subfolder ``dist``:

::

    python setup.py sdist

Generate the documentation in folder ``dist/html``:

::

    python -m sphinx -T -b html doc dist/html

Run the unit tests:

::

    python -m unittest discover tests

Or:

::

    python -m pytest
    
To check style:

::

    python -m flake8 td3a_cpp tests examples
