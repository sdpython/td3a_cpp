
.. image:: https://circleci.com/gh/sdpython/td3a_cpp/tree/master.svg?style=svg
    :target: https://circleci.com/gh/sdpython/td3a_cpp/tree/master

.. image:: https://travis-ci.org/sdpython/td3a_cpp.svg?branch=master
    :target: https://travis-ci.org/sdpython/td3a_cpp
    :alt: Build status

.. image:: https://ci.appveyor.com/api/projects/status/wvo6ovlaxi8ypua4?svg=true
    :target: https://ci.appveyor.com/project/sdpython/td3a-cpp
    :alt: Build Status Windows

.. image:: https://dev.azure.com/xavierdupre3/td3a_cpp/_apis/build/status/sdpython.td3a_cpp
    :target: https://dev.azure.com/xavierdupre3/td3a_cpp/

.. image:: https://badge.fury.io/py/td3a_cpp.svg
    :target: http://badge.fury.io/py/td3a_cpp

.. image:: http://img.shields.io/github/issues/sdpython/td3a_cpp.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/td3a_cpp/issues

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: http://opensource.org/licenses/MIT

.. image:: https://img.shields.io/github/repo-size/sdpython/td3a_cpp
    :target: https://github.com/sdpython/td3a_cpp/
    :alt: size

td3a_cpp: template to use cython and C++ with python
====================================================

.. image:: https://raw.githubusercontent.com/sdpython/td3a_cpp/master/doc/_static/logo.png
    :width: 50

`documentation <http://www.xavierdupre.fr/app/td3a_cpp/helpsphinx/index.html>`_

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

The function *check* or the command line ``python -m td3a_cpp check``
checks the module is properly installed and returns processing
time for a couple of functions or simply:

::

    import td3a_cpp
    td3a_cpp.check()
