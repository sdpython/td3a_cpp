
===
API
===

.. contents::
    :local:

Tools
+++++

.. autofunction:: td3a_cpp.tools.measure_time

.. autofunction:: td3a_cpp.tools.measure_time_dim

Tutorial
++++++++

dot
^^^

.. autofunction:: td3a_cpp.tutorial.pydot

.. autofunction:: td3a_cpp.tutorial.cblas_ddot

.. autofunction:: td3a_cpp.tutorial.dot_cython.dot_product

**float32**

.. autofunction:: td3a_cpp.tutorial.dot_cython.sdot_cython_array

.. autofunction:: td3a_cpp.tutorial.dot_cython.sdot_cython_array_optim

.. autofunction:: td3a_cpp.tutorial.dot_cython.sdot_array

.. autofunction:: td3a_cpp.tutorial.dot_cython.sdot_array_16

.. autofunction:: td3a_cpp.tutorial.dot_cython.sdot_array_16_sse

**double = float64**

.. autofunction:: td3a_cpp.tutorial.dot_cython.ddot_cython_array

.. autofunction:: td3a_cpp.tutorial.dot_cython.ddot_cython_array_optim

.. autofunction:: td3a_cpp.tutorial.dot_cython.ddot_array

.. autofunction:: td3a_cpp.tutorial.dot_cython.ddot_array_16

.. autofunction:: td3a_cpp.tutorial.dot_cython.ddot_array_16_sse

**openmp**

.. autofunction:: td3a_cpp.tutorial.dot_cython_omp.get_omp_max_threads

.. autofunction:: td3a_cpp.tutorial.dot_cython_omp.ddot_cython_array_omp

.. autofunction:: td3a_cpp.tutorial.dot_cython_omp.ddot_array_openmp

.. autofunction:: td3a_cpp.tutorial.dot_cython_omp.ddot_array_openmp_16

filter
^^^^^^

.. autofunction:: td3a_cpp.tutorial.experiment_cython.pyfilter_dmax

.. autofunction:: td3a_cpp.tutorial.experiment_cython.filter_dmax_cython

.. autofunction:: td3a_cpp.tutorial.experiment_cython.filter_dmax_cython_optim

.. autofunction:: td3a_cpp.tutorial.experiment_cython.cyfilter_dmax

.. autofunction:: td3a_cpp.tutorial.experiment_cython.cfilter_dmax

.. autofunction:: td3a_cpp.tutorial.experiment_cython.cfilter_dmax2

.. autofunction:: td3a_cpp.tutorial.experiment_cython.cfilter_dmax4

.. autofunction:: td3a_cpp.tutorial.experiment_cython.cfilter_dmax16

matrix multiplication
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: td3a_cpp.tutorial.mul_cython_omp.dmul_cython_omp
