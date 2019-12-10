#pragma once


#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#define undef__CRT_SECURE_NO_WARNINGS 1
#endif

int get_omp_max_threads_cpp();                          

double vector_ddot_openmp(const double *p1, const double *p2,
                          int size, int nthreads);

double vector_ddot_openmp_16(const double *p1, const double *p2,
                             int size, int nthreads);

#if defined(undef_CRT_SECURE_NO_WARNINGS)
#undef _CRT_SECURE_NO_WARNINGS
#endif
