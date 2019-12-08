#pragma once


#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#define undef__CRT_SECURE_NO_WARNINGS 1
#endif

double vector_dot_product_pointer16(const double *p1, const double *p2, int size);
double vector_dot_product_pointer16_sse(const double *p1, const double *p2, int size);


#if defined(undef_CRT_SECURE_NO_WARNINGS)
#undef _CRT_SECURE_NO_WARNINGS
#endif
