#pragma once


#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#define undef__CRT_SECURE_NO_WARNINGS 1
#endif

double vector_ddot_product_pointer16(const double *p1, const double *p2, int size);
double vector_ddot_product_pointer16_sse(const double *p1, const double *p2, int size);

float vector_sdot_product_pointer16(const float *p1, const float *p2, int size);
float vector_sdot_product_pointer16_sse(const float *p1, const float *p2, int size);

#if defined(undef_CRT_SECURE_NO_WARNINGS)
#undef _CRT_SECURE_NO_WARNINGS
#endif
