#pragma once


#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#define undef__CRT_SECURE_NO_WARNINGS 1
#endif

void filter_dmax(double *p1, int size, double mx);
void filter_dmax2(double *p1, int size, double mx);
void filter_dmax16(double *p1, int size, double mx);
void filter_dmax4(double *p1, int size, double mx);

#if defined(undef_CRT_SECURE_NO_WARNINGS)
#undef _CRT_SECURE_NO_WARNINGS
#endif
