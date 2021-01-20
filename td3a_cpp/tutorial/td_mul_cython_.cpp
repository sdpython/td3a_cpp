#include "mul_cython_omp_.h"
#include <stdio.h>
#include <emmintrin.h>  // for double m128d


double vector_ddot_product_pointer16_sse(const double *p1, const double *p2, int size) {
    if (size == 0)
        return 0;
    if (size == 1)
        return *p1 * *p2;
    double sum = 0;
    const double* end = p1 + size;
    if (size % 2 == 1) {
        sum += *p1 * *p2;
        ++p1;
        ++p2;
        --size;
    }
    __m128d c1 = _mm_loadu_pd(p1);
    __m128d c2 = _mm_loadu_pd(p2);
    __m128d r1 = _mm_mul_pd(c1, c2);
    p1 += 2;
    p2 += 2;
    for( ; p1 != end; p1 += 2, p2 += 2) {
        c1 = _mm_loadu_pd(p1);
        c2 = _mm_loadu_pd(p2);
        r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));
    }
    double r[2];  // r is not necessary aligned.
    _mm_storeu_pd(r, r1);
    sum += r[0] + r[1];
    return sum;
}
