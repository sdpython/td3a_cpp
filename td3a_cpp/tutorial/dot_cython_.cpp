#include "dot_cython_.h"
#include <emmintrin.h>  // for double m128d
#include <intrin.h>     // m256d
#include <xmmintrin.h>  // for doubles m128

#define BYN 16

double vector_dot_product_pointer16(const double *p1, const double *p2)
{
    // Branching optimization must be done in a separate function.
    double sum = 0;
    
    sum += *(p1++) * *(p2++);
    sum += *(p1++) * *(p2++);
    sum += *(p1++) * *(p2++);
    sum += *(p1++) * *(p2++);
    sum += *(p1++) * *(p2++);
    sum += *(p1++) * *(p2++);
    sum += *(p1++) * *(p2++);
    sum += *(p1++) * *(p2++);

    sum += *(p1++) * *(p2++);
    sum += *(p1++) * *(p2++);
    sum += *(p1++) * *(p2++);
    sum += *(p1++) * *(p2++);
    sum += *(p1++) * *(p2++);
    sum += *(p1++) * *(p2++);
    sum += *(p1++) * *(p2++);
    sum += *(p1++) * *(p2++);

    return sum;
}

double vector_dot_product_pointer16(const double *p1, const double *p2, int sizei)
{
    size_t size = (size_t)sizei;
    double sum = 0;
    size_t i = 0;
    if (size >= BYN) {
        size_t size_ = size - BYN;
        for(; i < size_; i += BYN, p1 += BYN, p2 += BYN)
            sum += vector_dot_product_pointer16(p1, p2);
    }
    size -= i;
    for(; size > 0; ++p1, ++p2, --size)
        sum += *p1 * *p2;
    return sum;
}


double vector_dot_product_pointer16_sse(const double *p1, const double *p2)
{
    __m256d c1 = _mm256_load_pd(p1);
    __m256d c2 = _mm256_load_pd(p2);
    __m256d r1 = _mm256_mul_pd(c1, c2);
    
    p1 += 4;
    p2 += 4;
    
    c1 = _mm256_load_pd(p1);
    c2 = _mm256_load_pd(p2);
    r1 = _mm256_add_pd(r1, _mm256_mul_pd(c1, c2));
    
    p1 += 4;
    p2 += 4;
    
    c1 = _mm256_load_pd(p1);
    c2 = _mm256_load_pd(p2);
    r1 = _mm256_add_pd(r1, _mm256_mul_pd(c1, c2));
    
    p1 += 4;
    p2 += 4;
    
    c1 = _mm256_load_pd(p1);
    c2 = _mm256_load_pd(p2);
    r1 = _mm256_add_pd(r1, _mm256_mul_pd(c1, c2));

    double r[4];
    _mm256_store_pd(r, r1);

    return r[0] + r[1] + r[2] + r[3];
}


double vector_dot_product_pointer16_sse(const double *p1, const double *p2, int sizei)
{
    double sum = 0;
    size_t size = (size_t)sizei;
    size_t i = 0;
    if (size >= BYN) {
        size_t size_ = size - BYN;
        for(; i < size_; i += BYN, p1 += BYN, p2 += BYN)
            sum += vector_dot_product_pointer16_sse(p1, p2);
    }
    size -= i;
    for(; size > 0; ++p1, ++p2, --size)
        sum += *p1 * *p2;
    return sum;
}
