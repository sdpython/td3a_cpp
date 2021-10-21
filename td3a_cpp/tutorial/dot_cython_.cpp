#include "dot_cython_.h"

//////////////////////////
// branching
//////////////////////////

#define BYN 16

double vector_ddot_product_pointer16(const double *p1, const double *p2) {
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


double vector_ddot_product_pointer16(const double *p1, const double *p2, int size) {
    double sum = 0;
    int i = 0;
    if (size >= BYN) {
        int size_ = size - BYN;
        for(; i < size_; i += BYN, p1 += BYN, p2 += BYN)
            sum += vector_ddot_product_pointer16(p1, p2);
    }
    size -= i;
    for(; size > 0; ++p1, ++p2, --size)
        sum += *p1 * *p2;
    return sum;
}


float vector_sdot_product_pointer16(const float *p1, const float *p2) {
    // Branching optimization must be done in a separate function.
    float sum = 0;
    
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


float vector_sdot_product_pointer16(const float *p1, const float *p2, int size) {
    float sum = 0;
    int i = 0;
    if (size >= BYN) {
        int size_ = size - BYN;
        for(; i < size_; i += BYN, p1 += BYN, p2 += BYN)
            sum += vector_sdot_product_pointer16(p1, p2);
    }
    size -= i;
    for(; size > 0; ++p1, ++p2, --size)
        sum += *p1 * *p2;
    return sum;
}


//////////////////////////
// branching + AVX
//////////////////////////


#ifdef _WIN32
// Not available on all machines
// It should depends on AVX constant not WIN32.
#include <immintrin.h>  // double double m256d


double vector_ddot_product_pointer16_sse(const double *p1, const double *p2) {
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


#else

#include <emmintrin.h>  // for double m128d

double vector_ddot_product_pointer16_sse(const double *p1, const double *p2) {
    __m128d c1 = _mm_load_pd(p1);
    __m128d c2 = _mm_load_pd(p2);
    __m128d r1 = _mm_mul_pd(c1, c2);
    
    p1 += 2;
    p2 += 2;
    
    c1 = _mm_load_pd(p1);
    c2 = _mm_load_pd(p2);
    r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));
    
    p1 += 2;
    p2 += 2;
    
    c1 = _mm_load_pd(p1);
    c2 = _mm_load_pd(p2);
    r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));
    
    p1 += 2;
    p2 += 2;
    
    c1 = _mm_load_pd(p1);
    c2 = _mm_load_pd(p2);
    r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));
    
    // 8
    
    p1 += 2;
    p2 += 2;
    
    c1 = _mm_load_pd(p1);
    c2 = _mm_load_pd(p2);
    r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));
    
    p1 += 2;
    p2 += 2;
    
    c1 = _mm_load_pd(p1);
    c2 = _mm_load_pd(p2);
    r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));
    
    p1 += 2;
    p2 += 2;
    
    c1 = _mm_load_pd(p1);
    c2 = _mm_load_pd(p2);
    r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));
    
    p1 += 2;
    p2 += 2;
    
    c1 = _mm_load_pd(p1);
    c2 = _mm_load_pd(p2);
    r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));

    double r[2];
    _mm_store_pd(r, r1);

    return r[0] + r[1];
}

#endif


double vector_ddot_product_pointer16_sse(const double *p1, const double *p2, int size) {
    double sum = 0;
    int i = 0;
    if (size >= BYN) {
        int size_ = size - BYN;
        for(; i < size_; i += BYN, p1 += BYN, p2 += BYN)
            sum += vector_ddot_product_pointer16_sse(p1, p2);
    }
    size -= i;
    for(; size > 0; ++p1, ++p2, --size)
        sum += *p1 * *p2;
    return sum;
}

#include <xmmintrin.h>  // for float m128

float vector_sdot_product_pointer16_sse(const float *p1, const float *p2) {
    __m128 c1 = _mm_load_ps(p1);
    __m128 c2 = _mm_load_ps(p2);
    __m128 r1 = _mm_mul_ps(c1, c2);
    
    p1 += 4;
    p2 += 4;
    
    c1 = _mm_load_ps(p1);
    c2 = _mm_load_ps(p2);
    r1 = _mm_add_ps(r1, _mm_mul_ps(c1, c2));
    
    p1 += 4;
    p2 += 4;
    
    c1 = _mm_load_ps(p1);
    c2 = _mm_load_ps(p2);
    r1 = _mm_add_ps(r1, _mm_mul_ps(c1, c2));
    
    p1 += 4;
    p2 += 4;
    
    c1 = _mm_load_ps(p1);
    c2 = _mm_load_ps(p2);
    r1 = _mm_add_ps(r1, _mm_mul_ps(c1, c2));

    float r[4];
    _mm_store_ps(r, r1);

    return r[0] + r[1] + r[2] + r[3];
}


float vector_sdot_product_pointer16_sse(const float *p1, const float *p2, int size) {
    float sum = 0;
    int i = 0;
    if (size >= BYN) {
        int size_ = size - BYN;
        for(; i < size_; i += BYN, p1 += BYN, p2 += BYN)
            sum += vector_sdot_product_pointer16_sse(p1, p2);
    }
    size -= i;
    for(; size > 0; ++p1, ++p2, --size)
        sum += *p1 * *p2;
    return sum;
}

