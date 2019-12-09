#include "experiment_cython_.h"

void filter_dmax(double *p1, int size, double mx)
{
    double * end = p1 + size;
    for(; p1 != end; ++p1)
        if (*p1 > mx)
            *p1 = mx;
}


void filter_dmax2(double *p1, int size, double mx)
{
    double * end = p1 + size;
    for(; p1 != end; ++p1)
        *p1 = *p1 > mx ? mx : *p1;
}


void _filter_dmax16(double *&p1, double mx)
{
    *p1 = *p1 > mx ? mx : *p1; ++p1;
    *p1 = *p1 > mx ? mx : *p1; ++p1;
    *p1 = *p1 > mx ? mx : *p1; ++p1;
    *p1 = *p1 > mx ? mx : *p1; ++p1;

    *p1 = *p1 > mx ? mx : *p1; ++p1;
    *p1 = *p1 > mx ? mx : *p1; ++p1;
    *p1 = *p1 > mx ? mx : *p1; ++p1;
    *p1 = *p1 > mx ? mx : *p1; ++p1;

    *p1 = *p1 > mx ? mx : *p1; ++p1;
    *p1 = *p1 > mx ? mx : *p1; ++p1;
    *p1 = *p1 > mx ? mx : *p1; ++p1;
    *p1 = *p1 > mx ? mx : *p1; ++p1;

    *p1 = *p1 > mx ? mx : *p1; ++p1;
    *p1 = *p1 > mx ? mx : *p1; ++p1;
    *p1 = *p1 > mx ? mx : *p1; ++p1;
    *p1 = *p1 > mx ? mx : *p1; ++p1;
}


void filter_dmax16(double *p1, int size, double mx)
{
    int size16 = size % 16;
    double * end = p1 + size - size16;
    for(; p1 != end; )
        _filter_dmax16(p1, mx);
    end += size16;
    for(; p1 != end; ++p1)
        *p1 = *p1 > mx ? mx : *p1;
}


void _filter_dmax4(double *p1, double mx)
{
    *p1 = *p1 > mx ? mx : *p1; ++p1;
    *p1 = *p1 > mx ? mx : *p1; ++p1;
    *p1 = *p1 > mx ? mx : *p1; ++p1;
    *p1 = *p1 > mx ? mx : *p1; ++p1;
}


void filter_dmax4(double *p1, int size, double mx)
{
    int size4 = size % 4;
    double * end = p1 + size - size4;
    for(; p1 != end; p1 += 4)
        _filter_dmax4(p1, mx);
    end += size4;
    for(; p1 != end; ++p1)
        *p1 = *p1 > mx ? mx : *p1;
}
