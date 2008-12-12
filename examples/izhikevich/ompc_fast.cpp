
#include <stdio.h>

#define API __declspec(dllexport)

inline int _prod(int N, int* p)
{
    int i;
    int out = 1;
    for (i=0; i<N; i++)
        out *= p[i];
    return out;
}


class gen
{
public:
    gen() {};
    virtual ~gen() {};
    virtual int next() = 0;
};

class ccr : public gen
{
    int *_a, _n, _cp;
    int _i, _j;
public:
    ccr(int n, int* a, int cp)
    {
        _i = 0;
        _j = 0;
        _n = n;
        _a = a;
        _cp = cp;
    }
    virtual int next()
    {
        int res = _a[_i];
        _j++;
        if (_j == _cp)
        {
            _j=0;
            _i++;
            if (_i == _n) _i = 0;
        }
        return res;
    }
};

class cr : public gen
{
    int *_a, _n, _cp;
    int _i, _j;
    int _die;
public:
    cr(int n, int* a, int cp)
    {
        _die = 0;
        _i = 0;
        _j = 0;
        _n = n;
        _a = a;
        _cp = cp;
    }
    virtual int next()
    {
        if (_die) throw "StopIteration";
        int res = _a[_i];
        _j++;
        if (_j == _cp)
        {
            _j=0;
            _i++;
            if (_i == _n) _die = 1;
        }
        return res;
    }
};

extern "C" {

API void ompc_sum(double* A, int ndim, int* msize, int dim, double* out)
{
    int i, j, k;
    int for_end = 1, stride_big = 5, cp;
    for (i=0; i<dim; i++) for_end *= msize[i];
    for (i=1; i<dim+1; i++) stride_big *= msize[i];
    if (dim > 0)
        cp = stride_big - stride_big/msize[dim];
    else
        cp = stride_big - 1;
    int prod_msize = for_end;
    for (i=dim; i<ndim; i++) prod_msize *= msize[i];
    int oi = 0;
    i = 0;
    while (oi < prod_msize/msize[dim])
    {
        for (k=0; k<for_end; k+=1)
        {
            double s = 0.0;
            for (j=i; j<i+msize[dim]*for_end; j+=for_end)
                s += A[j];
            out[oi] = s;
            oi += 1;
            i += 1;
        }
        i += cp;
    }
}

API void ompc_ndi0(int ndim, int* n, int **ins, int* shp, int* out)
{
    int i, j;
    int p = _prod(ndim, n);
    int cp = 1;
    int* fp = new int[ndim];
    fp[0] = 1;
    for (i=0; i<ndim-1; i++)
        fp[i+1] = fp[i]*shp[i];
    gen** gs = new gen*[ndim];
    for (i=0; i<ndim-1; i++)
    {
        gs[i] = new ccr(n[i], ins[i], cp);
        cp *= n[i];
    }
    gs[ndim-1] = new cr(n[i], ins[ndim-1], cp);
    for (i=0; i<p; i++)
    {
        int sum = 0;
        for (j=0; j<ndim; j++)
            sum += gs[j]->next()*fp[j];
        *out = sum;
        *out++;
    }
    for (i=0; i<ndim; i++)
        delete gs[i];
    delete [] gs;
    delete [] fp;
}

API void ompc_ndi1(int ndim, int* n, int **ins, int* shp, int* out)
{
    int i, j;
    int p = _prod(ndim, n);
    int cp = 1;
    int* fp = new int[ndim];
    fp[0] = 1;
    for (i=0; i<ndim-1; i++)
        fp[i+1] = fp[i]*shp[i];
    gen** gs = new gen*[ndim];
    for (i=0; i<ndim-1; i++)
    {
        gs[i] = new ccr(n[i], ins[i], cp);
        cp *= n[i];
    }
    gs[ndim-1] = new cr(n[i], ins[ndim-1], cp);
    for (i=0; i<p; i++)
    {
        int sum = 0;
        for (j=0; j<ndim; j++)
            sum += (gs[j]->next()-1)*fp[j];
        *out = sum;
        *out++;
    }
    for (i=0; i<ndim; i++)
        delete gs[i];
    delete [] gs;
    delete [] fp;
}

/*
from ctypes import *
ndi_fast = cdll.ompc_fast
def _ndi(shp, *ins):
    ndim = len(shp)
    n = (c_int*ndim)(*map(len,ins))
    cins = (POINTER(c_int)*ndim)()
    for i in xrange(ndim): cins[i] = (c_int*n[i])(*ins[i])
    shp = (c_int*ndim)(*shp)
    nout = n[0]
    for x in n[1:]: nout *= x
    out = zeros(nout, 'i4')
    ndi_fast(ndim, n, cins, shp, out.ctypes)
    return out
*/
}
