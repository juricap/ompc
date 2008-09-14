
#include <cmath>
#include <list>
#include <stdexcept>

#include "mexlib.h"
#include "mex.h"

using namespace std;

list< mxArray* > __mex_vars;
list< mxMem* > __mex_mems;

char koko[1000];

mxArray* __new_mxArray(unsigned int _ndim, unsigned int *_shape, size_t _sz, mxtype _mt)
{
    mxArray *ret = new mxArray();
    size_t sz = 1, strides = _sz;
    int i;
    
    ret->shape = new unsigned int[_ndim];
    ret->strides = new unsigned int[_ndim];
    ret->ndim = _ndim;
    ret->dtype = _mt;
    
    for (i=0; i<_ndim; i++)
    {
        sz = sz*_shape[i];
        ret->shape[i] = _shape[i];
        ret->strides[_ndim-i] = strides;
        strides *= _shape[i];
    }
    
    printf("OMPCpy: Allocated %d bytes for %d dims %x!\n", sz*_sz, _ndim, ret);
    
    //ret->pdata = new char[sz*_sz];
    ret->pdata = koko;//(char*)malloc(sz*_sz);
    //memset(ret->pdata,0,sz*_sz);
    __mex_vars.push_back(ret);
    
    printf("OMPCpy: Allocated %d bytes for %d dims %x!\n", sz*_sz, _ndim, ret);
    
    return ret;
}

void __del_mxArray(mxArray* _m)
{
    delete [] _m->pdata;
    delete [] _m->shape;
    delete [] _m->strides;
    
    __mex_vars.remove(_m);
    delete _m;
}

mxMem* __new_mxMem(unsigned int _n)
{
    mxMem *ret = new mxMem();
    
    ret->pdata = new char[_n];
    memset(ret->pdata,0,_n);
    ret->n = _n;
    __mex_mems.push_back(ret);
    
    return ret;
}

void __del_mxMem(mxMem* _m)
{
    delete [] _m->pdata;
    __mex_mems.remove(_m);
    delete _m;
}

static const unsigned long __nan[2]={0xffffffff, 0x7fffffff};
double mxGetNaN()
{
    //return *(double*)__nan;
    return *(double*)&__nan;
}

int mxIsNaN(const double x)
{
    const unsigned long *p = (const unsigned long*)(&x);
    return __nan[0] == p[0] && __nan[1] == p[1];
}

void mexErrMsgTxt(char* _str_err)
{
    printf("Error: %s", _str_err);
    //throw new invalid_argument(_str_err);
}

size_t mxGetM(const mxArray* _m)
{
    return _m->shape[0];
}

size_t mxGetN(const mxArray* _m)
{
    size_t ret;
    if (_m->ndim == 1)
        ret = 1;
    else
        ret = _m->shape[1];
    return ret;
}

double* mxGetPr(const mxArray* _m)
{
    printf("pt = %X, data = %X\n", (unsigned int)_m, (unsigned int)(_m->pdata));
    return (double*)(_m->pdata);
}

double mxGetScalar(const mxArray* _m)
{
    return *(double*)(_m->pdata);
}

mxArray* mxCreateDoubleMatrix(int m, int n, mxComplexity ComplexFlag)
{
    mxArray *ret;
    static size_t dims[2];
    
    dims[0] = m;
    dims[1] = n;
    
    if (ComplexFlag == mxREAL)
        ret = __new_mxArray(2, dims, sizeof(double), mxtype_FLOAT64);
    else if (ComplexFlag == mxCOMPLEX)
        ret = __new_mxArray(2, dims, 2*sizeof(double), mxtype_FLOAT64);
    
    return ret;
}

mxArray* mxCreateNumericArray(int ndims, int* dims, mxClassID clsid, mxComplexity ComplexFlag)
{
    mxArray *ret;
    size_t newsz = sizeof(double);
    
    // check the arguments
    if (ndims <= 0) 
        mexErrMsgTxt("Dimensions must be greter than 0!");
    for (int i=0; i< ndims; i++)
        if (dims[i] <= 0) 
            mexErrMsgTxt("Dimensions must be greter than 0!");
    
    if (clsid == mxUINT8_CLASS)
        newsz = 1;
    
    if (ComplexFlag == mxREAL)
        ret = __new_mxArray((unsigned int)ndims, (unsigned int*)dims, newsz, mxtype_FLOAT64);
    else if (ComplexFlag == mxCOMPLEX)
        ret = __new_mxArray(2, (unsigned int*)dims, 2*newsz, mxtype_FLOAT64);
    
    return ret;
}

void mxDestroyArray(mxArray *_m)
{
    __del_mxArray(_m);
}

void mexRegisterWorkspaceCallback(__CALL_CB_TYPE _cb)
{
    __call_callback = _cb;
}

int mexCallMATLAB(int nlhs, mxArray **plhs, int nrhs, mxArray **prhs, const char *name)
{
    if (__call_callback == NULL)
    {
        mexErrMsgTxt("Callback to workspace not initialized!");
        return 1;
    }
    
    __call_callback(nlhs, plhs, nrhs, prhs, name);
    
    // return 0 if everything is oke;
    return 0;
}

void mexSetTrapFlag(int flag)
{
    // pass, don't do anything
}
