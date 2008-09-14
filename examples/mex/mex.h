

#include "mexlib.h"

#define NULL 0

#ifdef __cplusplus
extern "C" {
#endif

double mxGetNaN();
int mxIsNaN(const double x);
void mexErrMsgTxt(char*);
size_t mxGetM(const mxArray*);
size_t mxGetN(const mxArray*);
double* mxGetPr(const mxArray*);
double mxGetScalar(const mxArray* _m);

mxArray* mxCreateDoubleMatrix(int m, int n, mxComplexity ComplexFlag);
mxArray* mxCreateNumericArray(int ndims, int* dims, mxClassID clsid, mxComplexity ComplexFlag);
void mxDestroyArray(mxArray *);

int mexCallMATLAB(int nlhs, mxArray **plhs, int nrhs, mxArray **prhs, const char *name);
void mexSetTrapFlag(int flag);

typedef void (*__CALL_CB_TYPE)(int nlhs, mxArray **plhs, int nrhs, mxArray **prhs, const char *name);
static __CALL_CB_TYPE __call_callback = NULL;

void mexRegisterWorkspaceCallback(__CALL_CB_TYPE _cb);

#ifdef __cplusplus
}
#endif