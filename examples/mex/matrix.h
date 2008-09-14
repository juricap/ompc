
#include "mexlib.h"

#ifdef __cplusplus
extern "C" {
#else
#define bool int
#endif


void mxFree(void *p);
void *mxCalloc(mxsize_t n, mxsize_t sz);
int mxGetString(const mxArray *p, char *buf, int buflen);
int mexPrintf(const char* format, ...);
bool mxIsChar(const mxArray *p);
bool mxIsComplex(const mxArray *p);
bool mxIsSparse(const mxArray *p);
bool mxIsDouble(const mxArray *p);
bool mxIsNumeric(const mxArray *p);

#ifdef __cplusplus
}
#endif