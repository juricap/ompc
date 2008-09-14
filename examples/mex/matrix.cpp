
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "matrix.h"

#define MAXBUF 10000

int mexPrintf(const char *format, ...)
{
//     int count = 0, i = 0;
     char buf[MAXBUF];
     va_list marker;
     
     va_start( marker, format );     /* Initialize variable arguments. */
//     while( p != -1 )
//     {
//       sum += i;
//       count++;
//       i = va_arg( marker, void*);
//     }
//     va_end( marker );              /* Reset variable arguments.      */
//     return strlen(buf);
    int ret = sprintf(buf, format, &marker);
    printf(buf);
    return ret;
}

// int mexPrintf(const char *fmt, ...)
// {
//     int ret;
//     va_list ap;
//     va_start(ap, fmt);
//     ret = vsprintf(buffer, fmt, ap);
//     va_end(ap);
//     return ret;
// }

void mxFree(void *p)
{
    __del_mxArray((mxArray*)p);
}

void *mxCalloc(mxsize_t n, mxsize_t sz)
{
    return __new_mxMem(n*sz)->pdata;
}

int mxGetString(const mxArray *_m, char *buf, int buflen)
{
    strncpy(buf,(char*)(_m->pdata),buflen);
    return strlen(buf);
}

bool mxIsChar(const mxArray *p)
{
    return p->dtype == mxtype_CHAR;
}

bool mxIsComplex(const mxArray *p)
{
    return (p->dtype == mxtype_COMPLEXFLOAT32 ||
            p->dtype == mxtype_COMPLEXFLOAT64);
}

bool mxIsSparse(const mxArray *p)
{
    return (p->flags & mxflag_SPARSE > 0);
}

bool mxIsDouble(const mxArray *p)
{
    return (p->dtype == mxtype_FLOAT64);
}

bool mxIsNumeric(const mxArray *p)
{
    return (p->flags == 0);
}

