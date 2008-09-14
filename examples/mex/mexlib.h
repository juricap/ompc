
#ifndef __mexlib_h__
#define __mexlib_h__

#define mxREAL 8
#define mxCOMPLEX 16

typedef enum {
    mxtype_CHAR,
    mxtype_INT16,
    mxtype_INT32,
    mxtype_INT64,
    mxtype_FLOAT32,
    mxtype_COMPLEXFLOAT32,
    mxtype_FLOAT64,
    mxtype_COMPLEXFLOAT64
} mxtype;

typedef enum {
        mxUNKNOWN_CLASS = 0,
        mxCELL_CLASS,
        mxSTRUCT_CLASS,
        mxLOGICAL_CLASS,
        mxCHAR_CLASS,
        mxNOTUSED_CLASS,            // not used
        mxDOUBLE_CLASS,
        mxSINGLE_CLASS,
        mxINT8_CLASS,
        mxUINT8_CLASS,
        mxINT16_CLASS,
        mxUINT16_CLASS,
        mxINT32_CLASS,
        mxUINT32_CLASS,
        mxINT64_CLASS,
        mxUINT64_CLASS,
        mxFUNCTION_CLASS
} mxClassID;

#define mxflag_NUMERIC 1
#define mxflag_SPARSE 8
#define mxflag_CLASS 16

struct _pxMem
{
   char *pdata;
   unsigned int n;
};
//#define pxMem struct _pxMem;

struct _pxArray
{
   char *pdata;
   unsigned int *shape;
   unsigned int *strides;
   unsigned int ndim;
   
   mxtype dtype;
   unsigned int flags;
};
#define pxArray struct _pxArray;

typedef struct _pxArray mxArray;
typedef struct _pxMem mxMem;
typedef unsigned int mxComplexity;

typedef unsigned int mxsize_t;

mxArray* __new_mxArray(mxsize_t);
void __del_mxArray(mxArray*);
mxMem* __new_mxMem(mxsize_t);
void __del_mxMem(mxMem*);

#endif