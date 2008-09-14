
#include <math.h>
#include <stdio.h>
#include "mex.h"

static const double PPD = 26.5;

inline void derivate( double* a, double* out, int len, double sampling )
{
    double nan = mxGetNaN();
    out[0] = nan;
    out[1] = nan;
    double tc = 1000.0/sampling;
    for ( int i = 2; i < len-2; i++ )
        out[i] = tc*( a[i+2] + a[i+1] - a[i-1] - a[i-2] )/(6.0*PPD);
    out[len-1] = nan;
    out[len-2] = nan;
}

void mexFunction(
    int nlhs,              // Number of left hand side (output) arguments
    mxArray *plhs[],       // Array of left hand side arguments
    int nrhs,              // Number of right hand side (input) arguments
    const mxArray *prhs[]  // Array of right hand side arguments
)
{
    double sampling = 4.0;
    if (nrhs < 1)
    {
        mexErrMsgTxt("Input vector missing!");
    }
    else if (nrhs == 2)
    {
        sampling = mxGetScalar(prhs[1]);
    }
    
    int len = mxGetM(prhs[0]);
    double *input = mxGetPr(prhs[0]);
    
    if ( mxGetN(prhs[0]) == 1 )
    {
        plhs[0] = mxCreateDoubleMatrix( len, 1, mxREAL );
        double *output = mxGetPr(plhs[0]);
        
        derivate( input, output, len, sampling );
    }
    else if ( mxGetN(prhs[0]) == 2 )
    {
        plhs[0] = mxCreateDoubleMatrix( len, 3, mxREAL );
        double *output = mxGetPr(plhs[0]);
        
        derivate( input, output, len, sampling );
        derivate( input+len, output+len, len, sampling );
        double* out1 = output;
        double* out2 = output+(len);
        double* out3 = output+(2*len);
        for ( int i = 0; i < len; i++ )
            out3[i] = sqrt( out1[i]*out1[i] + out2[i]*out2[i] );
    }
}
