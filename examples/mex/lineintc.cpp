#include <math.h>
#include <vector>
#include "mex.h"

using namespace std;

typedef vector<double> xyvec;
double NaN;

extern void _main();

void lineint8(double x1,double y1,double x2,double y2,
             double x3,double y3,double x4,double y4,
             double *out)
{
   out[0] = NaN, out[1] = NaN;

   double den = ((y4-y3)*(x2-x1)-(x4-x3)*(y2-y1));
   double numa = ((x4-x3)*(y1-y3)-(y4-y3)*(x1-x3));
   double numb = ((x2-x1)*(y1-y3)-(y2-y1)*(x1-x3));
   if (den == 0)
   {
       // if nums are 0 they are coincident
       return;
   }
   double t1 = numa/den;
   double t2 = numb/den;
   if (t1 >= 0 && t1 <= 1 && t2 >= 0 && t2 <= 1)
   {
       out[0] = x1+t1*(x2-x1);
       out[1] = y1+t1*(y2-y1);
   }
}

void intersects(double *data1, int N1, double *data2, int N2, xyvec& out)
{
   int i, j;
   double *p1, *p2;

   for (p1=data1, i = 0; i< N1-1; i++, p1+=2)
   {
       double p1x, p1y, p2x, p2y;
       p1x = *p1; p1y = *(p1+1);
       p2x = *(p1+2); p2y = *(p1+3);
       for (p2 = data2, j = 0; j < N2-1; j++, p2+=2)
       {
           double p3x, p3y, p4x, p4y;
           p3x = *p2; p3y = *(p2+1);
           p4x = *(p2+2); p4y = *(p2+3);
           double ins[2];
           lineint8(p1x,p1y,p2x,p2y,p3x,p3y,p4x,p4y,ins);
           if (!mxIsNaN(ins[0]))
           {
               out.push_back(ins[0]);
               out.push_back(ins[1]);
           }
       }
   }
}

void mexFunction(
       int nlhs,              // Number of left hand side (output) arguments
       mxArray *plhs[],       // Array of left hand side arguments
       int nrhs,              // Number of right hand side (input) arguments
       const mxArray *prhs[]  // Array of right hand side arguments
)
{
       if (nrhs != 2)
       {
               mexErrMsgTxt("Two 2-row vectors are expected!");
       }

       NaN = mxGetNaN();
       int len1 = mxGetN(prhs[0]);
       int len2 = mxGetN(prhs[1]);
       double *input1 = mxGetPr(prhs[0]);
       double *input2 = mxGetPr(prhs[1]);
       xyvec tout;
       xyvec::iterator it;
       intersects(input1,len1,input2,len2,tout);
       plhs[0] = mxCreateDoubleMatrix(2, tout.size()/2, mxREAL );
   double *output = mxGetPr(plhs[0]);
   for (it = tout.begin(); it != tout.end(); it++, output++)
       *output = *it;
}
