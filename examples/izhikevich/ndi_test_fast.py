
import numpy as np

from ctypes import cdll, c_int, POINTER
ndi0_fast = cdll.ompc_fast.ompc_ndi0
ndi1_fast = cdll.ompc_fast.ompc_ndi1
def _ndi(fn, shp, *ins):
    ndim = len(shp)
    n = (c_int*ndim)(*map(len,ins))
    cins = (POINTER(c_int)*ndim)()
    for i in xrange(ndim): cins[i] = (c_int*n[i])(*ins[i])
    shp = (c_int*ndim)(*shp)
    nout = n[0]
    for x in n[1:]: nout *= x
    out = np.zeros(nout, 'i4')
    fn(ndim, n, cins, shp, out.ctypes)
    return out

def _ndi0(shp, *ins):
    return _ndi(ndi0_fast, shp, *ins)

def _ndi1(shp, *ins):
    return _ndi(ndi1_fast, shp, *ins)

print _ndi0((5,3,2), range(5), range(3), range(2))
print _ndi0((1000,1000), range(1000), range(800))

print _ndi1((5,3,2), range(1,6), range(1,4), range(1,3))
print _ndi1((1000,1000), range(1,1001), range(1,801))

import time
print (list(_ndi0((5,3,2), range(5), range(3), range(2))))
t0 = time.clock()
for x in range(10):
    list(_ndi0((5,3,2), range(5), range(3), range(2)))
print( (time.clock()-t0)/10.0)

t0 = time.clock()
a = list(_ndi0((1000, 1000), range(1000), range(800)))
print( (time.clock()-t0))

