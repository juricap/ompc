
import ompc
from ompclib.ompclib_numpy import _marray, _size, _dtype

def build_return(nargout, *args):
    ret = []
    for x in args[:nargout]:
        if isinstance(x, _marray): ret += [ x ]
        else: ret += [ _marray(_dtype(x), _size(x), x) ]
    if len(ret) == 1:
        ret = ret[0]
    return ret

def func1():
    nargout = 1
    a, b = 1, 2
    return build_return(nargout, a, b)

def func2():
    nargout = 2
    a, b = [1,2,3], [[2]]
    return build_return(nargout, a, b)

from ompc import byteplay
c1 = byteplay.Code.from_code(func1.func_code)
c2 = byteplay.Code.from_code(func2.func_code)

print c1.code
print c2.code

print func1()
a, b = func2()
print a
print b
