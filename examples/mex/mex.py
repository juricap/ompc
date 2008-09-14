
from numpy import frombuffer, dtype, ndarray, array, c_
from weakref import WeakKeyDictionary, ref
from ctypes import *

freeers = {}
def free(addr):
    global freeers
    #print "Freeing ... ",  addr
    cdll.msvcrt.free(addr)
    del freeers[addr]

from os.path import split as psplit, join as pjoin
__pth = psplit(__file__)[0]

import inspect,dis

def get_nargout():
    """Return how many values the caller is expecting"""
    f = inspect.currentframe()
    f = f.f_back.f_back
    c = f.f_code
    i = f.f_lasti
    bytecode = c.co_code
    instruction = ord(bytecode[i+3])
    if instruction == dis.opmap['UNPACK_SEQUENCE']:
        howmany = ord(bytecode[i+4])
        return howmany
    elif instruction == dis.opmap['POP_TOP']:
        return 0
    return 1

# probably much better would be to use the idea from
# http://www.scipy.org/Cookbook/A_Numerical_Agnostic_Pyrex_Class
# too generate interface anytime necessary
    
class mxArray(Structure):
    _fields_ = [ ('pdata', POINTER(c_char)),
                 ('shape', POINTER(c_uint)),
                 ('strides', POINTER(c_uint)),
                 ('ndim', c_uint),
                 ('dtype',c_int),
                 ('flags',c_uint) ]
    __has_data = False
    def __init__(self,_a):
        local_ctype = c_double
        n = c_int()
        if type(_a) is not ndarray:
            if type(_a) is str:
                #_a = array(_a)
                local_ctype = c_char
                pythonapi.PyObject_AsReadBuffer(py_object(_a), byref(self.pdata), byref(n))
                self.__has_data = True
                self.shape = (c_ulong*2)(1,len(_a))
                self.strides = (c_ulong*2)(1, len(_a))
                self.ndim = c_ulong(2)
                self.dtype = 0
                self.flags = 0
            else:
                print '!!!!! Converting %s to ndarray'%type(_a)
                _a = array(_a).astype('f8')
                
                if _a.ndim < 2:
                    _a = c_[_a].T
                print _a
                sz = reduce(lambda x,y: x*y, _a.shape, 1)
                data = (local_ctype*sz)()
                for i in range(sz):
                    data[i] = float(_a.flat[i])
                self.pdata = cast(data,POINTER(c_char))
                self.__has_data = True
                self.shape = (c_ulong*_a.ndim)(*_a.shape)
                self.strides = (c_ulong*_a.ndim)(*_a.strides)
                self.ndim = c_ulong(_a.ndim)
                self.dtype = 6
                self.flags = 0
        elif _a.dtype == dtype('f8'):
            pythonapi.PyObject_AsReadBuffer(py_object(_a), byref(self.pdata), byref(n))
            if _a.ndim < 2:
                _a = c_[_a].T
            self.__has_data = True
            self.shape = (c_ulong*_a.ndim)(*_a.shape)
            self.strides = (c_ulong*_a.ndim)(*_a.strides)
            self.ndim = c_ulong(_a.ndim)
            self.dtype = 6
            self.flags = 0
        else:
            print '====>'
            print 'Converting %s to double'%_a.dtype
            _a = _a.astype('f8')
            print _a
            
            if _a.ndim < 2:
                _a = c_[_a].T
            print _a
            sz = reduce(lambda x,y: x*y, _a.shape, 1)
            data = (local_ctype*sz)()
            for i in range(sz):
                data[i] = float(_a.flat[i])
            self.pdata = cast(data,POINTER(c_char))
            self.__has_data = True
            self.shape = (c_ulong*_a.ndim)(*_a.shape)
            self.strides = (c_ulong*_a.ndim)(*_a.strides)
            self.ndim = c_ulong(_a.ndim)
            self.dtype = 6
            self.flags = 0
    
    def release_data(self):
        self.__has_data = False
        shape = [ self.shape[i] for i in xrange(self.ndim) ]
        sz = reduce(lambda x,y: x*y, shape)
        shape[0], shape[1] = shape[1], shape[0]
        ret = frombuffer((c_double*sz).from_address(addressof(self.pdata.contents)), 'f8', sz).reshape(shape)
        addr = addressof(self.pdata.contents)
        fn = lambda x: free(addr)
        freeers[addr] = (fn, ref(ret,fn))
        ret = ret.swapaxes(0,1)
        return ret
 
def mexFunc(fname,debug=False):
    if debug:
        fname += '_d'
    try:
        f = cdll.LoadLibrary('%s.dll'%fname).mexFunction
    except:
        f = cdll.LoadLibrary('%s.dll'%pjoin(__pth,fname)).mexFunction
    f.restype = None
    f.argtypes = (c_int, POINTER(POINTER(mxArray)), c_int, POINTER(POINTER(mxArray)))
    def func(*args):
        nargout = get_nargout()
        nargin = len(args)
        print nargin, args, nargout
        plhs = (POINTER(mxArray)*nargout)()
        prhs = (POINTER(mxArray)*nargin)(*[ pointer(mxArray(x)) for x in args ])
        f(nargout, plhs, nargin, prhs)
        if nargout == 1:
            return plhs[0].contents.release_data()
        print plhs
        return tuple([ x.contents.release_data() for x in plhs ])
    return func

#upConv = mexFunc('upConv')
#upConv = mexFunc('upConv',debug=True)

def test():
    from numpy import arange, concatenate, r_
    im = c_[ 0., 0., 0.107517, 0.074893, -0.46955, 0., 0.46955, -0.074893, -0.107517, 0., 0.]
    filt = c_[ 0.08838835, 0.35355339, 0.53033009, 0.35355339, 0.08838835]
    a = upConv(im, filt,'reflect1',r_[1, 2])
    print 'Result is', a

def test():
    from numpy import arange, concatenate
    derivate = mexFunc('derivate')
    print derivate(c_[arange(10.0)**2])
    print derivate(c_[arange(10.0)**2,arange(10.0)])

def test(): 
    from pylab import rand, plot, show
    lineintc = mexFunc('lineintc')
    x1, x2 = rand(2,10),rand(2,10)
    a = lineintc(x1, x2)
    print a.T

if __name__ == "__main__":
    test()