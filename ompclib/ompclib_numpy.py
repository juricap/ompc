
# This file is a part of OMPC (http://ompc.juricap.com/)
# 
# for testing:
#  import ompclib_numpy; reload(ompclib_numpy); from ompclib_numpy import *

# TODO
# - remove all references to array, use "ompc_base._init_data" instead

import sys, os; sys.path.append(os.path.abspath('..'))

from itertools import izip as _izip, cycle as _cycle, repeat as _repeat
from ompc import _get_narginout
import numpy as np
import pylab as mpl

# Functions that are to be exported have to be listed in the __ompc_all__
# array.
# This decorator adds a function to the "toolbox-less" OMPC base library.
__ompc_all__ = ['end', 'mslice', 'mstring', 'OMPCSEMI',
                'OMPCException', 'elmul', 'elpow']

def _ompc_base(func):
    global __ompc_all__
    __ompc_all__ += [ func.__name__ ]
    return func

OMPCSEMI = Ellipsis
OMPCEND = None
end = OMPCEND

_dtype2numpy = {'complex': 'complex128',
                'double': 'f8', 'single': 'f4',
                'int32': 'i4', 'uint32': 'u4',
                'int16': 'i2', 'uint16': 'u2',
                'int8': 'i1', 'uint8': 'u1',
                'char': 'u1',
                'bool': 'bool',
               }

# errors and warnings

class OMPCException(Exception):
    def __init__(self,msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

@_ompc_base
def error(x):
    raise OMPCException(x)

class mvar(object):
    @staticmethod
    def _DataObject(dtype, data):
        return np.array(data, dtype=_dtype2numpy[dtype])
    
    def __new__(cls, *args, **kwargs):
        a = super(mvar, cls).__new__(cls, *args, **kwargs)
        a._a = None
        a.dtype = 'double'
        a.msize = (0, 0)
        return a
    def _init_data(self, dtype, msize, data):
        self.dtype = dtype
        self.msize = msize
        self._a = self._DataObject(dtype, data)
    def __call__(self, *i):
        mview = self.__getitem1__(i)
        mview.__ompc_view__ = _mview(self, i, False)
        return mview
    def _ctypes_get(self):
        return self._a.ctypes
    ctypes = property(_ctypes_get, None, None, 
                      "Ctypes-wrapped data object.")
    def _lvalue_set(self, val):
        assert hasattr(self, '__ompc_view__')
        o = self.__ompc_view__
        # FIXME: o.linear
        o.viewed.__setitem1__(o.ins, val)
    lvalue = property(None, _lvalue_set, None, "")
    def __copy__(self):
        return _marray(self.dtype, self.msize, self._a.copy())
    
    def __deepcopy__(self):
        return _marray(self.dtype, self.msize, self._a.copy())
    
    # FIXME: warn people about using numpy functions directly
#     def __array__(self):
#         print repr(self)
#         print self
#         raise NotImplementedError("At the moment using numpy functions " \
#               "directly is not possible! Please read the documentation at " \
#               "http://ompc.juricap.com/documentation/.")

class _mview(mvar):
    def __init__(self, viewed, ins, linear):
        self.viewed = viewed
        self.ins = ins
        self.linear = linear
    def __repr__(self):
        return "_mview(%r, %r, %r)"%(self.viewed, self.ins, self.linear)
    def __str__(self):
        return "<view of %r>"%(self.viewed)

class _el:
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right
    
    def __pow__(self, right):
        if self.left is None: return _el(right=right)
        return self.left.__elpow__(right)
    def __rpow__(self, left):
        if self.right is None: return _el(left=left)
        return left.__elpow__(self.right)
    
    def __mul__(self, right):
        if self.left is None: return _el(right=right)
        return self.left.__elmul__(right)
    def __rmul__(self, left):
        if self.right is None: return _el(left=left)
        return left.__elmul__(self.right)

elpow = _el()
elmul = _el()

def _dsize(dtype):
    return _dsize_dict[dtype]

def _flatten(seq):
    for item in seq:
        if _isscalar(item) and not hasattr(item, '__len__'):
            yield item
        else:
            for subitem in _flatten(item):
               yield subitem

def _ndi(*i):
    """Returns a generator of tuples that iterate over elements specified
    by slices and indices in `i`."""
    from itertools import chain, repeat, cycle, izip
    r = lambda x: range(x.start, x.stop, x.step is None and 1 or x.step)
    res = []
    for x in i:
        if isinstance(x, slice): res.append(r(x))
        elif _isscalar(x): res.append([x])
        else: res.append(x)
    i = res
    cp = 1
    gs = []
    for x in i[:-1]:
        gs += [ cycle(chain(*(repeat(j,cp) for j in x))) ]
        cp *= len(x)
    gs += [ chain(*(repeat(j,cp) for j in i[-1])) ]
    return izip(*gs)

def _isscalar(A):
    if isinstance(A, str):
        return False
    elif hasattr(A, '__len__') and len(A) > 1:
        return False
    elif hasattr(A, '__getitem__'):
        try: A[1]
        except: return True
        else: return False
    elif hasattr(A, '__iter__'):
        return False
    # doesn't have length nor multiple elements and doesn't support iteration
    return True

def _typegreater_(Adt, Bdt):
    """Returns type with higher precision."""
    if isinstance(Adt, _marray): Adt = Adt.dtype
    if isinstance(Bdt, _marray): Bdt = Bdt.dtype
    return _dsize_dict[Adt] >= _dsize_dict[Bdt] and Adt or Bdt

def _typegreater(Adt, Bdt):
    """Returns type with higher precision."""
    return _dsize_dict[Adt] >= _dsize_dict[Bdt] and Adt or Bdt

def _dtype(X):
#     from operator import isSequenceType
#     while isSequenceType(X):
#         X = X[0]
#     res = tuple(reversed(shp))
    # FIXME: return
    if isinstance(X, str):
        return 'char'
    return 'double'

def _size(X, d=None):
    if isinstance(X, _marray):
        res = X.msize
    elif _isscalar(X):
        return (1, 1)
    else:
        from operator import isSequenceType
        shp = []
        while isSequenceType(X):
            shp.append(len(X))
            X = X[0]
        res = tuple(reversed(shp))
    # minimum shape is 2 dimensional
    if len(res) == 1:
        res = (1, res[0])
    if d is None:
        return res
    else:
        return res[d]

def _ndshape(msize, *i):
    """Determine the shape of a view on A with slicing specified in `i`.
    """
    shp = []
    for idim, x in enumerate(i):
        if isinstance(x, slice):
            start, stop, step = x.start, x.stop, x.step
            if x.start is None: start = 0
            if x.stop == sys.maxint or x.stop is None: stop = msize[idim]
            if x.step is None: step = 1
            shp.append( len(range(start,stop,step)) )
        elif _isscalar(x):
            shp.append(1)
        elif hasattr(x, '__len__'):
            shp.append(len(x))
        else:
            raise NotImplementedError()
    if len(shp) == 1: shp[:0] = [1]
    return shp

def _ndshape1(msize, *i):
    """Determine shape of a view on size msize with slicing specified in `i`.
    """
    shp = []
    for idim, x in enumerate(i):
        if isinstance(x, _mslice):
            if x.hasnoend():
                shp.append( len(mslice[x.start:x.step:msize[idim]]) )
            else:
                shp.append( len(x) )
        elif _isscalar(x):
            shp.append(1)
        elif hasattr(x, '__len__'):
            shp.append(len(x))
        else:
            if isinstance(x, slice):
                raise NotImplementedError()
                shp.append(mrange(x))
            else:
                raise NotImplementedError()
    #if len(shp) == 1: shp[:0] = [1]
    if len(shp) == 1:
        if msize[0] == 1: shp[:0] = [1]
        else: shp.append(1)
    return shp

@_ompc_base
def isempty(A):
    return np.prod(A.msize) == 0

def _dot(A, B):
    if not isinstance(A, _marray) or not isinstance(B, _marray):
        raise NotImplementedError("arguments must be 'marray's.")
    # FIXME: wrong dtype, needs to be the higher one
    na = np.dot(B._a, A._a)
    return _marray('double', na.shape[::-1], na)

def _squeeze(A):
    res = A.__copy__()
    res.msize = [ x for x in res.msize if x > 1 ]
    return res

def _msize(*args):
    if len(args) == 1 and hasattr(args, '__len__'):
        args = args[0]
    if len(args) > 2 and args[-1] == 1: args = args[:-1]
    if len(args) == 1:
        if construct: args = (args[0], args[0])
        else: args = (args[0], 1)
    return args

import __builtin__
def doublestr(x,prec=4):
    try:
        float(x)
    except:
        return x
    else:
        return '%6s'%__builtin__.round(x,4)

import __builtin__
def complexstr(x,prec=4):
    try:
        x = complex(x)
    except:
        return x
    else:
        return '%6s %s %6sj'%(__builtin__.round(x.real,4),
                              x.imag >= 0.0 and '+' or '-',
                              __builtin__.round(x.imag,4))

def print_marray(A, ans=True):
    nstr = doublestr
    if A.dtype == 'complex': nstr = complexstr
    pre = ''
    if ans:
        pre = '\nans = \n\n'
    if len(A.msize) > 2:
        for i in _ndi(*[slice(0,x) for x in A.msize[2:]]):
            pre += '(:, :, %s)\n\n'%', '.join([str(x+1) for x in i])
            cur = (slice(0,A.msize[0]), slice(0, A.msize[1])) + i
            sA = A.__getitem__(cur)
            sA.msize = A.msize[:2]
            pre += print_marray(sA, False)
        return pre
    else:
        #return str(A._a.T) + '\n\n'
        M, N = A.msize
        if N < 10: srow = lambda i: A._a[:,i]
        else: srow = lambda i: list(A._a[[0, 1, 2],i]) + \
                               ['...'] + \
                               list(A._a[[N-3, N-2, N-1], i])
        if M < 10: rows = ( srow(i) for i in xrange(M) )
        else: rows = [ srow(i) for i in xrange(3) ] + \
                     [('...',)] + \
                     [ srow(i) for i in [M-3,M-2,M-1] ]
        res = pre + '  ' + \
            '\n  '.join(', '.join(map(nstr,x)) for x in rows)
        if ans: res += '\n\n'
        return res

@_ompc_base
def disp(X):
    if isinstance(X, _marray):
        print print_marray(X, False)
    if isinstance(X, str):
        print X
    else:
        print X

class _marray(mvar):
    
    @staticmethod
    def empty(shp, dtype):
        return _marray(dtype, shp)
    
    @staticmethod
    def zeros(shp, dtype):
        na = _marray(dtype, shp)
        na._a.flat[:] = 0 #np.zeros(na.msize[::-1], _dtype2numpy[dtype])
        #na.msize = shp
        return na
    
    @staticmethod
    def ones(shp, dtype):
        na = _marray(dtype, shp)
        na._a.flat[:] = 1 #np.ones(na.msize[::-1], _dtype2numpy[dtype])
        #na.msize = shp
        return na
    
    def __init__(self, dtype, msize, a=None):
        from operator import isSequenceType
        if not isSequenceType(msize):
            msize = (msize, msize)
        elif len(msize) == 1:
            msize = (msize[0], 1)
        if a is None:
            self._a = np.empty(msize[::-1], _dtype2numpy[dtype])
        elif isinstance(a, np.ndarray):
            self._a = a
        else:
            self._a = np.array(a, _dtype2numpy[dtype]).reshape(msize[::-1])
        self.msize = msize
        self.dtype = dtype
    
    def __copy__(self):
        return _marray(self.dtype, self.msize, self._a.copy())
    def __deepcopy__(self):
        return _marray(self.dtype, self.msize, self._a.copy())
    
    # operators
    def __elpow__(self, him):
        if isinstance(him, _marray): him = him._a
        return _marray(self.dtype, self.msize, self._a**him)
    
    def __elmul__(self, him):
        if isinstance(him, _marray): him = him._a
        return _marray(self.dtype, self.msize, self._a*him)
    
    def __mul__(self, right):   
        if len(self.msize) != 2:
            # FIXME
            raise OMPCError('??? Error using ==> mtimes\n'
                            'Input arguments must be 2-D')
        # if multiplying with _el object, call the elementwise operation
        if isinstance(right, _el): return _el(left=self)
        elif _isscalar(right): return self.__elmul__(right)
        # matrix multiplication
        return _dot(self, right)
    
    def __rmul__(self, left):
        # if multiplying with _el object, call the elementwise operation
        if isinstance(left, _el): return _el(right=self)
        elif _isscalar(left): return self.__elmul__(left)
        # matrix multiplication
        return _dot(left, self)
    
    def __add__(self, him):
        if isinstance(him, _marray): him = him._a
        na = self._a + him
        return _marray(self.dtype, na.shape[::-1], na)
    __radd__ = __add__
    
    def __sub__(self, him):
        if isinstance(him, _marray): him = him._a
        na = self._a-him
        return _marray(self.dtype, na.shape[::-1], na)
    def __rsub__(self, him):
        if isinstance(him, _marray): him = him._a
        na = him - self._a
        return _marray(self.dtype, na.shape[::-1], na)
    
    def __div__(self, him):
        if isinstance(him, _marray): him = him._a
        na = self._a / him
        return _marray(self.dtype, na.shape[::-1], na)
    def __rdiv__(self, him):
        if isinstance(him, _marray): him = him._a
        na = self._a + him / self._a
        return _marray(self.dtype, na.shape[::-1], na)
    
    def __neg__(self):
        return _marray(self.dtype, self.msize, -self._a)
    
    # comparisons
    def __ge__(self, other):
        if isinstance(other, _marray): other = other._a
        return _marray('bool', self.msize, self._a >= other)
    def __gt__(self, other):
        if isinstance(other, _marray): other = other._a
        return _marray('bool', self.msize, self._a > other)
    def __le__(self, other):
        if isinstance(other, _marray): other = other._a
        return _marray('bool', self.msize, self._a <= other)
    def __lt__(self, other):
        if isinstance(other, _marray): other = other._a
        return _marray('bool', self.msize, self._a < other)
    def __eq__(self, other):
        if isinstance(other, _marray): other = other._a
        return _marray('bool', self.msize, self._a == other)
    def __ne__(self, other):
        if isinstance(other, _marray): other = other._a
        return _marray('bool', self.msize, self._a != other)
            
    
    # element access
    def __iter__(self):
        #return (_marray(self.dtype, (1,1), x) for x in self._a.flat )
        return iter(self._a.flat)#(_marray(self.dtype, (1,1), x) for x in self._a.flat )
    
    def __len__(self):
        return max(self.msize)
    
    def __getitem__(self, i):
        # determine the size of the new array
        #if not hasattr(i, '__len__'): i = [i]
        nshp = _ndshape(self.msize, *i)
        #return _marray(self.dtype, nshp, self._a.__getitem__(reversed(i)))
        return _marray(self.dtype, nshp, self._a.__getitem__(i[::-1]))
    
    # >> a = reshape(1:15,5,3)
    # >> a(eye(3)==1)
    # ans = [1, 5, 9]
    def __getitem1__(self, i):
        # determine the size of the new array
        #if not hasattr(i, '__len__'): i = [i]
        nshp = _ndshape1(self.msize, *i)
        ri = []
        if len(i) == 1:
            if self.msize[0] == 1: ri = (i[0]._a.astype('i4').reshape(-1)-1, 0)
            elif self.msize[1] == 1: ri = (0, i[0]._a.astype('i4').reshape(-1)-1)
            else:
                raise NotImplementedError()
        else:
            di = len(self.msize)-1
            for x in reversed(i):
                if isinstance(x, _marray): ri.append(x._a.astype('i4').reshape(-1)-1)
                elif isinstance(x, _mslice): ri.append(x.__base0__(self.msize[di]))
                else: ri.append(x-1)
                di -= 1
        na = self._a.__getitem__(tuple(ri))
        return _marray(self.dtype, nshp, na.reshape(nshp[::-1]))
    
    def __setitem__(self, i, val):
        if isinstance(val, _marray): val = val._a
        self._a.__setitem__(i[::-1], val)
    
    def __setitem1__(self, i, val):
        # determine the size of the new array
        if isinstance(val, _marray): val = val._a
        ri = []
        if len(i) == 1:
            # stupid numpy a = rand(1,10); b = rand(1,2); a[0,[3,4]] = b
            # doesn't work
            if self.msize[0] == 1:
                ri = (i[0]._a.astype('i4').reshape(-1)-1, 0)
                val = val[0]
            elif self.msize[1] == 1:
                ri = (0, i[0]._a.astype('i4').reshape(-1)-1)
                val = val[0]
            else:
                raise NotImplementedError()
        else:
            di = len(self.msize)-1
            for x in reversed(i):
                if isinstance(x, _marray): ri.append(x._a.astype('i4').reshape(-1)-1)
                elif isinstance(x, _mslice): ri.append(x.__base0__(self.msize[di]))
                else: ri.append(x-1)
                di -= 1
        self._a.__setitem__(tuple(ri), val)
    
    # properties
    def transposed(self):
        assert len(self.msize) == 2
        return _marray(self.dtype, self.msize[::-1], 
                        self._a.T.reshape(self.msize).copy())
    def ctransposed(self):
        assert len(self.msize) == 2
        return _marray(self.dtype, self.msize[::-1], 
                        self._a.conj().T.reshape(self.msize).copy())
    T = property(transposed, None, None, "Transpose.")
    cT = property(transposed, None, None, "Conjugate transpose.")
    
    # IO
    def __str__(self):
        return print_marray(self)
    def __repr__(self):
        return "marray(%r, %r)"%(self.dtype, self.msize)

class mcellarray(mvar, list):
    pass

# from the end of 
# http://code.activestate.com/recipes/52558/
class _MEnd(object):
    '''This object serves as an emulator of the "end" statement of MATLAB.
    We want to use the "is" operator therefore we need a singletion.'''
    __instance = None  # the unique instance
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
            object.__init__(cls.__instance)
        return cls.__instance
    def __init__(self):
        # prevent the automatic call of object's __init__, it is init-ed once
        # in the __new__ function
        pass
    def __repr__(self):
        return 'end'
    def __str__(self):
        return '(m-end object)'
    def __int__(self):
        return sys.maxint

end = _MEnd()

def _mslicelen(start, stop, step):
    if stop is end or stop is None:
        return sys.maxint
    return int(np.floor(stop-start)/step + 1)

class _mslice(mvar):
    """m-slice MATLAB style slice object.
    You can instantiate this class only by the helper mslice:
    >>> mslice[1:10]
    """
    def __init__(self, start, stop=None, step=None):
        raise NotImplementedError("Direct instantiation is not allowed.")
    
    def init(self, start, stop, step):
        if start is None: start = 1
        if step is None: step = 1
        self.start = start
        self.stop = stop
        self.step = step
        self.dtype = 'double'
        self.msize = (1, _mslicelen(self.start, self.stop, self.step))
    
    def init_data(self):
        if self._a is None:
            self._a = np.array(list(self), dtype='f8').reshape(self.msize[::-1])
    
    def evaluate_end(self, i):
        start = self.start
        step = self.step
        stop = self.stop
        if stop is end:
            return mslice[start:step:i]
        else:
            return self
    
    def _ctypes_get(self):
        # Create and initialize a real data buffer, then let the default 
        # function to return the ctypes pointer
        if self.stop is end:
            raise RuntimeError("Infinite slice can be only used as an index.")
            # return None
        self.init_data()
        return self._a.ctypes
    ctypes = property(_ctypes_get, None, None, 
                      "Ctypes-wrapped data object.")
    
    def __iter__(self):
        value = self.start
        while value <= self.stop:
            yield float(value)
            value += self.step
    
    def __getitem__(self, i):
        self.init_data()
        na = self._a.__getitem__(i)
        return _marray('double', na.shape[::-1], na.reshape(na.shape[::-1]))
    
    def __getitem1__(self, i):
        self.init_data()
        return _marray('double', self.msize, self._a).__getitem1__(i)
    
    def __len__(self):
        if self.stop is end:
            # FIXME: how should this be done
            # raise AssertionError("This is impossible for a code translated "
            #                      "from a functional MATLAB code.")
            # Python allows returning of positive integers only!
            return sys.maxint
        return _mslicelen(self.start, self.stop, self.step)
    
    def __repr__(self):
        return 'mslice[%r:%r:%r]'%\
                            (self.start, self.step, self.stop)
    def __str__(self):
        if self.stop is None:
            it = iter(self)
            return ', '.join( str(it.next()) for i in xrange(3) ) + ' ...'
        elif len(self) > 10:
            it = iter(self)
            retval = self.__repr__() + '\n'
            retval += ', '.join( str(it.next()) for i in xrange(3) ) + ' ... '
            lastval = self.start + (len(self)-1)*self.step
            return retval + str(lastval)
        return ', '.join( map(str, self) )
    
    def hasnoend(self):
        'Returns true if "self.stop is end".'
        return self.stop is end
    
    def __copy__(self):
        self.init_data()
        return _marray(self.dtype, self.msize, self._a.copy())
        
    def __deepcopy__(self):
        self.init_data()
        return _marray(self.dtype, self.msize, self._a.copy())

    def __base0__(self,shp=None):
        if self.hasnoend():
            assert shp is not None
            return slice(self.start-1, shp, self.step)
        return slice(self.start-1, self.stop, self.step)

class _mslice_helper:
    def __getitem__(self, i):
        s = _mslice.__new__(_mslice)
        # FIXME: there is no way of differentiating between mslice[:]
        # and mslice[0:], the second will never appear in a code written for
        # MATLAB.
        # !!! actually, maybe possible by look-back in the stack ?!!
        start, stop, step = i.start, end, 1
        if i.step is None:
            # there are only 2 arguments, stop is i.stop
            if i.start == 0 and i.stop == sys.maxint:
                # a special case
                start = 1
            elif i.stop == sys.maxint:
                # this is what happens when syntax [start:] is used
                raise IndexError(
                    'Use 2- and 3-slices only. Use "end" instead of "None".')
            else: stop = i.stop
        else:
            # there are all 3 arguments, stop is actually i.step
            # 1:2:10 -> slice(1,2,10) -> mslice(1,10,2)
            stop = i.step
            step = i.stop
        s.init(start, stop, step)
        return s

class _mslice_helper:
    def __getitem__(self, i):
        s = _mslice.__new__(_mslice)
        # FIXME: there is no way of differentiating between mslice[:]
        # and mslice[0:], the second will never appear in a code written for
        # MATLAB.
        # !!! actually, maybe possible by look-back in the stack ?!!
        start, stop, step = i.start, end, 1
        if i.step is None:
            # there are only 2 arguments, stop is i.stop
            if i.start == 0 and i.stop == sys.maxint:
                # a special case
                start = 1
            elif i.stop == sys.maxint:
                # this is what happens when syntax [start:] is used
                raise IndexError(
                    'Use 2- and 3-slices only. Use "end" instead of "None".')
            else: stop = i.stop
        else:
            # there are all 3 arguments, stop is actually i.step
            # 1:2:10 -> slice(1,2,10) -> mslice(1,10,2)
            stop = i.step
            step = i.stop
        s.init(start, stop, step)
        if not s.hasnoend():
            s.init_data()
            return _marray('double', s.msize, s._a.copy())
        return s

mslice = _mslice_helper()

class mstring(mvar, str):
    def __init__(self, s):
        mvar.__init__(self)
        self.dtype = 'char'
        self.msize = (1, len(s))
        self._a = s
    def __len__(self):
        return len(self._a)
    def __str__(self):
        return self._a
    def __repr__(self):
        return 'mstring(%r)'%self._a

def _m_constructor_args(*X):
    from operator import isSequenceType
    dtype = 'double'
    if type(X[-1]) is str:
        dtype = X[-1]
        X = X[:-1]
    if len(X) == 1 and isSequenceType(X):
        X = X[0]
    return X, dtype

@_ompc_base
def empty(*X):
    # check for class
    X, dt = _m_constructor_args(*X)
    return _marray.empty(X, dt)

@_ompc_base
def zeros(*X):
    # check for class
    X, dt = _m_constructor_args(*X)
    return _marray.zeros(X, dt)

@_ompc_base
def ones(*X):
    # check for class
    X, dt = _m_constructor_args(*X)
    return _marray.ones(X, dt)

@_ompc_base
def mcat(i):
    """Concatenate a list of matrices into a single matrix using separators
    ',' and ';'. The ',' means horizontal concatenation and the ';' means 
    vertical concatenation.
    """
    if i is None:
        return marray()
    # calculate the shape
    rows = [[]]
    final_rows = 0
    final_cols = 0
    crows = ccols = 0
    pos = []
    pos2 = []
    for x in i:
        #if x == ';':
        if x is Ellipsis:
            rows.append([])
            if final_cols > 0 and final_cols != ccols:
                error("Incompatible shapes!")
            else:
                final_cols = ccols
            final_rows += crows
            ccols = 0
            pos.append(Ellipsis)
        elif isinstance(x, mvar):
            shp = x.msize
            if len(shp) < 1: shp = [0]
            if len(shp) < 2: shp += [0]
            rows[-1].append(shp[0])
            pos.append( (slice(final_rows, final_rows+shp[0]), 
                         slice(ccols, ccols+shp[1])) )
            crows = shp[0]   # FIXME
            ccols += shp[1]
        elif _isscalar(x):
            rows[-1].append(1)
            pos.append( (final_rows, ccols) )
            crows = 1
            ccols += 1
        else:
            raise OMPCException("Unsupported type: %s!"%type(x))
    if final_cols > 0 and final_cols != ccols:
        error("Incompatible shapes!")
    else:
        final_cols = ccols
    final_rows += crows
    
    out = empty((final_rows, final_cols), 'double')
    for sl, x in _izip(pos, i):
        if x is not Ellipsis:
            if isinstance(x, _marray): x = x._a.T
            out._a.T.__setitem__(sl, x)
            #out._a.reshape(final_cols, final_rows).T.__setitem__(sl, x)
    return out

def who(*args,**kwargs):
    nargin, nargout = _get_narginout(0)
    import __main__
    ns = __main__.__dict__
    vars = [ x for x in ns \
                if isinstance(ns[x], mvar) and x[0] != '_' ]
    vars.sort()
    
    if nargout == 0:
        print 'Your variables are:'
        print '    '.join(vars)
    else:
        return mcellarray(vars)

@_ompc_base
def whos(*args,**kwargs):
    """Return list of variables in the current workspace."""
    nargin, nargout = _get_narginout(0)
    import __main__
    ns = __main__.__dict__
    vars = [ x for x in ns \
                if isinstance(ns[x], mvar) and x[0] != '_' ]
    vars.sort()
    
    if nargout == 0:
        cols = ['Name', 'Size', 'Bytes', 'Class', 'Attributes']
        print '  %10s  %15s  %15s  %10s  %10s  '%tuple(cols)
        for xname in vars:
            x = ns[xname]
            print '  %10s  %15r  %15r  %10s  '%(xname, x.msize, x._a.nbytes, x.dtype)
        print
    else:
        raise NotImplementedError()

@_ompc_base
def size(X):
    return X.msize

@_ompc_base
def rand(*args):
    if isinstance(args[0], str):
        raise NotImplemented
    if len(args) == 1:
        args = (args[0], args[0])
    return _marray('double', args, np.random.rand(*args[::-1]))

@_ompc_base
def randn(*args):
    if isinstance(args[0], str):
        raise NotImplemented
    if len(args) == 1:
        args = (args[0], args[0])
    return _marray('double', args, np.random.randn(*args[::-1]))

@_ompc_base
def reshape(A, *newsize):
    if len(newsize) == 0:
        raise OMPCError('??? Error using ==> reshape\n'
                        'Not enough input arguments.')
    if len(newsize) == 1 and hasattr(newsize, '__len__'):
        newsize = newsize[0]
    
    if not np.prod(A.msize) == np.prod(newsize):
        raise OMPCError('??? Error using ==> reshape\n'
                        'To RESHAPE the number of elements must not change.')
    out = A.__copy__()
    out.msize = newsize
    out._a = out._a.reshape(newsize[::-1])
    return out

@_ompc_base
def sum(A, *dimtype):
    restype = 'double'
    dim = 1
    if len(dimtype) == 2:
        dim = dimtype[0]
        dimtype = dimtype[1]
    elif len(dimtype) == 1:
        dimtype = dimtype[0]        
        if isinstance(dimtype, str):
            if dimtype == 'native':
                restype = A.dtype
            else:
                restype = dimtype
        else:
            dim = dimtype
    
    msize = A.msize
    if A.msize[dim-1] == 1:
        return A.__copy__()
    nshp = list(msize)
    nshp[dim-1] = 1
    if len(nshp) > 2 and nshp[-1] == 1: nshp = nshp[:-1]
    # use numpy's sum
    a = np.sum(A._a, len(msize)-dim)
    return _marray(A.dtype, nshp, a.reshape(nshp[::-1]))

@_ompc_base
def find(cond):
    a = mpl.find(cond._a.reshape(-1)) + 1
    msize = (len(a), 1)
    if len(cond.msize) == 2 and cond.msize[0] == 1:
        msize = (1, len(a))
    return _marray('double', msize, a.astype('f8').reshape(msize[::-1]))

try: _inv = np.oldnumeric.linear_algebra.inverse
except: _inv = np.linalg.inv
@_ompc_base
def inv(X):
    assert len(X.msize) == 2 and X.msize[0] == X.msize[1]
    return _marray('double', X.msize, _inv(X._a.T).T)

_eig = np.linalg.eig
@_ompc_base
def eig(X):
    assert len(X.msize) == 2 and X.msize[0] == X.msize[1]
    nargin, nargout = _get_narginout(1)
    [V, D] = _eig(X._a.T)
    if nargout == 1:
        return _marray('double', (len(V), 1), V.reshape(1, -1))
    elif nargout == 2:
        V = np.diag(V.reshape(-1))
        return _marray('double', D.shape[::-1], D.T), \
               _marray('double', V.shape, V)
               
    else:
        raise OMPCException('Too many output arguments.')

_svd = np.linalg.svd
@_ompc_base
def svd(X, *args):
    if len(args) > 0:
        raise NotImplementedError()
    assert len(X.msize) == 2 and X.msize[0] == X.msize[1]
    nargin, nargout = _get_narginout(1)
    [U, S, V] = _svd(X._a.T)
    # V is transposed already
    if nargout == 1:
        return _marray('double', (len(S), 1), S.reshape(1, -1))
    elif nargout == 3:
        S = np.diag(S.reshape(-1))
        return _marray('double', U.shape[::-1], U.T), \
               _marray('double', S.shape[::-1], S), \
               _marray('double', V.shape, V)
    else:
        raise OMPCException('Incorrect number of output arguments.')

@_ompc_base
def poly(X):
    na = np.poly(X._a.T)
    return _marray('double', (1, len(na)), na.reshape(-1, 1))

@_ompc_base
def roots(X):
    assert len(X.msize) == 2 and (X.msize[0] == 1 or X.msize[1] == 1)
    na = np.roots(X._a.reshape(-1))
    return _marray('double', (len(na), 1), na.reshape(1, -1))

@_ompc_base
def conv(X, Y):
    assert len(X.msize) == 2 and (X.msize[0] == 1 or X.msize[1] == 1)
    assert len(Y.msize) == 2 and (Y.msize[0] == 1 or Y.msize[1] == 1)
    na = np.convolve(X._a.reshape(-1), Y._a.reshape(-1))
    msize = (1, len(na))
    if Y.msize[1] == 1:
        msize = (len(na), 1)
    return _marray('double', msize, na.reshape(msize[::-1]))

@_ompc_base
def round(X):
    return _marray('double', X.msize, np.around(X._a))

@_ompc_base
def sqrt(X):
    if _isscalar(X):
        X = _marray('double', (1,1), [X])
    if np.any(X._a < 0):
        return _marray('complex', X.msize, np.sqrt(X._a.astype('complex128')))
    else:
        return _marray('double', X.msize, np.sqrt(X._a))

def magic(n):
    # from Octave's magic.m
    if n == 0:
        return marray([])
    elif mod (n, 2) == 1:
        shift = floor ((m_[0:n*n-1])/n)
        c = mod (m_[1:n*n] - shift + (n-3)/2, n)
        r = mod (m_[n*n:-1:1] + 2*shift, n)
        A(c*n+r+1).lvalue = m_[1:n*n]
        A = reshape(A, n, n);
#     elif mod(n, 4) == 0:
#         A = reshape(r_[1:n*n+1], n, n).T;
#         I = [1:4:n, 4:4:n];
#         I = r_[1:n+1:4, 4:n+1:4]
#         J = fliplr (I);
#         A(I,I) = A(J,J);
#         I = [2:4:n, 3:4:n];
#         J = fliplr (I);
#         A(I,I) = A(J,J);
#     elif mod(n, 4) == 2:
#         m = n/2;
#         A = magic (m);
#         A = [A, A+2*m*m; A+3*m*m, A+m*m];
#         k = (m-1)/2;
#         if (k>1)
#           I = 1:m;
#           J = [2:k, n-k+2:n];
#           A([I,I+m],J) = A([I+m,I],J);
#         endif
#         I = [1:k, k+2:m];
#         A([I,I+m],1) = A([I+m,I],1);
#         I = k + 1;
#         A([I,I+m],I) = A([I+m,I],I);
    return A

class mhandle(_marray):
    def __init__(self, arg):
        _marray.__init__(self, 'double', (1,1), [float(id(arg))])
        self._arg = arg

def _plot_args(*args):
    from sets import Set
    args = list(args)
    arrs = [[]]
    d = {}
    colorspec = False
    i = 0
    while len(args) > 0:
        arg = args.pop(0)
        if isinstance(arg, _marray):
            sz = arg.msize
            if len(sz) == 2 and sz[0] == 1:
                arrs[-1] += [ arg._a ]
            else:
                arrs[-1] += [ arg._a.T ]
        elif _isscalar(arg):
            arrs[-1] += [ _marray('double', (1,1), arg) ]
        elif isinstance(arg, str):
            if not colorspec:
                if len(arg) <= 3 and \
                    len(Set(arg).intersection(
                            'bgrcmykw'+'*-.:,o^v<>s+xDd1234hHp|_')) > 0:
                    if arg == '*': arg = '+'
                    arrs[-1] += [ arg ]
                    colorspec = True
                elif isinstance(arg, str):
                    if len(args) < 1:
                        raise OMPCException(
                            'Missing value for parameter "%s".'%arg)
                    val = args.pop(0)
                    d[arg] = val
            elif isinstance(arg, str):
                if len(args) < 1:
                    raise OMPCException(
                        'Missing value for parameter "%s".'%arg)
                val = args.pop(0)
                d[arg] = val
            else:
                args.insert(0, arg)
                colorspec = False
                arrs.extend([d, []])
                break
    if not isinstance(arrs[-1], dict): arrs.append(d)
    arrs = [ (a, d) for a, d in _izip(arrs[::2], arrs[1::2]) ]
    #print 'll', arrs
    return arrs

@_ompc_base
def set(x,*args):
    if not isinstance(x, mhandle):
        raise OMPCException("First argument must be an object handle!")
    o = x._arg
    args, kwargs = _plot_args_matlab(args, kwargs)
    assert len(args) == 0
    for k, v in kwargs.items():
        try: getattr(o,'set_%s'%k)(v)
        except: "Property '%s' not supported."%k

@_ompc_base
def xlabel(*args):
    # d = _plot_args(args[1:])
    return mhandle(mpl.xlabel(args[0]))

@_ompc_base
def ylabel(*args):
    # d = _plot_args(args[1:])
    return mhandle(mpl.ylabel(args[0]))

@_ompc_base
def zlabel(*args):
    # d = _plot_args(args[1:])
    return mhandle(mpl.zlabel(args[0]))

@_ompc_base
def plot(*args):
    d = _plot_args(*args)
    for x, kwargs in d:
        mpl.plot(*x, **kwargs)
    ha = mhandle(mpl.gca())
    mpl.draw()
    mpl.hold(False)
    mpl.show()
    return ha

@_ompc_base
def bar(*args):
    ha = mpl.gca()
    opts = None
    width = 0.8
    if isinstance(args[0], mhandle):
        ha = args[0]
        args = args[1:]
    if isinstance(args[-1], str):
        opts = args[-1]
        args = args[:-1]
    X, Y = None, None
    if len(args) == 1:
        Y = args[0]._a
        X = mslice[1:len(Y)]._a
    elif len(args) >= 2:
        X = args[0]._a
        Y = args[1]._a
        if len(args) == 3:
            width = args[2]._a
    X = X.reshape(-1)
    if len(Y.shape) == 2 and Y.shape[0] == 1: Y = Y.T
    res = mpl.bar(X, Y, width)
    ha = mhandle(res)
    mpl.hold(False)
    mpl.show()
    return ha

@_ompc_base
def axis(*args):
    nargin, nargout = _get_narginout(0)
    if nargout > 0:
        raise OMPCException('Too many output arguments.')
    ha = mpl.gca()
    if len(args) == 2:
        ha = args[0]._arg
        args = args[1:]
    if len(args) > 1:
        raise OMPCException('Too many input arguments.')
    x = args[0]
    if isinstance(x, str):
        try:
            mpl.axis(x)
        except:
            raise NotImplementedError()
    assert isinstance(x, _marray)
    mpl.axis(x._a.reshape(-1))

@_ompc_base
def grid(*args):
    b = None #trigger
    ha = mpl.gca()
    kwargs = {}
    if len(args) == 1:
        # isinstance(args[0], mstring) not necessary if mstring inherits str
        x = args[0].lower()
        if isinstance(x, str):
            if x == 'on': b = True
            elif x == 'off': b = False
            elif x == 'minor':
                # FIXME: issue a warning only
                raise NotImplementedError("Minor axis not ready yet!")
            else:
                raise OMPCException('Unknown option "%s"!'%x)
        elif isinstance(x, mhandle):
            ha = x
        else:
            raise OMPCException(
                    'First argument must be an axes handle or a string.')
    elif len(args) > 1:
        if not isinstance(args[0], mhandle):
            raise OMPCException('First argument must be an axes handle.')
        if len(args)%2 > 0:
            raise OMPCException('Name, value pairs expected after a handle.')
        kwargs = dict([(k, v) for k, v in _izip(args[1::2], args[2::2])])
        # FIXME: issue a warning only
        raise NotImplementedError("Setting parmeters not ready yet!")
    if not hasattr(ha, 'grid'):
        ha = ha.get_axes()
    ha.grid(b)
    mpl.draw()
