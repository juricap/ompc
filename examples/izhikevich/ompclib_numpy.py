
# This file is a part of OMPC (http://ompc.juricap.com/)
# 
# for testing:
#  import ompclib_numpy; reload(ompclib_numpy); from ompclib_numpy import *

# TODO
# - remove all references to array, use "ompc_base._init_data" instead

import sys

from itertools import izip as _izip, cycle as _cycle, repeat as _repeat
import numpy as np
import pylab as mpl

OMPCSEMI = Ellipsis
OMPCEND = None
end = OMPCEND

_dtype2numpy = {'double': 'f8', 'single': 'f4',
                'int32': 'i4', 'uint32': 'u4',
                'int16': 'i2', 'uint16': 'u2',
                'int8': 'i1', 'uint8': 'u1',
                'bool': 'bool',
               }

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

def _isscalar(A):
    if hasattr(A, '__len__') and len(A) > 1:
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

def _size(X, d=None):
    if isinstance(X, _marray):
        res = X.msize
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

def isempty(A):
    return np.prod(A.msize) == 0

def _dot(A, B):
    if not isinstance(A, _marray) or not isinstance(B, _marray):
        raise NotImplementedError("arguments must be 'marray's.")
    return np.dot(B.reshape(B.msize), A).T

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

def print_marray(A, ans=True):
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
        return str(A._a.T) + '\n\n'

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
        return _marray(self.dtype, self.msize, self._a+him)
    __radd__ = __add__
    
    def __sub__(self, him):
        if isinstance(him, _marray): him = him._a
        return _marray(self.dtype, self.msize, self._a-him)
    def __rsub__(self, him):
        if isinstance(him, _marray): him = him._a
        return _marray(self.dtype, self.msize, him-self._a)
    
    def __neg__(self):
        return _marray(self.dtype, self.msize, -self._a)
    
    # comparisons
    def __ge__(self, other):
        if isinstance(other, _marray):
            other = other._a
        return _marray('bool', self.msize, self._a >= other)
            
    
    # element access
    def __iter__(self):
        return iter(self._a)
    
    def __len__(self):
        return max(self.msize)
    
    def __getitem__(self, i):
        # determine the size of the new array
        nshp = _ndshape(self.msize, *i)
        return _marray(self.dtype, nshp, self._a.__getitem__(reversed(i)))
    
    # >> a = reshape(1:15,5,3)
    # >> a(eye(3)==1)
    # ans = [1, 5, 9]
    def __getitem1__(self, i):
        # determine the size of the new array
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
        na = self._a.__getitem__(ri)
        return _marray(self.dtype, nshp, na.reshape(nshp[::-1]))
    
    def __setitem__(self, i, val):
        if isinstance(val, _marray): val = val._a
        ins = list(_ndilin(self.msize, *ri))
        self._a.__setitem__(reversed(i), val)
    
    def __setitem1__(self, i, val):
        # determine the size of the new array
        nshp = _ndshape1(self.msize, *i)
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
        self._a.__setitem__(ri, val)
    
    # properties
    def transposed(self):
        assert len(self.msize) == 2
        return _marray(self.dtype, self.msize[::-1], 
                        self._a.T.flat.copy())
    T = property(transposed, None, None, "Transpose.")
    
    # IO
    def __str__(self):
        return print_marray(self)
    def __repr__(self):
        return "marray(%r, %r)"%(self.dtype, self.msize)

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
            self._a = np.array(list(self), dtype='f8')
    
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
        if isinstance(i, slice):
            raise NotImplemented
        retval = self.start + i*self.step
        if retval > self.stop:
            raise IndexError
        return retval
    
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

mslice = _mslice_helper()

class mstring(mvar):
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

def doublestr(x,prec=4):
    try:
        float(x)
    except:
        return x
    else:
        return str(round(x,4))

def _m_constructor_args(*X):
    from operator import isSequenceType
    dtype = 'double'
    if type(X[-1]) is str:
        dtype = X[-1]
        X = X[:-1]
    if len(X) == 1 and isSequenceType(X):
        X = X[0]
    return X, dtype

def empty(*X):
    # check for class
    X, dt = _m_constructor_args(*X)
    return _marray.empty(X, dt)

def zeros(*X):
    # check for class
    X, dt = _m_constructor_args(*X)
    return _marray.zeros(X, dt)

def ones(*X):
    # check for class
    X, dt = _m_constructor_args(*X)
    return _marray.ones(X, dt)

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
        else:
            shp = x.msize
            if len(shp) < 1: shp = [0]
            if len(shp) < 2: shp += [0]
            rows[-1].append(shp[0])
            pos.append( (slice(final_rows, final_rows+shp[0]), 
                         slice(ccols, ccols+shp[1])) )
            crows = shp[0]
            ccols += shp[1]
    if final_cols > 0 and final_cols != ccols:
        error("Incompatible shapes!")
    else:
        final_cols = ccols
    final_rows += crows
    
    out = empty((final_rows, final_cols), 'double')
    for sl, x in _izip(pos, i):
        if x is not Ellipsis:
            if isinstance(x, _marray): x = x._a.T
            out._a.reshape(final_cols, final_rows).T.__setitem__(sl, x)
    return out

def size(X):
    return X.msize

def rand(*args):
    if isinstance(args[0], str):
        raise NotImplemented
    if len(args) == 1:
        args = (args[0], args[0])
    return _marray('double', args, np.random.rand(*args[::-1]))

def randn(*args):
    if isinstance(args[0], str):
        raise NotImplemented
    if len(args) == 1:
        args = (args[0], args[0])
    return _marray('double', args, np.random.randn(*args[::-1]))

def reshape(A, *newsize):
    if len(newsize) == 0:
        raise OMPCError('??? Error using ==> reshape\n'
                        'Not enough input arguments.')
    if len(newsize) == 1 and hasattr(newsize, '__len__'):
        newsize = newsize[0]
    out = A.__copy__()
    if not np.prod(A.msize) == np.prod(newsize):
        raise OMPCError('??? Error using ==> reshape\n'
                        'To RESHAPE the number of elements must not change.')
    out.msize = newsize
    return out

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
    return _marray(A.dtype, nshp, a)

def find(cond):
    a = mpl.find(cond._a.reshape(-1)) + 1
    msize = (len(a), 1)
    if len(cond.msize) == 2 and cond.msize[0] == 1:
        msize = (1, len(a))
    return _marray('double', msize, a.astype('f8').reshape(msize[::-1]))

def plot(*args):
    #print [ x.msize for x in args ]
    nargs = []
    for x in args:
        if isinstance(x, _marray): nargs.append(x._a.T)
        elif isinstance(x, mstring): nargs.append(str(x))
        else: nargs.append(x)
    mpl.plot(*nargs)
    mpl.show()
