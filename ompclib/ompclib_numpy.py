
# This file is a part of OMPC (http://ompc.juricap.com/)
# 
# for testing:
#  import ompclib_numpy; reload(ompclib_numpy); from ompclib_numpy import *

# TODO
# - remove all references to array, use "ompc_base._init_data" instead

import sys, os; sys.path.append(os.path.abspath('..'))

from itertools import izip as _izip, cycle as _cycle, repeat as _repeat
from ompc import _get_narginout
import os, sys
import numpy as np
import pylab as mpl

# Functions that are to be exported have to be listed in the __ompc_all__
# array.
# This decorator adds a function to the "toolbox-less" OMPC base library.
__ompc_all__ = ['end', 'mslice', 'mstring', 'OMPCSEMI',
                'OMPCException', 'elmul', 'elpow', 'eldiv', 'ldiv', 'elldiv']

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

_numpy2dtype = {}
for k, v in _dtype2numpy.items():
    _numpy2dtype[np.dtype(v)] = k
    _numpy2dtype[str(np.dtype(v))] = k
    _numpy2dtype[v] = k

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
    
    def __base0__(self, shp=None):
        raise OMPCException(
                'Class "%s" cannot be used as index!'%self.__class__)
    
    # FIXME: warn people about using numpy functions directly
    def __array__(self):
        print 'in __array__', repr(self)
        print 'in __array__', self
        raise NotImplementedError("At the moment using numpy functions " \
              "directly is not possible! Please read the documentation at " \
              "http://ompc.juricap.com/documentation/.")
    
    def __nonzero__(self):
        return bool(np.any(self._a != 0))

class _mview(mvar):
    def __init__(self, viewed, ins, linear):
        self.viewed = viewed
        self.ins = ins
        self.linear = linear
    def __repr__(self):
        return "_mview(%r, %r, %r)"%(self.viewed, self.ins, self.linear)
    def __str__(self):
        return "<view of %r>"%(self.viewed)

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

###################### base mfunctions

@_ompc_base
def plus(A, B):
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    na = A+B
    return _marray(_numpy2dtype[na.dtype], na.shape[::-1], na)

@_ompc_base
def minus(A, B):
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    na = A-B
    return _marray(_numpy2dtype[na.dtype], na.shape[::-1], na)

@_ompc_base
def uminus(A):
    if isinstance(A, mvar): A = A._a
    na = -A
    return _marray(_numpy2dtype[na.dtype], na.shape[::-1], na)

@_ompc_base
def times(A, B):
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    na = A*B
    return _marray(_numpy2dtype[na.dtype], na.shape[::-1], na)

@_ompc_base
def mtimes(A, B):
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    # the arrays are stored transposed
    na = np.dot(B, A)
    return _marray(_numpy2dtype[na.dtype], na.shape[::-1], na)

@_ompc_base
def power(A, B):
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    na = A**B
    return _marray(_numpy2dtype[na.dtype], na.shape[::-1], na)

try:
    from numpy.linalg import matrix_power
except:
    def matrix_power(M,n):
        if len(M.shape) != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("input must be a square array")
        if not issubdtype(type(n),int):
            raise TypeError("exponent must be an integer")

        from numpy.linalg import inv

        if n==0:
            M = M.copy()
            M[:] = np.identity(M.shape[0])
            return M
        elif n<0:
            M = inv(M)
            n *= -1

        result = M
        if n <= 3:
            for _ in range(n-1):
                result = np.dot(result, M)
            return result

        # binary decomposition to reduce the number of Matrix
        # multiplications for n > 3.
        beta = np.binary_repr(n)
        Z, q, t = M, 0, len(beta)
        while beta[t-q-1] == '0':
            Z = np.dot(Z,Z)
            q += 1
        
        result = Z
        for k in range(q+1,t):
            Z = np.dot(Z,Z)
            if beta[t-k-1] == '1':
                result = np.dot(result,Z)
        return result

@_ompc_base
def mpower(A, B):
    if len(A.msize) != 2:
        raise OMPCException('??? Error using ==> mpower\n'
                            'marray must be 2-D')
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    if isinstance(B, float):
        if np.around(him) != him: raise NotImplementedError()
        else: B = int(B)
    na = matrix_power(A.T, B)
    return _marray(_numpy2dtype[na.dtype], na.shape[::-1], na)

_solve = np.linalg.solve
@_ompc_base
def mldivide(A, B):
    # FIXME A, B have to be matrices
    if A.msize[0] == A.msize[1]:
        if isinstance(A, mvar): A = A._a
        if isinstance(B, mvar): B = B._a
        na = _solve(A, B)
        msize = na.shape[::-1]
        if len(msize) == 1: msize = (msize[0], 1)
        return _marray(_numpy2dtype(na.dtype), msize, na.T)
    else:
        raise NotImplementedError()
    raise NotImplementedError()

@_ompc_base
def mrdivide(A, B):
    "A/B = (B.T\A.T).T"
    return mldivide(B.T, A.T).T
    # raise NotImplementedError()

@_ompc_base
def ldivide(A, B):
    return rdivide(B, A)

@_ompc_base
def rdivide(A, B):
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    na = A / B
    return _marray(_numpy2dtype[na.dtype], na.shape[::-1], na)

@_ompc_base
def eq(A, B):
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    na = A == B
    return _marray('bool', na.shape[::-1], na)

@_ompc_base
def ne(A, B):
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    na = A != B
    return _marray('bool', na.shape[::-1], na)

@_ompc_base
def lt(A, B):
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    na = A < B
    return _marray('bool', na.shape[::-1], na)

@_ompc_base
def gt(A, B):
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    na = A > B
    return _marray('bool', na.shape[::-1], na)

@_ompc_base
def le(A, B):
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    na = A <= B
    return _marray('bool', na.shape[::-1], na)

@_ompc_base
def ge(A, B):
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    na = A >= B
    return _marray('bool', na.shape[::-1], na)

@_ompc_base
def and_(A, B):
    '''Element-wise logical AND.'''
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    na = np.logical_and(A, B)
    return _marray('bool', na.shape[::-1], na)

@_ompc_base
def or_(A, B):
    '''Element-wise logical OR.'''
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    na = np.logical_or(A, B)
    return _marray('bool', na.shape[::-1], na)

@_ompc_base
def not_(A):
    '''Logical NOT.'''
    if isinstance(A, mvar): A = A._a
    na = np.logical_not(A)
    return _marray('bool', na.shape[::-1], na)

@_ompc_base
def xor_(A, B):
    '''Logical XOR.'''
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    na = np.logical_xor(A, B)
    return _marray('bool', na.shape[::-1], na)

@_ompc_base
def any(A):
    '''True if any element of vector is nonzero'''
    if isinstance(A, mvar): A = A._a
    return bool(np.any(A))

@_ompc_base
def all(A):
    '''True if all elements of vector is nonzero'''
    if isinstance(A, mvar): A = A._a
    return bool(np.all(A))

def transpose(A):
    '''Transpose.'''
    if len(A.msize) != 2:
        raise OMPCException('Transpose on ND array is not defined.')
    return _marray(A.dtype, A.msize[::-1], A._a.T.copy())

def ctranspose(A):
    '''Complex conjugate transpose.'''
    if len(A.msize) != 2:
        raise OMPCException('Transpose on ND array is not defined.')
    return _marray(A.dtype, A.msize[::-1], A._a.conj().T.copy())

def horzcat(*X):
    '''Horizontal concatenation.'''
    # our _a member is transposed do vertcat
    X = [ isinstance(x, mvar) and x or x._a for x in X ]
    na = np.vstack(X)
    return _marray(_numpy2dtype[na.dtype], na.shape[::-1], na)

def vertcat(*X):
    '''Vertical concatenation.'''
    # our _a member is transposed do horztcat
    X = [ isinstance(x, mvar) and x or x._a for x in X ]
    na = np.hstack(X)
    return _marray(_numpy2dtype[na.dtype], na.shape[::-1], na)

# FIXME bit operations


def union(A, B, flag=None):
    '''Set union.'''
    # FIXME support nargout = 3
    if isinstance(A, mvar): A = A._a
    if isinstance(B, mvar): B = B._a
    if flag.lower() == 'rows':
        if A.ndim != 2 or B.ndim != 2:
            raise OMPCException('A nd B must 2D matrices!')
        if A.shape[0] != B.shape[0]:
            raise OMPCException('A nd B must have same number of columns!')
        # FIXME: slow and memory hungry
        sA = set( tuple(x) for x in A )
        na = np.array([ x for x in sA.union( tuple(x) for x in b._a.T ) ]).T
    else:
        # must be vectors
        if A.shape[0] == 1 or A.shape[1] == 1: A = A.reshape(-1)
        else: raise OMPCException('A nd B must vectors or 2D matrices!')
        if B.shape[0] == 1 or B.shape[1] == 1: B = B.reshape(-1)
        else: raise OMPCException('A nd B must vectors or 2D matrices!')
        na = np.union1d(A, B).reshape(-1,1)
    return _marray(_numpy2dtype[na.dtype], na.shape[::-1], na)

def unique(A, B):
    '''Set unique.'''
    raise NotImplementedError()

def intersect(A, B):
    '''Set intersection.'''
    raise NotImplementedError()

def setdiff(A, B):
    '''Set difference.'''
    raise NotImplementedError()

def setxor(A, B):
    '''Set exclusive-or.'''
    raise NotImplementedError()

def ismember(A, B):
    '''True for set member.'''
    raise NotImplementedError()

############ operators

class _el(object):
    def __new__(cls, left=None, right=None):
        if left is None or right is None:
            nel = super(_el, cls).__new__(_el)
            nel.__class__ = cls
            nel.left = left
            nel.right = right
            return nel
        else:
            return cls.op(left, right)

def make_operator(name, method):
    class _op(_el):
        op = staticmethod(method)
    def op(self, right):
        if self.left is None: return self.__class__(right=right)
        return self.op(self.left, right)
    def rop(self, left):
        if self.right is None: return self.__class__(left=left)
        return self.op(left, self.right)
    _op.__name__ = '_el%s'%name
    setattr(_op, '__%s__'%name, op)
    setattr(_op, '__r%s__'%name, rop)
    return _op()

elmul = make_operator('mul', times)
elpow = make_operator('pow', power)
eldiv = make_operator('div', rdivide)
ldiv = make_operator('div', mldivide)
elldiv = make_operator('div', ldivide)

############ support functions

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
        pre = '\nans = \n'
    if isempty(A):
        pre += '\n  []'
    if len(A.msize) > 2:
        for i in _ndi(*[slice(0,x) for x in A.msize[2:]]):
            pre += '\n(:, :, %s)\n'%', '.join([str(x+1) for x in i])
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
        res = pre + '\n  ' + \
            '\n  '.join(', '.join(map(nstr,x)) for x in rows)
        if ans: res += '\n\n'
        else: res += '\n'
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
        return power(self, him)
    def __pow__(self, right):
        # if multiplying with _el object, call the elementwise operation
        if isinstance(right, _el): return right.__class__(self, right.right)
        elif _isscalar(right): return power(self, right)
        return mpower(self, right)
    def __rpow__(self, left):
        # if multiplying with _el object, call the elementwise operation
        if isinstance(left, _el): return left.__class__(left.left, self)
        elif _isscalar(left): return power(left, self)
        return mpower(left, self)
    
    def __elmul__(self, him):
        return times(self, him)
    def __mul__(self, right):
        if isinstance(right, _el): return right.__class__(self, right.right)
        elif _isscalar(right): return times(self, right)
        return mtimes(self, right)
    def __rmul__(self, left):
        if isinstance(left, _el): return left.__class__(left.left, self)
        elif _isscalar(left): return times(left, self)
        return mtimes(left, self)
    
    def __eldiv__(self, him):
        return rdivide(self, him)
    def __div__(self, right):
        # if multiplying with _el object, call the elementwise operation
        if isinstance(right, _el): right.__class__(self, right.right)
        elif _isscalar(right): return rdivide(self, right)
        return mrdivide(self, right)
    def __rdiv__(self, left):
        # if multiplying with _el object, call the elementwise operation
        if isinstance(left, _el): return left.__class__(left.left, self)
        elif _isscalar(left): return rdivide(left, self)
        return mrdivide(left, self)
    
    def __add__(self, him): return plus(self, him)
    def __radd__(self, him): return plus(him, self)
    
    def __sub__(self, him): return minus(self, him)
    def __rsub__(self, him): return minus(him, self)
    
    def __neg__(self): return uminus(self)
    
    # comparisons
    def __ge__(self, other): return ge(self, other)

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
        #return iter(self._a.flat)#(_marray(self.dtype, (1,1), x) for x in self._a.flat )
        return ( float(x) for x in self._a.flat )
        #(_marray(self.dtype, (1,1), x) for x in self._a.flat )
    
    def __len__(self):
        return max(self.msize)
    
    def __base0__(self, shp=None):
        # FIXME: issue a warning on non-integers
        if self.dtype == 'bool':
            return self._a
        ind = (self._a - 1).T.astype('i4')
        if ind.ndim == 2 and ind.shape[0] == 1 or ind.shape[1] == 1:
            ind = ind.reshape(-1)
        # if ind.ndim == 2 and ind.shape[0] == 1:
        #     ind = ind[0]
        return ind
    
    def __getitem__(self, i):
        # determine the size of the new array
        if not hasattr(i, '__len__'): i = [i]
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
            i = i[0]
            if self.msize[0] == 1: ri = (i.__base0__(self.msize[1]), 0)
            elif self.msize[1] == 1: ri = (0, i.__base0__(self.msize[0]))
            else:
                # access to a flat array
                msize = _size(i)
                if isinstance(i, mvar): i = i.__base0__(len(self._a.flat))
                na = self._a.flat[i]
                return _marray(self.dtype, msize, na.reshape(msize[::-1]))
        else:
            di = len(self.msize)-1
            for x in reversed(i):
                if isinstance(x, mvar): ri.append(x.__base0__(self.msize[di]))
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
            i = i[0]
            if self.msize[0] == 1:
                ri = (i.__base0__(self.msize[1]), 0)
                val = val[0]
            elif self.msize[1] == 1:
                ri = (0, i.__base0__(self.msize[0]))
                val = val[0]
            else:
                # access to a flat array
                msize = _size(i)
                if isinstance(i, mvar): i = i.__base0__(len(self._a.flat))
                self._a.flat[i] = val
                return
        else:
            di = len(self.msize)-1
            for x in reversed(i):
                if isinstance(x, mvar): ri.append(x.__base0__(self.msize[di]))
                else: ri.append(x-1)
                di -= 1
        self._a.__setitem__(tuple(ri), val)
    
    # properties
    def transposed(self):
        return transpose(self)
    def ctransposed(self):
        return ctranspose(self)
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
        if self.step < 0:
            while value >= self.stop:
                yield float(value)
                value += self.step
        else:
            while value <= self.stop:
                yield float(value)
                value += self.step
    
    def __getitem__(self, i):
        val = self.start + self.step*i
        if val > self.stop:
            raise OMPCException('Index exceeds matrix dimensions!')
        return float(val)
#         self.init_data()
#         na = self._a.__getitem__(i)
#         return _marray('double', na.shape[::-1], na.reshape(na.shape[::-1]))
    
    def __getitem1__(self, i):
        val = self.start + self.step*(i-1)
        if val > self.stop:
            raise OMPCException('Index exceeds matrix dimensions!')
        return float(val)
#         self.init_data()
#         return _marray('double', self.msize, self._a).__getitem1__(i)
    
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
        # there are all 3 arguments, stop is actually i.step
        elif i.stop < 0:
            stop = i.step
            step = i.stop
        else:
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
    if len(X) == 1:# and isSequenceType(X):
        X = X[0]
        #X = X[0], X[0]
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
def eye(*X):
    if len(X) == 0:
        return _marray.ones((1,1), 'double')
    # check for class
    X, dt = _m_constructor_args(*X)
    kw = dict(dtype=_dtype2numpy[dt])
    if not hasattr(X, '__len__'): X = (X,)
    na = np.eye(*X[::-1], **kw)
    return _marray(dt, na.shape[::-1], na)

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
            if isinstance(x, mvar): x = x._a
            out._a.__setitem__(sl[::-1], x)
            #out._a.reshape(final_cols, final_rows).T.__setitem__(sl, x)
    return out

def who(*args,**kwargs):
    nargin, nargout = _get_narginout(0)
    import __main__
    ns = __main__.__dict__
    vars = [ x for x in ns \
                if isinstance(ns[x], mvar) and x[0] != '_' ]
    if args:
        vars = [ x for x in vars if x in args ]
    vars.sort()
    
    if nargout == 0:
        print 'Your variables are:'
        print '    '.join(vars)
    else:
        return mcellarray(vars)

@_ompc_base
def whos(*args, **kwargs):
    """Return list of variables in the current workspace."""
    nargin, nargout = _get_narginout(0)
    import __main__
    ns = __main__.__dict__
    vars = [ x for x in ns \
                if isinstance(ns[x], mvar) and x[0] != '_' ]
    if args:
        vars = [ x for x in vars if x in args ]
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
        raise OMPCException('??? Error using ==> reshape\n'
                        'Not enough input arguments.')
    if len(newsize) == 1 and hasattr(newsize, '__len__'):
        newsize = newsize[0]
    
    if not np.prod(A.msize) == np.prod(newsize):
        raise OMPCException('??? Error using ==> reshape\n'
                        'To RESHAPE the number of elements must not change.')
    out = A.__copy__()
    out.msize = newsize
    out._a = out._a.reshape(newsize[::-1])
    return out

@_ompc_base
def fliplr(X):
    if X._a.ndim != 2:
        error('X must be a 2-D matrix.')
    return _marray(X.dtype, X.msize, np.flipud(X._a))

@_ompc_base
def flipud(X):
    if X._a.ndim != 2:
        error('X must be a 2-D matrix.')
    return _marray(X.dtype, X.msize, np.fliplr(X._a))

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
def floor(X):
    return _marray('double', X.msize, np.floor(X._a))

@_ompc_base
def ceil(X):
    return _marray('double', X.msize, np.ceil(X._a))

@_ompc_base
def fix(X):
    return _marray('double', X.msize, np.fix(X._a))

def _what(X):
    if isinstance(X, mvar):
        return X.dtype, X.msize
    elif isinstance(X, int):
        return 'int32', (1, 1)
    elif isinstance(X, float):
        return 'double', (1, 1)
    else:
        raise NotImplementedError()

@_ompc_base
def mod(X, i):
    dtype, msize = _what(X)
    if isinstance(X, mvar): X = X._a
    if isinstance(i, mvar):
        if i.msize != msize:
            raise OMPCException("Matrix dimensions must agree!")
        i = i._a
    if i == 0:
        return _marray(dtype, msize, X)
    elif np.all(X == i):
        return zeros(msize, dtype)
    na = np.mod(X, i)
    return _marray(_numpy2dtype[na.dtype], msize, na)

@_ompc_base
def sqrt(X):
    if _isscalar(X):
        X = _marray('double', (1,1), [X])
    if np.any(X._a < 0):
        return _marray('complex', X.msize, np.sqrt(X._a.astype('complex128')))
    else:
        return _marray('double', X.msize, np.sqrt(X._a))

@_ompc_base
def magic(n):
    # from Octave's magic.m
    A = empty((n, n), 'double')
    if n == 0:
        return marray([])
    elif mod (n, 2) == 1:
        n = 3
        shift = floor ((mslice[0:n*n-1])/n)
        c = mod(mslice[1:n*n] - shift + (n-3)/2, n)
        r = mod(mslice[n*n:-1:1] + 2*shift, n)
        A(c*n+r+1).lvalue = mslice[1:n*n]
        A = reshape(A, n, n);
    elif mod(n, 4) == 0:
        A = reshape(mslice[1:n*n], n, n).cT;
        I = mcat([mslice[1:4:n], mslice[4:4:n]])
        J = fliplr(I);
        A(I,I).lvalue = A(J,J)
        I = mcat([mslice[2:4:n], mslice[3:4:n]]);
        J = fliplr(I);
        A(I,I).lvalue = A(J,J);
    elif mod(n, 4) == 2:
        m = n/2
        A = magic(m)
        A = mcat([A, A+2*m*m, OMPCSEMI, A+3*m*m, A+m*m])
        k = (m-1)/2
        if k > 1:
            I = mslice[1:m]
            J = mcat([mslice[2:k], mslice[n-k+2:n]])
            A([I,I+m],J).lvalue = A([I+m,I],J)
        I = mcat([mslice[1:k], mslic[k+2:m]])
        A([I,I+m],1).lvalue = A([I+m,I],1);
        I = k + 1
        A([I,I+m],I).lvalue = A([I+m,I],I)
    return A

from os.path import normpath as _normpath
import scipy.io
@_ompc_base
def load(*X):
    X = list(X)
    format = None
    re = []
    vars = []
    if X[0].strip()[0] == '-':
        op = X.pop(0).strip()
        if op.lower() == '-ascii': format = 'a'
        elif op.lower() == '-mat': format = 'm'
        else: raise OMPCException('Unknown option "%s".'%op)
    # next must be filename
    fname = X.pop(0)
    base, ext = os.path.splitext(fname)
    if not ext:
        if os.path.exists(fname):
            format = 'a'
        else:
            ext = '.mat'
            fname += ext
            format = 'm'
    elif ext == '.mat':
        format = 'm'
    if not os.path.exists(fname):
        raise OMPCException('Cannot find file "%s"!'%fname)
    # variables
    if len(X) > 0:
        if X[0].strip()[0] == '-':
            # regexp
            op = X.pop(0).strip().lower()
            if not op == '-regexp':
                raise OMPCException('Unknown option "%s".'%op)
            re = X
        else:
            vars = X
    fname = _normpath(fname)
    # load
    if format == 'm':
        # scipy makes imports really slow
        _loadmat = scipy.io.loadmat
        try: d = _loadmat(fname, matlab_compatible=True)
        except: raise OMPCException('Cannot open "%s" as an M-file!!'%fname)
        data = []
        if vars:
            data = [ (k, v) for k, v in d.items() if k in vars ]
        elif re:
            raise NotImplementedError()
        else:
            data = [ (k, v) for k, v in d.items() if k[:2] != '__' ]
        # populate the workspace
        import inspect
        cf = inspect.currentframe()
        for var, val in data:
            na = np.asfortranarray(val).T
            cf.f_back.f_globals[var] = \
                    _marray(_numpy2dtype[str(na.dtype)], na.shape[::-1], na)
    else:
        # ASCII
        try: f = file(fname, 'rU')
        except: raise OMPCException('Cannot open "%s"!'%fname)
        data = []
        for x in f:
            x = x.strip()
            if x.startswith('%'): continue
            data += [ map(float, x.split()) ]
        na = np.asfortranarray(data, 'f8').T
        import inspect
        cf = inspect.currentframe()
        base = os.path.basename(base)
        cf.f_back.f_globals[base] = _marray('double', na.shape[::-1], na)

# _ompc_base
# def save(*X):
#     import inspect
#     f = inspect.currentframe()
#     d = {}
#     for var in args:
#         d[var] = f.f_back.f_globals[var]
#     _savemat(fname, d)

@_ompc_base
def length(X):
    return len(X)

_fft = np.fft.fft
@_ompc_base
def fft(X,N=mcat([]),axis=None):
    if axis is not None: axis = len(X.msize) - i - 1
    if len(X.msize) == 2:
        if X.msize[0] == 1:
            X = X._a.reshape(-1)
        else:
            if axis is None: axis = len(X.msize)-1
            X = X._a
    elif len(X.msize) > 2:
        if axis is None:
            # first non-singleton dimension
            for i in xrange(len(X.msize),-1,-1):
                if X.msize[i] > 1: break
            axis = i
    else:
        raise NotImplementedError("Less than 2D?")
    # N
    if isempty(N): N = X.shape[axis]
    # do it
    na = _fft(X, N, axis)
    msize = na.shape[::-1]
    if len(msize) < 2: msize = (msize, 1)
    return _marray(_numpy2dtype[na.dtype], msize, na)

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

@_ompc_base
def title(*args):
    mpl.title(args)

@_ompc_base
def legend(*args):
    mpl.legend(args)
