
# This file is a part of OMPC (http://ompc.juricap.com/)
# 
# for testing:
#  import ompclib_demo; reload(ompclib_demo); from ompclib_demo import *

# TODO
# - remove all references to array, use "ompc_base._init_data" instead

import sys
from __builtin__ import sum as _sum

from itertools import izip as _izip, cycle as _cycle, repeat as _repeat
from array import array as _array
from math import floor as _floor

OMPCSEMI = Ellipsis

_dsize_dict = {'d': 8, 'f8': 8, 'double': 8,
               'f': 4, 'f4': 4, 'sing;e': 4,
               'i': 4, 'i4': 4, 'int32': 4,
              }

_dtype2array = {'double': 'd', 'single': 's',
                'int32': 'l', 'uint32': 'L',
                'int16': 'i', 'uint16': 'I',
                'int8': 'b', 'uint8': 'B',
                'bool': 'B',
               }

from ctypes import sizeof, c_double, c_float, \
                   c_int, c_uint, c_short, c_ushort, \
                   c_byte, c_ubyte

_dtype2ctype = {'double': c_double, 'single': c_float,
                'int32': c_int, 'uint32': c_uint,
                'int16': c_short, 'uint16': c_ushort,
                'int8': c_byte, 'uint8': c_ubyte,
                'bool': c_byte,
               }

class mvar(object):
    @staticmethod
    def _DataObject(dtype, data):
        return _array(_dtype2array[dtype], data)
    
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
        if self._a is None: return None
        carr_type = (_dtype2ctype[self.dtype]*len(self._a))
        data = carr_type.from_address(self._a.buffer_info()[0])
        # to prevent the aray from being deleted before the ctypes wrapper
        data.refmarray = self
        return data
    ctypes = property(_ctypes_get, None, None, 
                      "Ctypes-wrapped data object.")
    def _lvalue_set(self, val):
        assert hasattr(self, '__ompc_view__')
        o = self.__ompc_view__
        # FIXME: o.linear
        o.viewed.__setitem1__(o.ins, val)
    lvalue = property(None, _lvalue_set, None, "")
#     def __str__(self):
#         if hasattr(self, '__ompc_view__'):
#             return repr(self.__ompc_view__)
    def __copy__(self):
        a = _marray(self.dtype, self.msize)
        a._a[:] = self._a
        return a
    def __deepcopy__(self):
        a = _marray(self.dtype, self.msize)
        a._a[:] = self._a
        return a

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
    from operator import isSequenceType
    if not isSequenceType(seq): return seq
    while isSequenceType(seq):
        if type(item) in (TupleType, ListType):
            res.extend(flatten(item))
        else:
            res.append(item)
    return res

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

def _ndilin(shp, *i):
    """Generator of linear indices into an array of shape `shp`. Indices are
    specified by slices of indices in `i`."""
    cp = [1]
    for x in shp[:-1]:
        cp += [cp[-1]*x]
    i = list(i)
    for j, x in enumerate(i):
        if isinstance(x, slice):
            start, stop, step = x.start, x.stop, x.step
            if x.start is None: start = 0
            if x.stop == sys.maxint or x.stop is None: stop = shp[j]
            if x.step is None: step = 1
            i[j] = slice(start, stop, step)
    res = []
    for x in _ndi(*i):
        res.append(int(_sum( x*y for x, y in _izip(cp, x) )))
        #yield int(_sum( x*y for x, y in _izip(cp, x) ))
    return res

def _ndi1(*i):
    """Returns a generator of tuples that iterate over elements specified
    by slices and indices in `i`.
    The index base of input parameters is 1."""
    from itertools import chain, repeat, cycle, izip
    r = lambda x: _marray.frommslice(x)
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

def _ndilin1(shp, *i):
    """Generator of linear indices into an array of shape `shp`. Indices are
    specified by slices of indices in `i`.
    The input index is base 1 the linear indices returned are base 0.
    This function produces indices, threfore the output type is 'int'."""
    cp = [1]
    for x in shp[:-1]:
        cp += [cp[-1]*x]
    i = list(i)
    for j, x in enumerate(i):
        if isinstance(x, _mslice) and x.hasnoend():
            i[j] = x.evaluate_end(shp[j])
    for x in _ndi1(*i):
        yield int(_sum( x*(y-1) for x, y in _izip(cp, x) ))

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
    """Determine the shape of a view on A with slicing specified in `i`.
    """
    shp = []
    for idim, x in enumerate(i):
        if isinstance(x, _mslice):
            if x.hasnoend():
                shp.append( len(mslice[x.start:x.step:msize[idim]]) )
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
    if len(shp) == 1: shp[:0] = [1]
    return shp

def dot(A, B):
    """Dot product."""
    M, L = A.msize
    N = B.msize[1]
    MN = M*N
    na = _marray(_typegreater(A.dtype, B.dtype), (M, N))
    n = na._a; a = A._a; b = B._a
    i = 0
    for c in xrange(N):
        for r in xrange(M):
            res = 0.0
            bi = c*L
            for ai in xrange(r, M*L, M):
                res += a[ai]*b[bi]
                bi += 1
            n[i] = res
            i += 1
    return na

def find(cond):
    """Return linear index of elements where the condition is True.
    The index is based on 1."""
    res = []
    for i, x in enumerate(cond):
        if bool(x): res.append(i+1)
    shp = _size(cond)
    if shp[0] > 1:
        if shp[1] > 1: shp = (len(res), 1)
        else: shp = (len(res), 1)
    elif shp[1] > 1:
        shp = (1, len(res))
    else:
        shp = (len(res), 1)
    na = _marray('double', shp,
                 _array(_dtype2array['double'], res))
    return na

def isempty(A):
    return _prod(A.msize) == 0

def _dot(A, B):
    """Dot product."""
    M, N = A.msize[0], B.msize[1]
    MN = M*N
    na = _marray(_typegreater(A.dtype, B.dtype), (M, N))
    cols = (_islice(B._a, i*M, (i+1)*M) for i in _cycle(xrange(N)))
    rows = _cycle( _islice(A._a, i, MN+i, M) for i in xrange(M) )
    # fill in the result in the FORTRAN order
    for i in xrange(MN):
        col = cols.next()
        for j in xrange(N):
            s = sum( a*b for a, b in _izip(rows.next(), col) )
            na._a[i] = s
    return na

def _prod(X):
    from operator import isSequenceType, mul
    if not isSequenceType(X):
        return X
    return reduce(mul, X, 1)

def _squeeze(A):
    res = A.__copy__()
    res.msize = [ x for x in res.msize if x > 1 ]
    return res

def print_marray(A, ans=True):
    pre = ''
    if ans:
        pre = '\nans = \n\n'
    if len(A.msize) > 2:
        for i in _ndi(*[slice(0,x) for x in A.msize[2:]]):
            pre += '(:, :, %s)\n\n'%', '.join([str(x+1) for x in i])
            cur = (slice(0,A.msize[0]), slice(0, A.msize[1])) + i
            sA = A.__getitem__(cur)
            #print A.msize, sA.msize
            sA.msize = A.msize[:2]
            pre += print_marray(sA, False)
        return pre
    else:
        M, N = A.msize
        if N < 10: srow = lambda i: A.row(i)
        else: srow = lambda i: list(A.row(i, [0, 1, 2])) + \
                               ['...'] + \
                               list(A.row(i, [N-3, N-2, N-1]))
        if M < 10: rows = ( srow(i) for i in xrange(M) )
        else: rows = [ srow(i) for i in xrange(3) ] + \
                     [('...',)] + \
                     [ srow(i) for i in [M-3,M-2,M-1] ]
        return pre + ' [' + \
            '\n  '.join(', '.join(map(doublestr,x)) for x in rows) + ']\n\n'

class _marray(mvar):
    
    @staticmethod
    def empty(shp, dtype):
        return _marray(dtype, shp)
    
    @staticmethod
    def zeros(shp, dtype):
        return _marray(dtype, shp)
    
    @staticmethod
    def ones(shp, dtype):
        from array import array
        a = _marray(dtype, shp)
        a._a[:] = array(_dtype2array[dtype], [1]*_prod(shp))
        return a
    
    @staticmethod
    def frommslice(i):
        # the slice is in terpreted in the m-way start:step:stop
        if i.step is None:
            data = list(_mrange(i.start, i.stop))
        else:
            data = list(_mrange(i.start, i.stop, i.step))
        return _marray('double', (1, len(data)), data)
    
    def __init__(self, dtype, msize, a=None):
        from array import array
        from operator import isSequenceType
        if not isSequenceType(msize):
            msize = (msize, msize)
        elif len(msize) == 1:
            msize = (1, 1)
        if a is None:
            self._a = array(_dtype2array[dtype], 
                            '\x00'*(_dsize(dtype)*_prod(msize)))
        elif isinstance(a, array):
            self._a = a
        else:
            self._a = _array(_dtype2array[dtype], _flatten(a))
        self.msize = msize
        self.dtype = dtype
    
    def __copy__(self):
        a = _marray(self.dtype, self.msize)
        a._a[:] = self._a
        return a
    def __deepcopy__(self):
        a = _marray(self.dtype, self.msize)
        a._a[:] = self._a
        return a
    
    # operators
    def __elpow__(self, him):
        from array import array
        na = _marray(self.dtype, self.msize, 
                    array(_dtype2array[self.dtype], (x**him for x in self._a)))
        return na
    
    def __elmul__(self, him):
        from array import array
        if _isscalar(him):
            if hasattr(him, '__len__'): him = _cycle(him)
            else: him = _repeat(him)
        na = _marray(self.dtype, self.msize, 
                     array(_dtype2array[self.dtype], 
                           (x*y for x, y in _izip(self._a, him))))
        return na
    
    def __mul__(self, right):
        # if multiplying with _el object, call the elementwise operation
        if isinstance(right, _el): return _el(left=self)
        elif _isscalar(right): return self.__elmul__(right)
        # matrix multiplication
        return _dot(self, him)
    
    def __rmul__(self, left):
        # if multiplying with _el object, call the elementwise operation
        if isinstance(left, _el): return _el(right=self)
        elif _isscalar(left): return self.__elmul__(left)
        # matrix multiplication
        return _dot(left, self)
    
    def __add__(self, him):
        from operator import isSequenceType
        a = self._a
        na = _marray(self.dtype, self.msize)
        if isSequenceType(him):
            for i, x in enumerate(him):
                na._a[i] = a[i] + x
        else:
            for i in xrange(len(a)):
                na._a[i] = a[i] + him
        return na
    __radd__ = __add__
    
    def __sub__(self, him):
        from operator import isSequenceType
        a = self._a
        na = _marray(self.dtype, self.msize)
        if isSequenceType(him):
            for i, x in enumerate(him):
                na._a[i] = a[i] - x
        else:
            for i in xrange(len(a)):
                na._a[i] = a[i] - him
        return na
    def __rsub__(self, him):
        from operator import isSequenceType
        a = self._a
        na = _marray(self.dtype, self.msize)
        if isSequenceType(him):
            for i, x in enumerate(him):
                na._a[i] = x - a[i]
        else:
            for i in xrange(len(a)):
                na._a[i] = him - a[i]
        return na
    
    def __neg__(self):
        from array import array
        na = _marray(self.dtype, self.msize, 
                     array(_dtype2array[self.dtype], (-x for x in self._a)))
        return na
    
    # comparisons
    def __ge__(self, other):
        from array import array
        if _isscalar(other):
            if hasattr(other, '__len__'): other = _cycle(other)
            else: other = _repeat(other)
        na = _marray('bool', self.msize, 
                     array(_dtype2array['bool'],
                           (x >= y for x, y in _izip(self, other))))
        return na
    
    # element access
    def __iter__(self):
        return iter(self._a)
    
    def __len__(self):
        return max(self.msize)
    
    # same interface as __getitem__ but 1-based indexing
    # def __call__(self, *i):
    #     return self.__getitem1__(i)
    
    def __getitem__(self, i):
        # determine the size of the new array
        nshp = _ndshape(self.msize, *i)
        ins = list(_ndilin(self.msize, *i))
        return _marray(self.dtype, nshp, [ self._a[i] for i in ins ])
    
    # >> a = reshape(1:15,5,3)
    # >> a(eye(3)==1)
    # ans = [1, 5, 9]
    def __getitem1__(self, i):
        # determine the size of the new array
        nshp = _ndshape1(self.msize, *i)
        i = ( isinstance(x, _marray) and iter(x._a) or x for x in i )
        ins = list(_ndilin1(self.msize, *i))
        na = _marray(self.dtype, (0,0))
        na._init_data(self.dtype, nshp, ( self._a[i] for i in ins ))
        return na
    
    def __setitem__(self, i, val):
        for i, v in _izip(_ndilin(self.msize, *i), val):
            self._a[i] = v
    
    def __setitem1__(self, i, val):
        # determine the size of the new array
        nshp = _ndshape1(self.msize, *i)
        i = ( isinstance(x, _marray) and iter(x._a) or x for x in i )
        ins = list(_ndilin1(self.msize, *i))
        if _isscalar(val):
            if hasattr(val, '__len__'): val = _cycle(val)
            else: val = _repeat(val)
        for j, v in _izip(ins, val):
            self._a[j] = v
    
    # types
    def __asindex__(self):
        return ( int(x) for x in self._a )
    
    # properties
    def transposed(self):
        assert len(self.msize) == 2
        from array import array
        na = _marray(self.dtype, self.msize[::-1], 
                     array(_dtype2array[self.dtype], self._a))
        return na
    T = property(transposed, None, None, "Transpose.")
    
    def row(self, row, i=None):
        M, N = self.msize
        if i is None: i = xrange(N)
        return ( self._a[row + x*M] for x in i )
    
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
    return int(_floor(stop-start)/step + 1)

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
        if self._a is None:
            self._init_data('double', (1, len(self)), self.__iter__())
        return mvar._ctypes_get(self)
    ctypes = property(_ctypes_get, None, None, 
                      "Ctypes-wrapped data object.")
    
    def __iter__(self):
        value = self.start
        while value <= self.stop:
            yield value
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
        a = _marray(self.dtype, self.msize)
        if self._a is None:
            self._init_data('double', (1, len(self)), self.__iter__())
        a._a[:] = self._a
        return a
    def __deepcopy__(self):
        a = _marray(self.dtype, self.msize)
        if self._a is None:
            self._init_data('double', (1, len(self)), self.__iter__())
        a._a[:] = self._a
        return a

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
            if isinstance(x, slice):
                x = marray(mrange(*mslice2nslice(x)))
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
            out.__setitem__(sl, x)
    return out

def size(X):
    if isinstance(X, _marray):
        return X.msize
    from operator import isSequenceType
    shp = []
    if X == []: return [0, 0]
    while isSequenceType(X):
        shp.append(len(X))
        X = X[0]
    if shp: shp = shp[::-1]
    else: shp = [0, 0]
    if len(shp) == 1: shp = [1, shp[0]]
    return shp

def rand(*args):
    from random import random
    from array import array
    if isinstance(args[0], str):
        raise NotImplemented
    if len(args) == 1:
        args = (args[0], args[0])
    a = _marray('double', args, 
                array('d', [ random() for x in xrange(_prod(args))]))
    return a

# better for psyco
def rand(*args):
    from random import random
    from array import array
    if isinstance(args[0], str):
        raise NotImplemented
    if len(args) == 1:
        args = (args[0], args[0])
    na = _marray.zeros(args, 'double')
    for i in xrange(_prod(args)):
        na._a[i] = random()
    return na

def randn(*args):
    from random import gauss
    from array import array
    if isinstance(args[0], str):
        raise NotImplemented
    if len(args) == 1:
        args = (args[0], args[0])
    na = _marray.zeros(args, 'double')
    for i in xrange(_prod(args)):
        na._a[i] = gauss(0, 1)
    return na

def reshape(A, *newsize):
    if len(newsize) == 1 and hasattr(newsize, '__len__'):
        newsize = newsize[0]
    out = A.__copy__()
    assert _prod(A.msize) == _prod(newsize)
    out.msize = newsize
    return out

# In [41]: time a = array('d', xrange(1000000))
# CPU times: user 0.92 s, sys: 0.00 s, total: 0.92 s, Wall time: 0.91 s
# In [42]: time a = array('d', range(1000000))
# CPU times: user 0.43 s, sys: 0.00 s, total: 0.43 s, Wall time: 0.43 s

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
    # finally, our internal arrays are 0-based
    dim -= 1
    n = A.msize[dim]
    stride = 1
    for x in A.msize[:dim]: stride *= x
    nshp = list(A.msize)
    nshp[dim] = 1
    # the result is nshp-aped array, lin. index xrange(_prod(nshp))
    res = []
    all = [ slice(0,n) for n in A.msize ]
    all[dim] = 0
    if dim > 1 and dim == len(nshp)-1: nshp.pop()
    for res_i, el0 in _izip(xrange(_prod(nshp)), _ndilin(A.msize, *all)):
        res.append( _sum(A._a[i] for i in xrange(el0,el0+n*stride,stride)) )
    return _marray(A.dtype, nshp, res)

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
    dim -= 1
    if A.msize[dim] == 1:
        return A.__copy__()
    for_end = 1
    for s in msize[:dim]: for_end *= s
    stride_big = 5
    for s in msize[1:dim+1]: stride_big *= s
    cp = 0
    if dim > 0: cp = stride_big - stride_big/msize[dim]
    else: cp = stride_big - 1
    prod_msize = _prod(msize)
    nshp = list(msize)
    nshp[dim] = 1
    o = zeros(nshp)
    oi = 0
    i = 0
    while oi < _prod(nshp):
        for k in xrange(for_end):
            s = _sum(A._a[i:i+msize[dim]*for_end:for_end])
            o._a[oi] = s
            oi += 1
            i += 1
        i += cp
    if len(nshp) > 2 and nshp[-1] == 1:
        o.msize = o.msize[:-1]
    # FIXME, conversion to 'native' or other type
    return o

def plot(X, Y, ltype):
    import Tkinter
    a = Tkinter.Tk()
    f = Tkinter.Frame()
    c = Tkinter.Canvas(f, width=480, height=320,
                       bg='white', borderwidth=0.1,
                       highlightbackground='black')
    c.pack()#fill='both',expand=1)
    f.pack(padx=50, pady=50)#, fill='both',expand=1)
    c.pX, c.pY = 0.0, 0.0
    c.dX, c.dY = 480.0, 320.0
    x0, x1, y0, y1 = min(X), max(X), min(Y), max(Y)
    Tkinter.Label(a,text='%0.1f'%x0).place(in_=c,x=-10,y=c.dY+5)
    Tkinter.Label(a,text='%0.1f'%x1).place(in_=c,x=c.dX-5,y=c.dY+5)
    Tkinter.Label(a,text='%0.1f'%y0).place(in_=c,x=-25,y=c.dY-10)
    Tkinter.Label(a,text='%0.1f'%y1).place(in_=c,x=-25,y=-10)
    rx, ry = (c.dX)/(x1-x0), (c.dY)/(y1-y0)
    if str(ltype) == '.':
        from itertools import izip
        for x, y in izip(X, Y):
            x, y = rx*(x-x0), -ry*(y0-y)
            c.create_oval(x-1.5, y-1.5, x+1.5, y+1.5, 
                          fill='blue', outline='blue')
    a.mainloop()
