
class _el(object):
    def __new__(cls, left=None, right=None):
        #print '---> New %s'%cls, left, right
        if left is None or right is None:
            nel = super(_el, cls).__new__(_el)
            nel.__class__ = cls
            nel.left = left
            nel.right = right
            return nel
        else:
            return cls.op(left, right)
    
    def __init__(self, left=None, right=None):
        pass

def power(A, B):
    print 'power', A, B
    if hasattr(A, 'val'): A = A.val
    if hasattr(B, 'val'): B = B.val
    return A**B

def mpower(A, B):
    print 'mpower', A, B
    if hasattr(A, 'val'): A = A.val
    if hasattr(B, 'val'): B = B.val
    return A**B

from itertools import izip as _izip, repeat as _repeat
def times(A, B):
    print 'times', A, B
    if hasattr(A, 'val'): A = A.val
    if hasattr(B, 'val'): B = B.val
    if not hasattr(A, '__len__'):
        if not hasattr(B, '__len__'): return A*B
        else: A = _repeat(A, len(B))
    elif not hasattr(B, '__len__'):
        B = _repeat(B, len(A))
    res = [ a*b for a, b in _izip(A, B)]
    return res

def mtimes(A, B):
    print 'mtimes', A, B
    if hasattr(A, 'val'): A = A.val
    if hasattr(B, 'val'): B = B.val
    res = sum([ a*b for a, b in _izip(A, B)])
    return res

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

class A:
    def __init__(self, val):
        self.val = val
    def __elmul__(self, him):
        return times(self, him)
    def __mul__(self, right):   
        # if multiplying with _el object, call the elementwise operation
        if isinstance(right, _el): return right.__class__(self, right.right)
        elif _isscalar(right): return times(self, right)
        return mtimes(self, right)
    def __rmul__(self, left):
        # if multiplying with _el object, call the elementwise operation
        if isinstance(left, _el): return left.__class__(left.left, self)
        elif _isscalar(left): return times(left, self)
        return mtimes(left, self)
    def __str__(self):
        return '%r'%self.val
    def __len__(self): return len(self.val)

a = A(1)
b = A(2)

print '-----------------------------------------------------------------------'
print '-----------------------------------------------------------------------'

print 'SCALAR*SCALAR'
print '>>> 1.*2  \n', 1*elmul*2
print '>>> 1*2   \n', 1*2

print '-----------------------------------------------------------------------'

a = A([1,2,3,4])

print 'VECTOR*SCALAR'
print '>>> a = [1,2,3,4]'
print '>>> a.*2  \n', a*elmul*2
print '>>> a*2   \n', a*2
print '>>> 2*a   \n', 2*a

print '-----------------------------------------------------------------------'

b = A([2,3,4,5])

print 'VECTOR*VECTOR'
print '>>> b = [2,3,4,5]'
print '>>> a.*b  \n', a*elmul*b
print '>>> a*b   \n', a*b
print '>>> a.*b.*a   \n', a*elmul*b*elmul*b

print '-----------------------------------------------------------------------'
