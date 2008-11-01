
def _get_narginout():
    """Return how many values the caller is expecting.
    """
    import sys, dis
    f = sys._getframe()
    # step into the function that called us
    fb = f.f_back
    innames = fb.f_code.co_varnames[:fb.f_code.co_argcount]
    nargin = len([ x for x in innames if fb.f_locals.get(x, None) is not None ])
    vargin = fb.f_locals.get('varargin', None)
    if vargin is not None:
        nargin += len(vargin)
    # nargout is one frame back
    f = f.f_back.f_back
    c = f.f_code
    i = f.f_lasti
    bytecode = c.co_code
    instruction = ord(bytecode[i+3])
    if instruction == dis.opmap['UNPACK_SEQUENCE']:
        howmany = ord(bytecode[i+4])
        return nargin, howmany
    elif instruction == dis.opmap['POP_TOP']:
        return nargin, 0
    return nargin, 0

class mfunction_arguments:
    def __init__(self, *names):
        import types
        if len(names) == 1 and isinstance(names[0], types.FunctionType):
            self.names = []
            return mfunction_arguments.__call__(self, names[0])
        self.names = names
    def __call__(self, mfunc):
        from byteplay import Code, LOAD_GLOBAL, CALL_FUNCTION, POP_TOP, \
                                   UNPACK_SEQUENCE, STORE_FAST, LOAD_FAST, \
                                   LOAD_CONST, BUILD_LIST, BINARY_SUBTRACT, \
                                   BINARY_MULTIPLY, BINARY_ADD, RETURN_VALUE, \
                                   SLICE_2
        # load the current function code
        c = Code.from_code(mfunc.func_code)
        
        # for nargin/nargout emulation
        nargin = mfunc.func_code.co_argcount
        if 'varargin' in self.names:
            assert mfunc.func_code.co_flags&4 == 4
            nargin = -nargin - 1
        mfunc.func_nargin = nargin
        mfunc.func_nargout = len(self.names)
        if 'varargout' in self.names:
            mfunc.func_nargout = -mfunc.func_nargout
        
        # prepare the bytecode changes
        pre_code = [(LOAD_GLOBAL,'_get_narginout'),
                    (CALL_FUNCTION,0),
                    (UNPACK_SEQUENCE, 2),
                    (STORE_FAST,'nargin'),
                    (STORE_FAST,'nargout')]
        if 'varargin' in self.names:
            # varargin = mcellarray(varargin)
            pre_code += [(LOAD_GLOBAL, 'mcellarray'),
                    (LOAD_GLOBAL, 'varargin'),
                    (CALL_FUNCTION, 1),
                    (STORE_FAST, 'varargin')]
        if 'varargout' in self.names:
            # varargout = mcellarray([None]*(nargout-1))
            pre_code += [(LOAD_GLOBAL, 'mcellarray'),                   
                    (LOAD_CONST, None),
                    (BUILD_LIST, 1),
                    (LOAD_FAST, 'nargout'),
                    (LOAD_CONST, 1),
                    (BINARY_SUBTRACT, None),
                    (BINARY_MULTIPLY, None),
                    (CALL_FUNCTION, 1),
                    (STORE_FAST, 'varargout')]
        # adjust the function preamble
        c.code[:0] = pre_code
        # replace LOAD_GLOBAL with LOAD_FAST for 
        for i, x in enumerate(c.code):
            if x[0] == LOAD_GLOBAL and \
                (x[1] == 'nargout' or x[1] == 'nargin' or \
                 x[1] == 'varargout' or x[1] == 'varargin'):
                c.code[i] = (LOAD_FAST,x[1])
        
        # return value
        # first remove the original return
        assert c.code[-1][0] == RETURN_VALUE
        ret_code = [(POP_TOP, None)]     # easier than deleting LOAD_CONST
        # return what is supposed to be returned
        if 'varargout' in self.names:
            if len(self.names) > 1:
                ret_code += [(LOAD_FAST, x) for x in self.names[:-1]] + \
                            [(BUILD_LIST, len(self.names)-1),
                             (LOAD_FAST , 'varargout'),
                             (BINARY_ADD, None)]
            else:
                ret_code += [(LOAD_FAST , 'varargout'),
                             (BINARY_ADD, None)]
        else:
            ret_code += [(LOAD_FAST, x) for x in self.names] + \
                        [(BUILD_LIST, len(self.names))]
        ret_code += [(LOAD_FAST, 'nargout'),
                     (SLICE_2, None),
                     (RETURN_VALUE, None)]
        c.code[-1:] = ret_code
        
        # replace the original bytecode
        mfunc.func_code = c.to_code()
        return mfunc
    
    @staticmethod
    def make_varargout(n):
        return mcellarray([None]*n)

class mvar(object):
    def __init__(self, *args):
        self.args = args
    def __setattr__(self, attr, val):
        if attr == 'lvalue':
            if hasattr(self, '__ompc_args__'):
                s, i = self.__ompc_args__
                if len(s) <= i:
                    s.extend([None]*(i-len(s)) + [val])
                else:
                    s[i] = val
        else:
            object.__setattr__(self, attr, val)

class marray(list, mvar):
    def __call__(self, x):
        return list.__getitem__(self, x-1)

class mcellarray(list, mvar):
    def __new__(cls, x=None):
        if x is None: return super(mcellarray, cls).__new__(cls)
        return super(mcellarray, cls).__new__(cls, x)
    def __call__(self, i):
        out = mvar(list.__getitem__(self, i-1))
        out.__ompc_args__ = self, i-1
        return out

def size(X):
    from operator import isSequenceType
    out = marray()
    while isSequenceType(X):
        out.append(len(X))
        X = X[0]
    return out

def rand(*args):
    from random import random
    args = list(args)
    if len(args) > 1:
        return [ rand(*args) for i in xrange(args.pop(0)) ]
    else:
        return [ random() for i in xrange(args.pop()) ]

def mrange(start, stop=None, step=None):
    if step is None: step = 1
    v = start
    while v <= stop:
        yield v
        v += step

class _m(object):
    def __getitem__(self, i):
        assert isinstance(i, slice)
        return mrange(i.start, i.stop, i.step)

m_ = _m()

@mfunction_arguments("out", "varargout")
def example1(a=None, b=None, *varargin):
    print "NIN: %d, NOUT: %d"%(nargin, nargout)
    print "VARARGIN", varargin, "VARARGOUT:", varargout
    out = 12
    return nargout > 0 and range(nargout) or None

# from http://www.mathworks.com/access/helpdesk/help/techdoc/ref/varargout.html
@mfunction_arguments("s", "varargout")
def mysize(x=None):
    nout = max(nargout, 1) - 1
    s = size(x)
    for k in m_[1:nout]:
        varargout(k).lvalue = mcellarray([s(k)])

from byteplay import Code
#c = Code.from_code(example1.func_code)
c = Code.from_code(mysize.func_code)
print c.code

example1()
example1(1)
example1(1,2)
example1(1,2,4)
example1(1,2,4,'a')

a = example1(1,2,4,'a')
[a, b] = example1(1,2,4,'a')
[a, b, c] = example1(1,2,4,'a')
[a, b, c, d] = example1(1,2,4,'a')
print 'OUT:', a

[s,rows,cols] = mysize(rand(4,5))
print 's = %s, rows = %r, cols = %r'%(s, rows, cols)
