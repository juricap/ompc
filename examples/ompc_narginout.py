
def _get_narginout():
    """Return how many values the caller is expecting.
    """
    import inspect, dis
    f = inspect.currentframe()
    # step into the function that called us
    fb = f.f_back
    innames = fb.f_code.co_varnames[:fb.f_code.co_argcount]
    nargin = len([ x for x in innames if fb.f_locals.get(x, None) ])
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
        # MATLAB always assumes at least 1 value
        return nargin, 1
    return nargin, 1

class marray:
    def __init__(self, val=[]):
        self.val = []
    def __call__(self, *args):
        print "  return elements %r"%args
        return self
    def __setattr__(self, name, val):
        if name == "lvalue":
            print "  I am an L-value ('%s') being set to %r."%(name, val)
            self.val = val

def a(b=None,c=None):
    out1, out2 = None, None
    nargin, nargout = _get_narginout()
    print '  nargin = %s, nargout = %d'%(nargin, nargout)
    if nargin == 2:
        out1 = b + c
    elif nargin == 1:
        k = marray()
        out2 = '---'
    k = locals().get('k', marray())
    k(10).lvalue = 12
    return (out1, out2)[:nargout]

print '>>> a()'
print a()
print '>>> a(1)'
print a(1)
print '>>> b = a(1)'
b, c = a(1)
print b, c
print '>>> b, c = a(12, 11)'
b, c = a(12, 11)
print b, c
print ">>> a, b, c, d = a('++', '--')"
a, b, c, d = a('++', '--')
