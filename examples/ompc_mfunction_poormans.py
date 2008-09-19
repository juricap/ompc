
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

def _check_nargout(nargout, maxout):
    if nargout > maxout:
        error("Too manu output arguments!")

def get_source(obj):
    """Get object's source code. Returns None when source can't be found.
    """
    from inspect import findsource
    try: lines, lnum = findsource(obj)
    except (IOError, TypeError): return None
    return lines, lnum

def get_func_source(obj):
    """Get object's source code. Returns None when source can't be found.
    """
    from inspect import findsource
    from dis import findlinestarts
    try:
        lines, lnum = findsource(obj)
        ls = list(findlinestarts(obj.func_code))
        lstart = ls[0][1]
        lend = ls[-1][1]
    except (IOError, TypeError):
        return None
    return ''.join(lines[lstart-1:lend])

def get_comment_doc(obj, lines=None, lnum=None):
    """Get lines of comments immediately preceding an object's source code.
    Returns None when source can't be found.
    """
    from inspect import ismodule
    
    if lines is None:
        try:
            from inspect import findsource
            lines, lnum = findsource(obj)
        except (IOError, TypeError):
            return None
    
    if ismodule(obj):
        lnum = 0
        # Look for a comment block at the top of the file.
    start = lnum
    if lnum == 0:
        if lines and lines[0][:2] == '#!': start = 1
    else:
        start += 1
        if lines[start].lstrip()[:3] == 'def': start += 1
    
    while start < len(lines) and lines[start].strip() in ('', '#'):
        start = start + 1
    
    if start < len(lines) and lines[start].lstrip()[:1] == '#':
        comments = []
        end = start
        while end < len(lines) and lines[end].lstrip()[:1] == '#':
            comments.append(lines[end].strip()[2:])
            end = end + 1
        return '\n'.join(comments)

MFUNC_TEMPLATE = '''\

def %(name)s(%(args)s):
    """%(doc)s"""
    nargin, nargout = _get_narginout()
    if nargout > %(maxout)d:
        error("Too manu output arguments!")
%(body)s
    return (%(outnames)s,)[:nargout]
'''

def mfunction(outnames):
    # from functools import update_wrappe
    def dec(func):
        innames = func.func_code.co_varnames[:func.func_code.co_argcount]
        lines, lnum = get_source(func)
        doc = get_comment_doc(func, lines, lnum)
        func_src = MFUNC_TEMPLATE%{'name': func.func_name,
                                   'args': ', '.join(innames),
                                   'doc': doc,
                                   'maxout': len(outnames.split(',')),
                                   'body': get_func_source(func),
                                   'outnames': outnames}
        g = func.func_globals
        exec func_src in g
        open('%s.temp.py'%func.func_name, 'wb').write(func_src)
        func_new = g[func.func_name]
        return func_new
    return dec

@mfunction("x")
def coinflip(ndraws=None, p=None):
    # Generate a list x of zeros and ones according to coin flipping (i.e.
    # Bernoulli) statistics with probability p of getting a 1.
    # 1/20/97  dhb  Delete obsolet rand('uniform').
    # 7/24/04  awi  Cosmetic.    
    
    # Generate ndraws random variables on the real interval [0,1).
    unif = rand(ndraws, 1)
    # Find all of the ones that are less than p.
    # On average, this proportion will be p.
    index = find(unif < p)
    [nones, m] = size(index)
    # Generate an array of zeros and then set
    # the ones found in the previous step to 1.
    x = zeros(ndraws, 1)
    if (nones != 0):    
        x(index).lvalue = ones(nones, m)    
    end

@mfunction("out1, out2")
def add(a=None, b=None):
    # adds 2 numbers
    if nargin == 2:
        out2 = (a, b)
    out1 = a + b

#help(coinflip)
#help(add)

print "Adding 1 and 2"
print add(1, 2)
