
import numpy as _N
import pylab as _P
import scipy as _S

from ompc import _print, _keywords

# errors and warnings

class OMPCException(Exception):
    def __init__(self,msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

def error(x):
    raise OMPCException(x)

# documentation

def _func2str(f):
    from types import FunctionType, StringType
    if isinstance(f,FunctionType):
        fname = f.__name__
    elif isinstance(f,StringType):
        fname = f
    else:
        try:
            fname = str(f)
        except:
            raise error("Don't know how to lookup help for object %r"%f)
    return fname

try:
    from ompcdoc import open as _webopen
except:
    from webbrowser import open as _webopen

def mhelp(f):
    """mhelp(func)
    
    Navigate browser to the online MATLAB(R) documentation for function 
    specified in the 'func' parameter.
    func - can be either a function object or a string containing the name 
    of the function.
    """
    fname = _func2str(f)
    MURL = 'http://www.mathworks.com/access/helpdesk/help/techdoc/index.html'\
           '?/access/helpdesk/help/techdoc/ref/%s.html'
    return _webopen(MURL%fname)

def ohelp(func):
    """ohelp(func)
    
    Navigate browser to the online Octave documentation for function specified
    in the 'func' parameter.
    func - can be either a function object or a string containing the name 
    of the function.
    """
    fname = _func2str(f)
    #f0 = fname[0].upper()
    #if f0 == '_': f0 = 'Z'
    #elif not f0.isalpha(): f0 = 'A'
    OURL = 'http://octave.sourceforge.net/doc/f/%s.html'
    _webopen(OURL%(fname))

# helpers to allow matlab engine behavior

__ompc_builtin__ = {}
__ompc_whos__ = {}
_MEND = None
end = _MEND

def _get_nargout():
    """Return how many values the caller is expecting."""
    import inspect, dis
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

def _get_argout_name():
    """Return name of variable that return value will be assigned to."""
    import inspect, dis
    f = inspect.currentframe()
    f = f.f_back.f_back
    c = f.f_code
    # dis.disassemble_string(c.co_code)
    i = f.f_lasti
    bytecode = c.co_code
    instruction = ord(bytecode[i+3])
    if instruction != dis.opmap['STORE_NAME']:
        # POP_TOP, ROT_TWO and UNPACK_SEQUENCE are not allowed in MATLAB
        # fro constructors
        error("Construction assignment into multiple values is not allowed.")
    name = c.co_names[ord(bytecode[i+4])]
    return name

def mfunction(func,globals_=[],persistent_=[]):
    """Decorator that populates function with MATLAB compatible interface.
    It adds local variables,
        nargin  - number of actually submitted arguments,
        nargout - number of expected output arguments.
    
    All OMPC functions should return OMPC compatible arrays, a call is injected
    to transform all return values to OMPC arrays.
    """
    c = Code.from_code(func.func_code)
    c.code[:0] = [(LOAD_GLOBAL,'_get_nargout'),
                  (CALL_FUNCTION,0),
                  (STORE_FAST,'nargout')]
    # FIXME
    # the nargin is determined differently for varargin (==> *args) and
    # a regular list of arguments
    c.code[:0] = [(LOAD_GLOBAL,'len'),
                  (LOAD_FAST,'args'),
                  (CALL_FUNCTION,1),
                  (STORE_FAST,'nargin'),]
    # replace LOAD_GLOBAL with LOAD_FAST for 
    for i, x in enumerate(c.code):
        if x[0] == LOAD_GLOBAL and \
            (x[1] == 'nargout' or x[1] == 'nargin'):
            c.code[i] = (LOAD_FAST,x[1])
    
    # FIXME
    # inject return value conversion
    func.func_code = c.to_code()
    return func

# all MATLAB

# language

def iskeyword(word):
    from ompc import _keywords
    return word in _keywords

def who(*args,**kwargs):
    """Return list of variables in the current workspace."""
    nargin = len(args)
    nargout = _get_nargout()
    vars = []
    if kwargs.get('file'):
        # FIXME
        pass
    elif kwargs.get('regexp'):
        for x in __ompc_whos__.keys():
            if regexp(x, kwargs['regexp'], 'match'):
                vars.append(x)
    elif nargin > 0:
        for x in __ompc_whos__.keys():
            var = __ompc_whos__.get(x)
            if var is not None:
                vars.append(var)
    else:
        vars = __ompc_whos__.keys()
    if nargout > 0:
        # return as cell
        return mcellarray(vars)
        
    # print the results
    _print("Your variables are:")
    _print(vars)

def whos(*args,**kwargs):
    """Pretty-print a list of variables in the current workspace."""
    for var in who(*args, **kwargs):
        _print(var, var.shape, var.size, var.dtype)
    return 

# io

# look at http://mien.sourceforge.net/ they have a mat file reader too
# also http://abel.ee.ucla.edu/cvxopt/examples/extra-utilities/matfile.py/view

def load(fname, *args):
    import inspect
    a = _S.io.loadmat(fname, *args)
    f = inspect.currentframe()
    for var, val in a.iteritems():
        if var[:2] == '__': continue
        if args:
            if var in args:
                f.f_back.f_globals[var] = val
        else:
            f.f_back.f_globals[var] = val

def save(fname, *args):
    import inspect
    f = inspect.currentframe()
    d = {}
    for var in args:
        d[var] = f.f_back.f_globals[var]
    _S.io.savemat(fname, d)
        
def exist(name,tp=None):
    """ Return 
        1 if the name exists as a variable,
        2 if the name is a file of a function file on path,
        3 if the name is a binary function,
        4 is not returned at the moment,
        5 if the name is a built-in function,
        6 if there is a .pyc file with the name,
        7 if the name is a directory,
        8 id the name belongs to a Python instance.
    
    Second argument kind=['builtin'|'class'|'dir'|'file'|'var']
    """
    from os.path import exists, isdir
    if tp is None:
        if name in __ompc_whos__: return 1
        elif exists(name): return 2
        elif name in __ompc_builtin__: return 5
        elif isdir(name): return 7
        elif any(map(exists, [name, name+'.m'])): return 2
    
    if tp == 'file':
        from os.path import exists
        return exists(name) and 2 or 0
    elif tp == 'class':
        # is it a Python instance ?
        return (name in globals() and name not in __omcp_whos__) \
            and 8 or 0
    elif tp == 'dir':
        return isdir(name) and 7 or 0
    elif tp == 'builtin':
        return name in __ompc_builtin__ and 5 or 0
    elif tp == 'var':
        return name in __omcp_whos__ and 1 or 0
    
    return 0

class _mvar:
    def __init__(self, name):
        """
        """
        __ompc_whos__[_get_argout_name()] = self

class _marray_numpy(_mvar):
    def __init__(self,a):
        _mvar.__init__(self)
        self._array = a
    def __len__(self):
        return len(self._str)
    def __call__(self,i):
        if isinstance(i,_mslice):
            return self._array[i.__slice_base0__()]
        elif i is _MEND:
            i = len(self._str)
        if i < 1 or i > len(self._str):
            raise IndexError
        return self._str[i-1]
    def __str__(self):
        return self._str
    def __repr__(self):
        return repr(self._str)

class mcellarray(_mvar, list):
    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)
        _mvar.__init__(self, _get_argout_name())

# CHOOSE one of the implementations of array
marray = _marray_numpy

# __getslice__ is deprecated, don't even try, push people to higher Python
class _m_array(_mvar):
    def __getitem__(self,i):
        if len(i) > 1:
            raise IndexError('String is 1D.')
        if isinstance(ins,slice):
            self.__init__(i.start,i.stop,i.step)
        elif isinstance(ins,int):
            self.__init__(i,i,1)
        else:
            raise IndexError('Index must be slice or integer.')
        return slice(self._start,self._stop,self._step)

class _mtype(object):
    def __init__(self,*args):
        # the same parameters as zeros, a list of sizes and 'class'
        if type(args[-1]) is str:
            tp = i.pop()
            
        pass

class _mtype_meta(object):
    def __new__(cls,names,bases,dict):
        return object.__new__(cls)
    def __getitem__(cls,i):
        if i is None:
            return _mtype([])
        elif operator.isSequenceType(i):
            # split into rows by ';'
            pass
        return _mtype(*i)

class m_:
    __metaclass__ = _mtype_meta
    pass

class _mslice:
    def __init__(self,start=0,stop=None,step=1):
        self._start = int(start)
        # dN = 1e-9
        # if step < 0:
        #     self._stop = stop - (-dN > step/2.0 and dN or step/2.0)
        # else:
        #     self._stop = stop + ( dN < step/2.0 and dN or step/2.0)
        if step < 0: self._stop = int(stop) - 1
        else: self._stop = stop + 1
        self._step = step
    def __getitem__(self,i):
        if len(i) > 1:
            raise IndexError('String is 1D.')
        if isinstance(ins,slice):
            self.__init__(i.start,i.stop,i.step)
        elif isinstance(ins,int):
            self.__init__(i,i,1)
        else:
            raise IndexError('Index must be slice or integer.')
        return slice(self._start,self._stop,self._step)
    def __slice__(self):
        # FIXME, will this work?, with the dN probably yes
        return slice(self._start,self._stop,self._step)
    def __slice_base0__(self):
        # FIXME, will this work?, with the dN probably yes
        stop = self._stop-1
        if stop < 0: stop = None
        return slice(self._start-1,stop,self._step)

class _mstr(_mvar):
    def __init__(self,s):
        _mvar.__init__(self)
        self._str = s
    def __len__(self):
        return len(self._str)
    def __call__(self,i):
        if isinstance(i,_mslice):
            return self._str[i.__slice_base0__()]
        elif i is _MEND:
            i = len(self._str)
        if i < 1 or i > len(self._str):
            raise IndexError
            #raise error('String:')
        return self._str[i-1]
    def __str__(self):
        return self._str
    def __repr__(self):
        return repr(self._str)

def fgetl(fid):
    return _mstr(fid.readline())

# plotting

def figure(*args,**kwargs):
    #arrs, kwargs = __plot_args_matlab(args,kwargs)
    fig = _P.figure(args[0])#kwargs)
    drawnow()
    return fig

def drawnow():
    _P.draw()

class GCAManager:
    def __formatpy__(self):
        #print "getting"
        return _P.gca()
    def __call__(self,**kwargs):
        return _P.gca(**kwargs)

gca = GCAManager()

def set(x,*args,**kwargs):
    fmp = getattr(x,'__formatpy__',None)
    if fmp is not None:
        o = fmp()
        args, kwargs = __plot_args_matlab(args,kwargs)
        assert len(args) == 0
        for k, v in kwargs.items():
            try: getattr(o,'set_%s'%k)(v)
            except: "Property '%s' not supported."%k

def xlabel(*args,**kwargs):
    d, kwargs = __plot_args_matlab(args[1:],kwargs)
    return _P.xlabel(args[0], **kwargs)

def ylabel(*args,**kwargs):
    d, kwargs = __plot_args_matlab(args[1:],kwargs)
    return _P.ylabel(args[0], **kwargs)

def zlabel(*args,**kwargs):
    d, kwargs = __plot_args_matlab(args[1:],kwargs)
    return _P.zlabel(args[0], **kwargs)

def pause():
    from matpy_platform import getch
    #print 'Press a key to continue ...'
    getch()

# 3d

def surfc(X,Y,Z,*args):
    return pcolor(X,Y,Z,*args)
    
def __plot_args(args,kwargs):
    from sets import Set
    args = list(args)
    arrs = []
    colorspec = False
    for i in range(len(args)):
        arg = args.pop(0)
        if type(arg) is _N.ndarray:
            sz = size(arg)
            if arg.ndim == 2 and any(sz == 1):
                arrs += [ arg.reshape(sz.max()) ]
            else:
                arrs += [ arg ]
        elif type(arg) is str:
            if not colorspec and len(arg) <= 3 and \
                len(Set(arg).intersection('bgrcmykw'+'-.:,o^v<>s+xDd1234hHp|_')) > 0:
                arrs += [ arg ]
            else:
                args.insert(0,arg)
            colorspec = True
            break
        else:
            a = array(arg)
            if a.ndim == 0:
                a.resize(1,1)
                #print a
            elif a.ndim < 2:
                a.resize(len(a),1)
            arrs += [a]
    kwargs.update( zip([x.lower() for x in args[::2]],args[1::2]) )
    return arrs, kwargs

def __plot_args_matlab(args,kwargs):
    from sets import Set
    args = list(args)
    arrs = [[]]
    colorspec = False
    while args:
        arg = args.pop(0)
        if type(arg) is _N.ndarray:
            if colorspec:
                #arrs += [[]]
                colorspec = False
            sz = size(arg)
            if arg.ndim == 2 and any(sz == 1):
                arrs[-1] += [ arg.reshape(sz.max()) ]
            else:
                arrs[-1] += [ arg ]
        elif type(arg) is str:
            if not colorspec and len(arg) <= 4 and \
                len(Set(arg).intersection('bgrcmykw'+'-.:,o^v<>s+xDd1234hHp|_')) > 0:
                arrs[-1] += [ arg ]
            else:
                if arg.lower() in ['XTick','YTick']:
                    arg = arg.lower() + 's'
                args.insert(0,arg)
                break
            colorspec = True
        else:
            a = array(arg)
            if a.ndim == 0:
                print a.shape
                a.resize(0,0)
                #print a
            elif a.ndim < 2:
                a.shape = (len(a),1)
            arrs[-1] += [a]
    kwargs.update( zip([x.lower() for x in args[::2]],args[1::2]) )
    return arrs[0], kwargs

def plot(*args,**kwargs):
    arrs, kwargs = __plot_args_matlab(args,kwargs)
    return _P.plot(*arrs,**kwargs)

def loglog(*args,**kwargs):
    arrs, kwargs = __plot_args_matlab(args,kwargs)
    return _P.loglog(*arrs,**kwargs)

def plot3(*args,**kwargs):
    return plot(args[0],args[1],*args[3:],**kwargs)

# functions

def max(X,Y=[],axis=1):
    axis -= 1
    tX, tY = X, Y
    if _N.iscomplex(X.flat[0]): tX = abs(X)
    if len(tY) > 0:
        if _N.iscomplex(Y.flat[0]): tY = abs(Y)
        return _N.maximum(tX,tY)
    else:
        nargout = _get_nargout()
        print nargout
        if nargout == 1:
            return _N.max(tX,axis)
        elif nargout == 2:
            # slow
            i = _N.argmax(tX,axis)
            return _N.max(tX,axis), i
#             i = _N.argmax(tX,axis)
#             sh = X.shape
#             index = [ slice(0,x,1) for x in sh ]
#             if axis == 0:
#                 index[1] = range(sh[1])
#             else:
#                 index[0] = range(sh[0])
#             index[axis] = i
#             print index
#             return _N.ndarray.__getslice__(index)
        else:
            raise Exception('too many output vals')

def min(X,Y=[],axis=1):
    axis -= 1
    tX, tY = X, Y
    if _N.iscomplex(X.flat[0]): tX = abs(X)
    if len(tY) > 0:
        if _N.iscomplex(Y.flat[0]): tY = abs(Y)
        return _N.minimum(tX,tY)
    else:
        nargout = _get_nargout()
        print nargout
        if nargout == 1:
            return _N.min(tX,axis)
        elif nargout == 2:
            # slow
            i = _N.argmin(tX,axis)
            return _N.min(tX,axis), i
#             i = _N.argmin(tX,axis)
#             sh = X.shape
#             index = [ slice(0,x,1) for x in sh ]
#             if axis == 0:
#                 index[1] = range(sh[1])
#             else:
#                 index[0] = range(sh[0])
#             index[axis] = i
#             return _N.ndarray.__getslice__(index)
        else:
            raise Exception('too many output vals')

def gradient(f,*args):
    return _N.array(_N.gradient(f,*args))
gradient.__doc__ = _N.gradient.__doc__

def diff(x,M=[],axis=1):
    return diff(x,axis-1)

def fminsearch(fn, guess):
    return _S.optimize.fmin(fn,guess)

def quad(fn, a, b):
    return _S.integrate.quad(fn,a,b)[0]

def spline(x,y,xx=None):
    x = array(x,'f8')
    srep = None
    if len(y) > len(x):
        # WRONG
        # this little hack doesn't seem to help in getting the derivatives 
        # at the right
        dx = 1e-3
        x = r_[x[0]-dx,x,x[-1]+dx]
        y[0] = y[1] - y[0]*dx
        y[-1] = y[-2] + y[-1]*dx
    if xx is None:
        # WRONG
        # this doesn't return what MATLAB would, but can be used with ppval
        return _S.interpolate.fitpack.splrep(x,y)
    xx = array(xx)
    srep = _S.interpolate.fitpack.splrep(x,y,xb=xx.min(),xe=max(xx.max(),x[-1]))
    return _S.interpolate.fitpack.splev(xx,srep)

def ppval(pp,xx):
    return _S.interpolate.fitpack.splev(xx,pp)

def test_spline():
    x = mr_[-4:4]; y = r_[0, .15, 1.12, 2.36, 2.36, 1.46, .49, .06, 0];
    pp = spline(x,r_[0, y, 0]);
    xx = linspace(-4,4,101);
    plot(x,y,'o',xx,ppval(pp,xx),'-');

def spectrogram(x, window, Fs=1, NFFT=256, noverlap=None):
    """
    Compute a spectrogram of data in x.  Data are split into NFFT
    length segements and the PSD of each section is computed.  The
    windowing function window is applied to each segment, and the
    amount of overlap of each segment is specified with noverlap

    See pdf for more info.

    The returned times are the midpoints of the intervals over which
    the ffts are calculated
    """
    if not noverlap:
        noverlap = window/2
    x = _N.asarray(x)
    assert(NFFT>noverlap)
    if _N.log(NFFT)/_N.log(2) != int(_N.log(NFFT)/_N.log(2)):
       raise ValueError, 'NFFT must be a power of 2'
    
    # zero pad x up to NFFT if it is shorter than NFFT
    if len(x)<NFFT:
        n = len(x)
        x = _N.resize(x, (NFFT,))
        x[n:] = 0    
    
    # for real x, ignore the negative frequencies
    if _N.iscomplex(x): numFreqs=NFFT
    else: numFreqs = NFFT//2+1
    
    windowVals = window(ones((NFFT,),x.dtype))
    step = NFFT-noverlap
    ind = _N.arange(0,len(x)-NFFT+1,step)
    n = len(ind)
    Pxx = _N.zeros((numFreqs,n), 'f8')
    # do the ffts of the slices
    
    for i in range(n):
        thisX = x[ind[i]:ind[i]+NFFT]
        thisX = windowVals*detrend(thisX)
        fx = _N.absolute(fft(thisX))**2
        # Scale the spectrum by the norm of the window to compensate for
        # windowing loss; see Bendat & Piersol Sec 11.5.2
        Pxx[:,i] = _N.divide(fx[:numFreqs], norm(windowVals)**2)
    t = 1/Fs*(ind+NFFT/2)
    freqs = Fs/NFFT*_N.arange(numFreqs)
    
    return Pxx, freqs, t

# arrays

class _seti:
    def __getitem__(self,x):
        if type(x) is slice:
            s,e,t = (x.start is not None and x.start or 1)-1,x.step-1,x.stop is None and 1 or x.stop
            if round(t) != t:
                raise Exception('The step must be integer value!')
            e += 1e-6*t
            return arange(s,e,t)
        return array(x)-1
    def __getslice__(self,start,stop):
        return arange(start-1,stop)

mri_ = _seti()

class _setitem:
    def __init__(self,ibase=0):
        self.__ibase = ibase
    def __len__(self):
        return 0
    def __getitem__(self,x):
        if type(x) is slice:
            step = x.stop or 1.
            if x.start is None:
                return arange(1,x.step+0.1*step,x.stop)
            return arange(x.start,x.step+0.1*step,x.stop)
        return array(x)
    def __getslice__(self,start,stop):
        return arange(start,stop+1)

mr_ = _setitem()

def sub2ind(sz, *args):
    # index goes rows, cols, then stacking
    error("NotImplemented")

_m2d = { 'int8': 'i1' }
def __mtype_to_dtype(x):
    return getattr(_m2d, x, 'f8')

def __arraymaker_args(args):
    if type(args[-1]) is str:
        mtype = args.pop()
        return args,__mtype_to_dtype(mtype)
    return args, 'f8'

def ones(*args):
    args, dtype = __arraymaker_args(args)
    return _N.ones(args,dtype)

#@mfunction
def zeros(*args):
    args, dtype = __arraymaker_args(args)
    return _N.zeros(args,dtype)

#@mfunction
def size(x,ndim=None,ibase=1):
    #print '---------> size x is', x
    sz = _N.shape(x)
    if len(sz) < 1:
        if type(x) is str:
            sz = (1,len(x))
        else:
            sz = (1,1)
    if len(sz) < 2:
        sz = (1,sz[0])
    if ndim is not None:
        sz = sz[ndim-ibase]
    #print 'size', sz, type(sz)
    return sz

def isempty(x):
    return len(mat(x).A) == 0
    
def length(x):
    return len(x)

# cells

def cell2mat(x):
    return mat(x)

# strings

def str2num(x):
    r = array([])
    try: r = int(x)
    except:
        try: r = float(x)
        except: pass
    return r

def num2str(x):
    return str(x)

def sprintf(x,*args):
    return x%args

def strcmp(a,b):
    """Return bool(1) if a and b are identical strings.
    Should work on cellarrays and return size-matching array of bools.
    Return False on any non-strings.
    """
    return a == b

def strcmpi(a,b):
    return strcmp(a.lower(), b.lower())
strcmpi.__doc__ = strcmp.__doc__ + "The comparison is case-insensitive"

def deblank(s):
    """Strip trailing spaces blanks.
    """
    return 

def strtrim(s):
    """Takes also cellarrays and applies on all of them.
    """
    # create a new string that will contain stripped "s"
    return s.strip()

def cellstr(c):
    """Take all row strings in a char array, strip them of blanks and feed
    into a new cellarray.
    """
    ca = mcellarray()
    for x in c:
        ca.append(x.strip())
    return ca

def isspace(s):
    return 
def isstr(x):
    return type(x) is str

def regexp(s, expr, *args):
    """
        expr - can be a cell array (list) for multiple searches
        
        Can return the following:
        start, end, extents, matches, tokens, names, splits
    """
    qs = ['start','end','tokenExtents','match','tokens','names','split']
    opts = ['matchcase','ignorecase','dotall','dotexceptnewline',
            'stringanchors','lineanchors','literalspacing','freespacing',
            'once','warnings']
    ret_qs = [ args for x in args if x in qs ]
    options = [ args for x in args if x not in qs ]
    # http://www.mathworks.com/access/helpdesk/help/techdoc/index.html?/access/helpdesk/help/techdoc/ref/regexp.html
    # translate (?<tokenname>) to Python equivalent
    flags = re.S
    method = re.finditer
    if 'match' in ret_qs: re_method = re.finditer
    if 'split' in ret_qs: re_metho
    
    if 'matchcase' in options: flags ^= flags&re.I
    if 'ignorecase' in options: flags |= re.I
    if 'dotexceptnewline' in options: flags ^= flags&re.S
    if 'dotall' in options: flags |= re.S
    if 'literalspacing' in options: flags ^= flags&re.X
    if 'freespacing' in options: flags |= re.X
    if 'stringanchors' in options: flags ^= flags&re.M
    if 'lineanchors' in options: flags |= re.M
    
    import re
    # replace ((?-ismx))
    # FIXME
    
    res = re_method(expr,s,flags)
    for x in res:
        if 'tokens' in ret_qs:
            ret_vals.append()
        
        if 'once' in options:
            break
    
    return ret_vals

def disp(*args):
    # FIXME
    for x in args:
        _print(x,end='')
    _print()
