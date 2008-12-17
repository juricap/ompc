
# the default implementation at the moment is the ompc_numpy module 
import ompclib_numpy
from ompclib_numpy import *
from ompclib_numpy import _ompc_base, _marray, _size, _dtype
from ompc import _get_narginout
from ompclib_platform import getch

__ompc_all__ = ['addpath', 'tic', 'toc', 'mhelp', 'ohelp', 'pause']
__ompc_all__ += ompclib_numpy.__ompc_all__

# some functions that do not depend on the implementation of the numerical 
# object can be placed here

def addpath(*args):
    import sys
    pos = None
    last = args[-1]
    if last.lower() in ['-begin', '-end', '-frozen']:
        args = args[:-1]
        if last.lower() == '-begin':
            pos = 0
        elif sys.platform.startswith('win') and last.lower() == '-frozen':
            raise NotImplementedError()
    if pos is None:
        sys.path += list(args)
    else:
        sys.path.insert(pos, list(args))

_tictoc_t0 = None
from time import clock as _clock
def tic():
    global _tictoc_t0
    _tictoc_t0 = _clock()

def toc():
    global _tictoc_t0
    t1 = _clock() - _tictoc_t0
    nargin, nargout = _get_narginout(0)
    if nargout > 0:
        return t1
    print 'Elapsed time is %0.6f seconds.'%t1

class _m:
    """Magical helper constructor for all OMPC objects.
    m_['abc'] -> mstring
    m_[1:10] -> mslice
    m_[[1,2,3], 1:10] -> mcat([[1,2,3], mslice[1:10]]).
    """
    def __getitem__(self, i):
        # FIXME
        raise NotImplementedError()

m_ = _m()

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

from time import sleep
_pauseon = True
def pause(flag=None):
    global _pauseon
    if flag is None:
        if _pauseon: getch()
    elif isinstance(flag , str):
        if flag.lower() == 'on': _pausable = True
        elif flag.lower() == 'off': _pausable = False
        else: raise OMPCException('Unknown option %s"!'%flag)
    else:
        try: flag = float(flag)
        except: raise OMPCException(
                        'Parameter must be either a string or number!')
        sleep(flag)
