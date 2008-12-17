
import sys
from ompcply import translate_to_str

__all__ = ['mfunction', '_get_narginout', 'compile', 'addpath']

def mfunction_simple(names):
    def dec(func):
        from doc_comments import get_comment_doc
        doc = get_comment_doc(func)
        def func_new(*args):
            return func(*args)
        func_new.__doc__ = doc
        func_new.__name__ = func.__name__
        #func_new.__module__ = func.__module__
        return func_new
    return dec

from byteplay import Code, LOAD_GLOBAL, CALL_FUNCTION, \
                                   UNPACK_SEQUENCE, STORE_FAST, LOAD_FAST, \
                                   LOAD_CONST, BUILD_LIST, BINARY_SUBTRACT, \
                                   BINARY_MULTIPLY, BUILD_TUPLE, SLICE_2, \
                                   RETURN_VALUE

def _get_narginout(nargout_default=1):
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
        # MATLAB assumes at least 1 value
        return nargin, nargout_default
    return nargin, 1

class mfunction:
    """Decorator that allows emulation of MATLAB's treatement of functions.
    """
    def __init__(self, retstring):
        self._retvals = tuple( x.strip() for x in retstring.split(",") )
    
    def __call__(self, func):
        from byteplay import Code
        self._func = func
        self._c = Code.from_code(self._func.func_code)
        c = self._c
        
        # all return values must be initialized, for return (retvals)[:nargout]
        # not necessary anymore, the parser has to take care of this
        
        # check for maximum nargout, insert nargin nargout code
        self.__add_narg()
        # initialize novel variables, FIXME
        self.__add_return()
        self._func.func_code = self._c.to_code()
        return self._func
    
    def __add_narg(self):
        """Add nargin and nargout variables that emulate their behavior in
        MATLAB.
        """
        c = self._c
        pre_code = [(LOAD_GLOBAL,'_get_narginout'),
                    (CALL_FUNCTION,0),
                    (UNPACK_SEQUENCE, 2),
                    (STORE_FAST,'nargin'),
                    (STORE_FAST,'nargout')]
        
        # adjust the function preamble
        c.code[:0] = pre_code
        # replace LOAD_GLOBAL with LOAD_FAST for 
        for i, x in enumerate(c.code):
            if x[0] == LOAD_GLOBAL and \
                (x[1] == 'nargout' or x[1] == 'nargin'):
                c.code[i] = (LOAD_FAST,x[1])
    
    def __init_novel(self, names):
        """MATLAB allows assignment to variables that are not initialized.
        For example 'a(2) = 1;' would result in 'a = [0 1];'.
        This function takes a list of names as parameters and initializes
        all of them to the default empty marray().
        """
        return 
    
    def __add_return(self):
        # at first remove the original "return None"
        c = self._c
        c.code = c.code[:-2]
        #for i, x in enumerate(c.code):
        #    if x[0] == LOAD_CONST and x[1] is None:
        #        c.code[i] = (LOAD_FAST,x[1])
        
        postfix = []
        for name in self._retvals:                
            postfix += [(LOAD_FAST, name)]
        
        postfix.extend([(BUILD_TUPLE, len(self._retvals)),
                        (LOAD_FAST, 'nargout'),
                        (SLICE_2, None),
                        (RETURN_VALUE, None)])
        
        self._c.code.extend(postfix)

def get_pym_string(source):
    """get_pym_string(source) -> python_code_string
    
    Read MATLAB string in and return string that contains equivalent 
    Python compatible code that relies on features of OMPClib's mobject.
    This string can be further passed to Python's built-in compile method 
    with argument mode='exec' and executed by Python's interpreter.
    """
    return translate_to_str(source)

def get_pym_string_single(source):
    """get_pym_string_single(source) -> python_code_string
    
    Translate MATLAB single statement string into a functionally equivalent 
    Python compatible statement that relies on features of OMPClib's mobject.
    This string can be further passed to Python's built-in compile method 
    with argument mode='single' and executed by Python's interpreter.
    """
    return translate_to_str(source)

def get_pym_string_eval(source):
    """get_pym_string_eval(source) -> python_code_string
    
    Translate MATLAB single statement string inot a functionally equivalent 
    Python compatible statement that relies on features of OMPClib's mobject.
    This string can be further passed to Python's built-in compile method 
    with argument mode='single' and executed by Python's interpreter.
    """
    return translate_to_str(source)

PYM_TEMPLATE = """\
# This file was automatically translated by OMPC (http://ompc.juricap.com)

from ompc import *

%(pym_string)s

"""
def compile(source, filename, mode, flags=0, dont_inherit=0):
    """compile(source, filename, mode[, flags[, dont_inherit]]) -> code object

    Compile the source string (a MATLAB module, statement or expression)
    into a code object that can be executed by the exec statement or eval().
    The filename will be used for run-time error messages.
    The mode must be 'exec' to compile a module, 'single' to compile a
    single (interactive) statement, or 'eval' to compile an expression.
    
    The flags and dont_inherit arguments are ignored by OMPC at the moment 
    and are passed to the built-in compile method of Python."""
    import __builtin__
    # get the source code
    if mode == 'exec':
        pycode_str = get_pym_string(source)
    elif mode == 'single':
        pycode_str = get_pym_string_single(source)
    elif mode == 'eval':
        pycode_str = get_pym_string_eval(source)
    else:
        raise ValueError("compile() arg 3 must be 'exec' or 'eval' or 'single'")
    pycode_str = PYM_TEMPLATE%{"pym_string": pycode_str}
    pym = filename[:-1]+'pym'
    open(pym, 'wb').write(pycode_str)
    co = __builtin__.compile(pycode_str, pym, mode, flags, dont_inherit)
    # return the code object
    return co

import ihooks, imp, os
from os.path import join, exists, isdir, splitext, split
from glob import glob
from imp import PKG_DIRECTORY, PY_COMPILED, PY_SOURCE
import m_compile

M_COMPILABLE = ['.m']

def get_mfiles(path):
    """Function returns all MATLAB imoportable files in the 'path' folder.
    """
    return glob(join(path,'*.m'))

class MFileHooks(ihooks.Hooks):

    def load_source(self, name, filename, file=None):
        """Compile .m files."""
        if splitext(filename)[1] not in M_COMPILABLE:
            return ihooks.Hooks.load_source(self, name, filename, file)
        if file is not None:
            file.close()
        mfname = splitext(filename)[0]
        cfile = mfname + '.py' + (__debug__ and 'c' or 'o')
        m_compile.compile(filename, cfile)    # m-file compilation
        cfile = open(cfile, 'rb')
        try:
            print name, filename, cfile
            module = self.load_compiled(name, filename, cfile)
            #return module
            # Python at this point returns a module, we can actually return the
            # single function that is in it
            return getattr(module, name)
        finally:
            cfile.close()                
 
class MFileLoader(ihooks.ModuleLoader):
    """A hook to include .m files into importables and translate them 
    on-demand to .pym and .pyc files."""
    
    def load_module(self, name, stuff):
        """Special-case package directory imports."""
        file, filename, (suff, mode, type) = stuff
        path = None
        module = None
        if type == imp.PKG_DIRECTORY:
            # try first importing it as Python PKG_DIRECTORY
            stuff = self.find_module_in_dir("__init__", filename, 0)
            mfiles = get_mfiles(filename)
            if stuff is not None:
                file = stuff[0]             # package/__init__.py
                path = [filename]
            elif mfiles:
                # this is a directory with mfiles
                # create a new module and fill it with mfunctions
                module = imp.new_module(filename)
                for x in mfiles:
                    #setattr(module, x, load_mfunction(filename, x))
                    # all mfiles are considered source, the m2py compilation
                    # stuff is in the load_source function
                    mfunc_name = splitext(split(x)[1])[0]
                    mfile = ihooks.ModuleLoader.load_module(self, mfunc_name, 
                                    (open(x, 'U'), x, ('', '', PY_SOURCE)))
                    setattr(module, mfunc_name, getattr(mfile, mfunc_name, None))
        
        if module is None:
            try:                            # let superclass handle the rest
                module = ihooks.ModuleLoader.load_module(self, name, stuff)
            finally:
                if file:
                    file.close()
            if path:
                module.__path__ = path      # necessary for pkg.module imports
        return module
    
    def match_mfile(self, name, dir):
        """The function should look for all possible files in the 'dir' that 
        MATLAB would allow to call. This includes
            - .m, .dll, .mex ???? FIXME
        
        If the name is a directory the directory should be searched for all
        these files. If there are any importable files they should be loaded
        and presented to the importer behind a single module called 'name'.
        """
        path = join(dir, name)
        if exists(path) and isdir(path):
            # are there any mfiles
            mfiles = get_mfiles(path)
            if mfiles:
                # we have to load the mfiles ourselves create a module
                # with all the mfiles wrapped
                #from warnings import warn
                #warn("Importing directories not implemented yet!")
                return path
        elif exists(path+'.m'):
            return path+'.m'
    
    def find_module_in_dir(self, name, dir, allow_packages=1):
        if dir is None:
            # no documentation, dir=None maybe query the cache ???, TODO
            return ihooks.ModuleLoader.find_module_in_dir(
                self, name, dir, allow_packages)
        else:
            if allow_packages:
                resolved_path = self.match_mfile(name, dir)
                if resolved_path is not None:
                    if os.path.isdir(resolved_path):
                        return (None, resolved_path, ('', '', PKG_DIRECTORY))
                    else:
                        return (open(resolved_path,'rb'), resolved_path, ('', '', PY_SOURCE))
                        ## PY_COMPILED)) load_compiled would be called
            
            return ihooks.ModuleLoader.find_module_in_dir(
                self, name, dir, allow_packages)

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
        sys.path.insetr(pos, list(args))

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

import __main__
__main__.__dict__['addpath'] = addpath
__main__.__dict__['tic'] = tic
__main__.__dict__['toc'] = toc

def install():
    """Install the import hook"""
    ihooks.install(ihooks.ModuleImporter(MFileLoader(MFileHooks())))

install()

if __name__ == "__main__":
    pth = '../mfiles/'
    src = file(pth+'Uncertainty_function.m','U').read()
    print get_pym_string(src)
