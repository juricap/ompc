
import sys, string

REVISION = "$Revision: 0.1 $"
VERSION = REVISION.split()[1]

import sys
if sys.version_info[0] < 3:
    def _print(*args,**kwargs):
        """Emulation of Py3k's print.
        """
        from sys import stdout
        sep = kwargs.get('sep',' ')
        of = kwargs.get('file',stdout)
        end = kwargs.get('end','\n')
        print >>of, sep.join(map(str,args)),
        print >>of, end
else:
    # 2.x calls syntax error on _print = print 
    _print = eval('print')

_keywords = ["break", "case", "catch", "continue", "else", "elseif", "end", 
             "for", "function", "global", "if", "otherwise", "persistent", 
             "return", "switch", "try", "while"]

__debug__ = False

import imp
sys.modules['__ompc__'] = imp.new_module('__ompc__')
import __ompc__

def compile(source, filename, mode[, flags[, dont_inherit]])
    """compile(source, filename, mode[, flags[, dont_inherit]]) -> code object

    Compile the source string (a MATLAB module, statement or expression)
    into a code object that can be executed by the exec statement or eval().
    The filename will be used for run-time error messages.
    The mode must be 'exec' to compile a module, 'single' to compile a
    single (interactive) statement, or 'eval' to compile an expression.
    
    The flags and dont_inherit arguments are ignored at the moment and are
    preserved only for compatibility with Python's built-in compile function."""
    import __builtin__
    # get the source code
    if mode == 'exec':
        from ompclib.translator import get_pym_string
        pycode_str = get_pym_string(source)
    elif mode == 'single':
        from ompclib.translator import get_pym_string_single
        pycode_str = get_pym_string_single(source)
    elif mode == 'eval':
        from ompclib.translator import get_pym_string_eval
        pycode_str = get_pym_string_eval(source)
    else:
        raise ValueError("compile() arg 3 must be 'exec' or 'eval' or 'single'")
    co = __builtin__.compile(pycode_str, filename, mode, flags, dont)
    # return the code object
    return co
    
