
import sys
from ompcply import translate_to_str

def mfunction(names):
    def dec(func):
        from doc_comments import get_comment_doc
        doc = get_comment_doc(func)
        def func_new(*args):
            return func(*args, **kwargs)
        func_new.__doc__ = doc
        func_new.__name__ = func.__name__
        func_new.__module__ = func.__module__
        return func_new
    return dec

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
# This file was automatically transated by OMPC (http://ompc.juricap.com)

from ompc import mfunction

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
    co = __builtin__.compile(pycode_str, 
                             filename[:-1]+'pym', mode, flags, dont_inherit)
    # return the code object
    return co

if __name__ == "__main__":
    pth = '../mfiles/'
    src = file(pth+'Uncertainty_function.m','U').read()
    print get_pym_string(src)
