
import sys

from ompcply import _keywords, translate

def get_pym_string(source):
    """get_pym_string(source) -> python_code_string
    
    Read MATLAB string in and return string that contains equivalent 
    Python compatible code that relies on features of OMPClib's mobject.
    This string can be further passed to Python's built-in compile method 
    with argument mode='exec' and executed by Python's interpreter.
    """
    return translate(source)

def get_pym_string_single(source):
    """get_pym_string_single(source) -> python_code_string
    
    Translate MATLAB single statement string into a functionally equivalent 
    Python compatible statement that relies on features of OMPClib's mobject.
    This string can be further passed to Python's built-in compile method 
    with argument mode='single' and executed by Python's interpreter.
    """
    

def get_pym_string_eval(source):
    """get_pym_string_eval(source) -> python_code_string
    
    Translate MATLAB single statement string inot a functionally equivalent 
    Python compatible statement that relies on features of OMPClib's mobject.
    This string can be further passed to Python's built-in compile method 
    with argument mode='single' and executed by Python's interpreter.
    """

if __name__ == "__main__":
    pth = '../../examples/'
    src = file(pth+'octave_rat.m','U').read()
    get_pym_string(src)
