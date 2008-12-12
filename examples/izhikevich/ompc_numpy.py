
from ompclib_numpy import *

def main():
    import os, sys
    sys.argv = sys.argv[1:]
    if (len(sys.argv) > 0):
        sys.path.insert(0, os.path.dirname(sys.argv[0]))
        import __main__
        dict = __main__.__dict__
        exec 'execfile(%r)' % (sys.argv[0],) in dict, dict

# When invoked as main program, invoke the profiler on a script
if __name__ == '__main__':
    main()
