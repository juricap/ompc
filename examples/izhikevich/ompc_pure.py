
from ompclib_demo import *

try:
    import psyco
    psyco.full()
except:
    pass

def main():
    import os, sys
    sys.argv = sys.argv[1:]
    if (len(sys.argv) > 0):
        sys.path.insert(0, os.path.dirname(sys.argv[0]))
        import __main__
        dict = __main__.__dict__
        import time
        t0 = time.clock()
        exec 'execfile(%r)' % (sys.argv[0],) in dict, dict
        print 'Elapsed time is %0.6f seconds.'%(time.clock()-t0)

# When invoked as main program, invoke the profiler on a script
if __name__ == '__main__':
    main()
