
def _print(*args,**kwargs):
    """Emulation of Py3k's print.
    """
    from sys import stdout
    sep = kwargs.get('sep',' ')
    of = kwargs.get('file',stdout)
    end = kwargs.get('end','\n')
    print >>of, sep.join(map(str,args)),
    print >>of, end

f_file = ( x for x in open('functions.txt') )

for x in f_file:
    if x.strip().startswith('['):
        # category

f.file.close()