########################333
# from IPyhton/Shell.py
def get_tk():
    try: import Tkinter
    except ImportError: return None
    else:
        hijack_tk()
        r = Tkinter.Tk()
        r.withdraw()
        return r

def hijack_tk():
    def misc_mainloop(self, n=0): pass
    def tkinter_mainloop(n=0): pass
    
    import Tkinter
    Tkinter.Misc.mainloop = misc_mainloop
    #Tkinter.mainloop = tkinter_mainloop

_tk = get_tk()
import matplotlib
matplotlib.interactive(True)

def _init_test():
    exec execfile('test.py') in globals(), globals()

from threading import Thread
t = Thread(None, _init_test)
t.start()
_tk.mainloop()
print 'dsfds2'
t.join()
print 'dsfds'
