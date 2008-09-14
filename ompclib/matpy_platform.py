
def _GetchUnix():
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def _GetchWindows():
    import msvcrt
    return msvcrt.getch()
    
def _GetchMacCarbon():
    import Carbon
    if Carbon.Evt.EventAvail(0x0008)[0]==0: # 0x0008 is the keyDownMask
        return ''
    else:
        (what,msg,when,where,mod)=Carbon.Evt.GetNextEvent(0x0008)[1]
        return chr(msg & 0x000000FF)

try:
    import msvcrt as _m
    getch = _GetchWindows
except:
    try:
        import tty
        getch = _GetchUnix
    except:
        try:
            import Carbon
            getch = _GetchMacCarbon
        except:
            print "No getch for you!"

if __name__ == '__main__': # a little test
    print 'Press a key'
    import sys
    for i in xrange(sys.maxint):
        k = getch()
        if k != '':
            break
    print 'you pressed ',k
###