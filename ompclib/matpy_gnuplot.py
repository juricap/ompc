
from matpy import *

import gplot, gplot.funcutils
_g = gplot.Gnuplot()
gnuplot_has_pm3d = True

class _seti:
    def __getitem__(self,x):
        if type(x) is slice:
            s,e,t = (x.start is not None and x.start or 1)-1,x.step-1,x.stop is None and 1 or x.stop
            if round(t) != t:
                raise Exception('The step must be integer value!')
            e += 1e-6*t
            return arange(s,e,t)
        return array(x)-1

mri_ = _seti()

class _setitem:
    def __init__(self,ibase=0):
        self.__ibase = ibase
    def __getitem__(self,x):
        if type(x) is slice:
            step = x.stop or 1.
            if x.start is None:
                return arange(1,x.step+0.1*step,x.stop)
            return arange(x.start,x.step+0.1*step,x.stop)
        return array(x)

mr_ = _setitem()

def _is_vector(x):
    sz = size(x)
    
    return len(sz) <= 2 and ( sz[0] == 1 or sz[1] == 1 )


def _is_matrix(x):
    sz = size(x)
    return len(sz) == 2 and sz[0] > 1 and sz[1] > 1

def columns(x):
    return size(x,2)

def rows(x):
    return size(x,1)

def loglogsurfc(*varargin):
    global gnuplot_has_pm3d
    if gnuplot_has_pm3d:
        _g.set_string("log x;")
        _g.set_string("log y;")
        _g.set_string("pm3d at s ftriangles hidden3d 100;")
        _g.set_string("style line 100 lt 5 lw 0.5;")
        _g.unset_string("hidden3d;")
        _g.unset_string("surf;")
    meshc(*varargin)
    
def surfc(*varargin):
    global gnuplot_has_pm3d
    if gnuplot_has_pm3d:
        _g.set_string("pm3d at s ftriangles hidden3d 100;")
        _g.set_string("style line 100 lt 5 lw 0.5;")
        _g.unset_string("hidden3d;")
        _g.unset_string("surf;")
    meshc(*varargin)

def meshc(*varargin):
  ## XXX FIXME XXX -- the plot states should really just be set
  ## temporarily, probably inside an unwind_protect block, but there is
  ## no way to determine their current values.
  nargin = len(varargin)
  if nargin == 1:
    x = varargin[0]
    z = x;
    if _is_matrix(z):
      _g.unset_string("key;");
      _g.set_string("hidden3d;");
      _g.set_string("style data lines;");
      _g.set_string("surface;");
      _g.set_string("contour;");
      _g.unset_string("parametric;")
      _g.set_string("view 60, 30, 1, 1;");
      _g.splot(z.T);
    else:
      error ("meshc: argument must be a matrix");
  elif nargin == 3:
    x,y,z = varargin[:3]
    if _is_vector(x) and _is_vector(y) and _is_matrix(z):
      xlen = length (x);
      ylen = length (y);
      if xlen == columns(z) and ylen == rows(z):
        zz = gplot.GridData(z,x,y,binary=1)
        _g.unset_string("key;");
        _g.set_string("hidden3d;");
        _g.set_string("data style lines;");
        _g.set_string("surface;");
        _g.set_string("contour;");
        _g.set_string("parametric;");
        _g.set_string("view 60, 30, 1, 1;");
        _g.splot(zz)
        _g.unset_string("parametric;")
      else:
        msg = "meshc: rows (z) must be the same as length (y) and";
        msg = sprintf("%s\ncolumns (z) must be the same as length (x)", msg);
        error (msg);
    elif _is_matrix (x) and _is_matrix(y) and _is_matrix (z):
      xlen = columns(z);
      ylen = rows (z);
      if xlen == columns (x) and xlen == columns (y) and \
          ylen == rows (x) and ylen == rows(y):
        len_ = 3 * xlen;
        zz = gplot.GridData(z,x[0,:],y[:,0],binary=1)
        _g.unset_string("key;");
        _g.set_string("hidden3d;");
        _g.set_string("data style lines;");
        _g.set_string("surface;");
        _g.set_string("contour;");
        _g.set_string("parametric;")
        _g.set_string("view 60, 30, 1, 1;");
        _g.splot(zz)
        _g.unset_string("parametric;")
      else:
        error ("meshc: x, y, and z must have same dimensions");
    else:
      error ("meshc: x and y must be vectors and z must be a matrix");
  else:
    usage ("meshc (z)");

if __name__ == "__main__":
    from msvcrt import getch
    X, Y = meshgrid(linspace(0,1,10),linspace(0,1,10))
    surfc(X, Y,rand(10,10))
    getch()
    surfc(X[[1,2],:],Y[:,0],rand(10,10))
    getch()