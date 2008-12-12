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

def install():
    """Install the import hook"""
    ihooks.install(ihooks.ModuleImporter(MFileLoader(MFileHooks())))

install()

# don't export anything
__all__ = ['addpath']

if __name__ == "__main__":
    addpath('mfiles')
    import add
    print add(1,2)
    
    print add
    help(add)
