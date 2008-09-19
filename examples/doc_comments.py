
def get_comment_doc(obj):
    """Get lines of comments immediately preceding an object's source code.
    Returns None when source can't be found.
    """
    from inspect import findsource, ismodule
    
    try:
        lines, lnum = findsource(obj)
    except (IOError, TypeError):
        return None
    
    if ismodule(obj):
        lnum = 0
        # Look for a comment block at the top of the file.
    start = lnum
    if lnum == 0:
        if lines and lines[0][:2] == '#!': start = 1
    else:
        start += 1
        if lines[start].lstrip()[:3] == 'def': start += 1
    
    while start < len(lines) and lines[start].strip() in ('', '#'):
        start = start + 1
    
    if start < len(lines) and lines[start].lstrip()[:1] == '#':
        comments = []
        end = start
        while end < len(lines) and lines[end].lstrip()[:1] == '#':
            comments.append(lines[end].strip()[2:])
            end = end + 1
        return '\n'.join(comments)

def mfunction(names):
    # from functools import update_wrapper
    def dec(func):
        doc = get_comment_doc(func)
        def func_new(*args):
            return func(*args, **kwargs)
        func_new.__doc__ = doc
        func_new.__name__ = func.__name__
        func_new.__module__ = func.__module__
        return func_new
    return dec
    
@mfunction("x")
def coinflip(ndraws=None, p=None):
    # Generate a list x of zeros and ones according to coin flipping (i.e.
    # Bernoulli) statistics with probability p of getting a 1.
    # 1/20/97  dhb  Delete obsolet rand('uniform').
    # 7/24/04  awi  Cosmetic.    
    
    # Generate ndraws random variables on the real interval [0,1).
    unif = rand(ndraws, 1)
    # Find all of the ones that are less than p.
    # On average, this proportion will be p.
    index = find(unif < p)
    [nones, m] = size(index)
    # Generate an array of zeros and then set
    # the ones found in the previous step to 1.
    x = zeros(ndraws, 1)
    if (nones != 0):    
        x(index).lvalue = ones(nones, m)    
    end

if __name__ == "__main__":
    help(coinflip)
