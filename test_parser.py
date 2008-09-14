
import os, sys, re
from ompcply import lex, yacc, _print3000 as _print

def mysub(x):
    f, t = x.span()
    return 'x'*(t-f)

#data = open('../examples/Uncertainty_function.m', 'U').read()
#data = open('../examples/adaptive_exp_weibull_1PATCH.m', 'U').read()
#data = open('../examples/TestPython_GammaData.m', 'U').read()
#data = open('../examples/guidemo/sldemo.m', 'U').read()
data = open('../examples/rat.m', 'U').read()
data = open('../examples/octave_rat.m', 'U').read()
com = ''
d = []
for x in data.split('\n'):
    # skip empty statements
    if not x.strip():
        _print()
        continue
    # remove comments
    x2 = re.sub(r"'((?:''|[^\n'])*)'", mysub, x)
    pos = list(re.finditer(r'\s*%.*', x2))
    if pos:
        pos = pos[0].start()
        com = x[pos:].replace('%', '#', 1)
        x = x[:pos]
        if not x.strip(): com = com.lstrip()
    if x.strip().endswith('...'):
        d += [ x ]
        continue
    else:
        d = [ x ]
    yacc.myparse(''.join(d) + '\n')
    _print(com)
    com = ''
    d = []

# try the whole thing at once
#yacc.parse(data)
