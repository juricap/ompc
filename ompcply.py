
import sys
sys.path += ['../outside/ply']
OCTAVE = True

# TODO
# - make the no';' printou an option, the output is ugly
# - add 1 2, or anything like that, everything after the NAME is considered 
#   a string
# - print -png ... and similar

_keywords = ["break", "case", "catch", "continue", "else", "elseif", "end", 
             "for", "function", "global", "if", "otherwise", "persistent", 
             "return", "switch", "try", "while"]

if OCTAVE:
    _keywords += ["endif", "endwhile", "endfunction", "endswicth", "endfor"]

# functions that are known to not return a value, this will make the
# resulting code prettier
_special = ['pause', 'plot', 'hold', 'axis', 'pcolor', 'colorbar',
            'pause', 'disp', 'colormap', 'set', 'title',
            'xlabel', 'ylabel']

reserved = dict( (x.lower(), x.upper()) for x in _keywords )

tokens = [
    'NAME', 'NUMBER', 'STRING',
    'COMMA', 'SEMICOLON', 'NEWLINE',
    'DOTTIMES', 'DOTDIVIDE', 'DOTPOWER',
    'NOTEQUAL', 'ISEQUAL', 'TRANS', 'CONJTRANS',
    'LESS', 'GREATER', 'LESSEQUAL', 'GREATEREQUAL',
    'AND', 'OR', 'NOT', 'ELOR', 'ELAND',
    'LBRACKET', 'RBRACKET', 'LCURLY', 'RCURLY', 'LPAREN', 'RPAREN',
    'LAMBDA',
    'COMMENT',
    ] + reserved.values()

literals = ['=', '+', '-', '*', '/', '^', ':', "'", '.']

states = (
    ('comment', 'exclusive'),
    ('inlist',  'inclusive'),
    ('inparen', 'inclusive'),
    )

# def t_comment(t):
#     r'%(.*)'
#     t.type = 'COMMENT'
#     t.value = '%s'%t.value[1:]
#     t.lexer.lineno += 1
#     return t

def t_LPAREN(t):
    r'\('
    t.lexer.push_state('inparen')
    return t

def t_inparen_END(t):
    'end'
    t.value = 'end'
    t.type = 'NUMBER'
    return t

def t_inparen_RPAREN(t):
    r'\)'
    t.lexer.pop_state()
    return t

def t_LBRACKET(t):
    r'\['
    t.lexer.push_state('inlist')
    return t

def t_inlist_RBRACKET(t):
    r'\]'
    t.lexer.pop_state()
    return t

def t_LCURLY(t):
    r'\{'
    t.lexer.push_state('inlist')
    return t

# cannot do this because [a(1,2) b] = min(1:4);
#def t_inlist_COMMA(t):
#    r','
#    t.type = 'LISTCOMMA'
#    return t

def t_inlist_RCURLY(t):
    r'\}'
    t.lexer.pop_state()
    return t

t_COMMA = ','
t_SEMICOLON = r';'
#def t_SEMICOLON(t):
#    ';'
    #t.lexer.begin('INITIAL')
    #t.lexer.push_state('reset')
#    return t

# Comments
def t_PERCENT(t):
    r'%'
    t.lexer.push_state('comment')

def t_comment_body(t):
    r'([^\n]+)'
    t.type = 'COMMENT'
    t.lexer.pop_state()
    return t

t_comment_ignore = '.*'

def t_comment_error(t):
    pass


# Tokens

t_DOTTIMES = r'\.\*'
t_DOTDIVIDE = r'\./'
t_DOTPOWER = r'\.\^'
t_NOTEQUAL = r'~='
t_ISEQUAL = r'=='
t_LESS = r'<'
t_GREATER = r'>'
t_LESSEQUAL = r'<='
t_GREATEREQUAL = r'>='
t_ELAND = r'&'
t_ELOR = '\|'
t_AND = r'&&'
t_OR = '\|\|'
t_NOT = '~'

def t_NAME(t):
    r'[a-zA-Z][a-zA-Z0-9_]*'
    t.type = reserved.get(t.value, 'NAME')    # Check for reserved words
    return t

t_LAMBDA = r'@'

t_TRANS = r"\.'"
t_CONJTRANS = r"'"

def t_STRING(t):
    r"'((?:''|[^\n'])*)'"
    pos = t.lexer.lexpos - len(t.value)
    if pos == 0:
        return t
    
    prec = t.lexer.lexdata[pos-1]
    if prec == '.':
        t.value = ".'"
        t.type = "TRANS"
        t.lexer.lexpos = pos + 2
    elif prec in ' \t[{(=;,\n':
        # it's a string, translate "''" to 
        t.value = t.value.replace("''", r"\'")
    else:
        t.value = "'"
        t.type = "CONJTRANS"
        t.lexer.lexpos = pos + 1
    return t

def t_NUMBER(t):
    r'(?:\d+\.\d*|\d*\.\d+|\d+)(?:[e|E]-?\d+|)'
    try:
        float(t.value)
    except ValueError:
        print "Is this really a float?", t.value
    return t

def t_COMMENT(t):
    r'%'
    global _comment
    _comment = t.value
    t.lexer.lineno += 1
    #pass
    # No return value. Token discarded

t_ignore = " \t"

def t_NEWLINE(t):
    r'\n'
    #t.type = 'COMMA'
    #t.value = 'NEWLINE'
    #return t
    #t.lexer.lineno += t.value.count("\n")
    pass

# one way of doing lists with spaces as separators
# it is actually wrong, the whitespace has to be dealt with by parser
# 1 + 1 is not 1,+,1
# def t_inlist_WHITESPACE(t):
#    r'\s+'
#    t.type = 'COMMA'
#    return t
# t_inlist_COMMA = r'\s*,\s*'

# semicolon has a different function inside of [] and {}
def t_inlist_SEMICOLON(t):
    r';'
    t.type = 'COMMA'
    t.value = 'SEMICOLON'
    return t
    #pass

def t_error(t):
    print "Illegal character '%s'" % t.value[0]
    t.lexer.skip(1)
    
# Build the lexer
import sys
# FIXME
sys.path += ['../outside']
import lex
lex.lex()

# Parsing rules

precedence = (    
    ('left', 'TRANS', 'CONJTRANS'),
    ('nonassoc', 'LESS', 'GREATER'),
    ('left', '+', '-'),
    ('left', '*', '/', 'DOTTIMES', 'DOTDIVIDE'),
    ('left', '^', 'DOTPOWER'),
    ('right', 'UMINUS', 'UPLUS'),
    )

# dictionary of names
names = { }
_key_stack = []
_switch_stack = []
_tabs = 0
_comment = None
TABSHIFT = 4

def _gettabs():
    global _tabs, TABSHIFT
    return ' '*_tabs

def _print3000(*args,**kwargs):
    """Emulation of Py3k's print.
    """
    from sys import stdout
    sep = kwargs.get('sep',' ')
    of = kwargs.get('file',stdout)
    end = kwargs.get('end','\n')
    of.write(sep.join(map(str,args)))
    of.write(end)
    
momo = open('momo','wt')
def _print(src):
    global momo
    ss = src.split('\n')
    print >>momo, ss
    for x in ss[:-1]:
        _print3000(' '*_tabs + x, sep='', end='\n')
    _print3000(' '*_tabs + ss[-1], sep='', end='')

def sign(x):
    if x >= 0: return 1
    return -1

class mvar:
    pass

class marray(mvar):
    def __call__(self,*args):
        print '--- Calling me with args:', args

class _mslice:
    def __getitem__(self,args):
        if type(args) is slice:
            s = sign(args.step)
            args = slice(args.start,args.step-s,args.stop)
        return marray(r_[args])

mslice = _mslice()
mcat = marray

class mcellarray(list):
    def __init__(self,l):
        list.__init__(self, l)
    def __setitem__(self,i,v):
        if i >= len(self):
            self.extend([None]*(i-len(self)) + [v])

__ompc_whos__ = { 'mcat': mcat, 'mslice': mslice, 'marray': marray }

def p_statement_list(p):
    '''statement_final : statement_list NEWLINE'''
    p[0] = p[1]

def p_statement_list(p):
    '''statement_list : statement
                      | statement COMMA
                      | statement SEMICOLON'''
    p[0] = _print_statement(p[1], len(p) > 2 and p[2] or None, p[0])

_lvalues = []
_knoend = list(_keywords)
_knoend.remove('end')
def _print_statement(x, send, p0):
    global _lvalues, _key_stack, _tabs
    #print '--------------------', x, send, p0
    finish = ''
    if p0 and p0.strip()[-1] not in ':;': finish = '; '
    res = x
    # don't print results of keyword statements and commands, FIXME
    xs = x.strip() and x.strip().split()[0]
    dedent = False
    if not xs:
        pass
    elif xs[0] == '@':
        assert len(_key_stack) == 1 and _key_stack[0] == 'function'
        _key_stack.pop()
        _tabs = TABSHIFT
        dedent = True
    elif xs in _keywords or xs in _special or \
         xs[:2] == '__' or xs in ['elif', 'else:']:
        if xs != 'end':
            dedent = True
    elif send is None or send == ',':
        # we need to print also the result
        if _lvalues:
            for lv in _lvalues:
                res += '; print %s'%lv
        #else:
        #    res = 'ans = %s; print ans'%res
    _lvalues = []
    if dedent: _tabs -=  TABSHIFT
    _print(finish+res)
    if dedent: _tabs +=  TABSHIFT
    return res

def p_statement_list2(p):
    '''statement_list : statement_list statement
                      | statement_list COMMA statement
                      | statement_list SEMICOLON statement'''
    #print 'quaaaaaaaaaaaaaa', map(str, p)
    p[0] = _print_statement(p[-1], len(p)>3 and p[2] or None, p[0])

def p_statement_expr(p):
    '''statement : expression'''
    global __ompc_whos__
    #print '----------- expr -> statement', map(str, p)
    p[0] = p[1]
#     #if __debug__:
#     #    exec(p[0], __ompc_whos__, locals())

def p_statement_function(p):
    '''statement : FUNCTION LBRACKET name_list RBRACKET "=" NAME LPAREN name_list RPAREN
                 | FUNCTION LBRACKET name_list RBRACKET "=" NAME
                 | FUNCTION NAME "=" NAME LPAREN name_list RPAREN
                 | FUNCTION NAME "=" NAME
                 | FUNCTION NAME LPAREN name_list RPAREN
                 | FUNCTION NAME'''
    global _tabs, _key_stack
    argout, fname, argin = None, None, None
    if '=' in p:
        if p[2] == '[':
            argout, fname = p[3], p[6]
            if '(' in p: argin = p[8]
        else:
            argout, fname = p[2], p[4]
            if '(' in p: argin = p[6]
    else:
        fname = p[2]
        if '(' in p: argin = p[4]
    p[0] = '@mfunction(%s)\ndef %s(%s):'%(argout, fname, argin)
    _key_stack.append('function')
    #_print(p[0])
    _tabs += TABSHIFT

def p_expression_lambda(p):
    '''expression : LAMBDA LPAREN name_list RPAREN expression'''
    p[0] = 'lambda %s: %s'%(p[3], p[5])

def p_expression_lambda(p):
    '''expression : LAMBDA NAME'''
    # function handle
    p[0] = p[1]

def p_expression_name_list(p):
    '''name_list : name_list COMMA NAME'''
    p[0] = '%s, %s'%(p[1], p[3])

def p_expression_name_list_2(p):
    '''name_list : NAME'''
    p[0] = p[1]

# def '''statement : CLASSDEF NAME'''
#    pass

# properties 
# methods
# events 
    
def p_statement_for(p):
    '''statement : FOR NAME "=" expression'''
    global _tabs, _key_stack
    p[0] = 'for %s in %s:'%(p[2], p[4])
    _key_stack.append('for')
    #_print(p[0])
    _tabs += TABSHIFT

def p_statement_while(p):
    '''statement : WHILE expression'''
    global _tabs, _key_stack
    p[0] = 'while %s:'%p[2]
    _key_stack.append('while')
    #_print(p[0])
    _tabs += TABSHIFT

def p_statement_if(p):
    '''statement : IF expression'''
    global _key_stack, _tabs
    p[0] = 'if %s:'%p[2]
    _key_stack.append('if')
    #_print(p[0])
    _tabs += TABSHIFT

def p_statement_elseif(p):
    '''statement : ELSEIF expression'''
    global _tabs, _key_stack
    # FIXME if p is cellarray we should copare with in
    p[0] = 'elif %s:'%p[2]
    assert _key_stack[-1] == 'if'
    _tabs -= TABSHIFT
    #_print(p[0])
    _tabs += TABSHIFT

def p_statement_else(p):
    '''statement : ELSE'''
    global _tabs, _key_stack
    # FIXME if p is cellarray we should copare with in
    p[0] = 'else:'
    assert _key_stack[-1] == 'if'
    _tabs -= TABSHIFT
    #_print(p[0])
    _tabs += TABSHIFT

def p_statement_break(p):
    """statement : BREAK"""
    p[0] = 'break'

def p_statement_continue(p):
    """statement : CONTINUE"""
    p[0] = 'continue'
    #_print(p[0])

def p_statement_return(p):
    """statement : RETURN"""
    p[0] = 'return'

def p_statement_switch(p):
    '''statement : SWITCH expression'''
    global _tabs, _key_stack, _switch_stack
    svar = '__switch_%d__'%len(_switch_stack)
    p[0] = '%s = %s\nif 0:\n%spass'%(svar, p[2], ' '*TABSHIFT)
    _key_stack.append('switch')
    _switch_stack.append( svar )
    _tabs += TABSHIFT
    #_print(p[0])

def p_statement_case(p):
    '''statement : CASE expression'''
    global _tabs, _key_stack, _switch_stack
    # FIXME if p is cellarray we should copare with in
    p[0] = 'elif %s == %s:'%(_switch_stack[-1], p[2])
    assert _key_stack[-1] == 'switch'
    #_tabs -= TABSHIFT
    #_print(p[0])
    #_tabs += TABSHIFT

def p_statement_otherwise(p):
    """statement : OTHERWISE"""
    global _key_stack
    p[0] = 'else:'
    assert _key_stack[-1] == 'switch'
    #_tabs -= TABSHIFT
    #_print(p[0])
    #_tabs += TABSHIFT

def p_statement_global(p):
    """statement : GLOBAL list_spaces"""
    p[0] = 'global %s'%p[2]
    #_print(p[0])

def p_statement_persistent(p):
    """statement : PERSISTENT list_spaces"""
    # FIXME, store in in a module or thread ???
    p[0] = 'global __persistent__'
    #p[0] += 'for _x in "%s".split(','): locals'%p[2]

#def command

def p_expression_list_space(p):
    '''list_spaces : list_spaces NAME'''
    # print 'kooooooooooooooool2', map(str, p)
    p[0] = '%s, %s'%(p[1], p[2])

def p_expression_list_space_2(p):
    '''list_spaces : NAME'''
    # print 'kooooooooooooooool1', map(str, p)
    p[0] = p[1]

def p_statement_list_spaces(p):
    '''statement2 : list_spaces NEWLINE
                  | list_spaces COMMA
                  | list_spaces SEMICOLON'''
    # print 'kooooooooooooooool3', map(str, p)
    p[0] = p[1]
    _print('%s()'%p[0])

def p_statement_try(p):
    '''statement : TRY'''
    global _tabs, _key_stack
    p[0] = 'try:'
    _key_stack.append('try')
    _tabs += TABSHIFT
    #_print(p[0])

def p_statement_catch(p):
    '''statement : CATCH'''
    # FIXME if p is cellarray we should copare with in
    global _tabs, _key_stack
    p[0] = 'except:'%(_switch_stack[-1], p[2])
    assert _key_stack[-1] == 'try'
    _tabs -= TABSHIFT
    #_print(p[0])
    _tabs += TABSHIFT


def p_statement_end(p):
    'statement : END'
    global _tabs, _key_stack, _switch_stack
    _tabs -= TABSHIFT
    p[0] = 'end'
    kw = _key_stack.pop()
    if kw == 'switch':
        _switch_stack.pop()
    #_print(p[0])

def _getname(lname):
    pos = lname.find('(')
    if pos == -1:
        pos = lname.find('{')
    if pos == -1:
        return lname
    return lname[:pos]

# def _getname(lname):
#     return lname[:lname.find('(')]

def p_statement_assign(p):
    '''statement : name_sub "=" expression
                 | name_attr "=" expression
                 | exprmcat "=" expression
                 | NAME "=" expression'''
    global __ompc_whos__, _lvalues
    #print '----------- assign -> statement', map(str, p)
    #if not names.has_key(p[1]):
    #    eval('%s = marray(%s)'%p[3])
    lname = p[1]
    if lname[0] == '[':
        # [...]
        ns = []
        for x in lname[1:-1].split(','):
            ln = _getname(x.strip())
            names[ln] = p[3]
            _lvalues += [ln]
    elif '(' in lname:
        p[1] = '%s.lvalue'%lname
        lname = _getname(lname)
        _lvalues = [lname]
        names[lname] = '%s'%p[3]
    else:
        names[lname] = '%s'%p[3]
        _lvalues = [lname]
    #print '------------- set lvalues to', _lvalues
    p[0] = '%s = %s'%(p[1], p[3])
    #_print(p[0])
    if __debug__:
        __ompc_whos__[lname] = p[3] #eval(p[3])
        #print __ompc_whos__
        #exec(p[0], __ompc_whos__, locals())

# def p_statement_with_comment(p):
#     '''statement : statement comment_text'''
#     p[0] = '%s    %s'%(p[1], p[2])

def p_statement_nogroup(p):
    """statement : NAME NAME
                 | NAME NUMBER"""
    # treating cases like "hold on, axis square"
    #print '---------jojo'
    p[0] = '%s("%s")'%(p[1], p[2])
    #_print(p[0])

def p_expr_list(p):
    '''exprlist : exprlist COMMA expression'''
    p[0] = '%s, %s'%(p[1], p[3])

def p_expr_list_2(p):
    'exprlist : expression'
    p[0] = p[1]

def p_expr_inlist(p):
    '''exprinlist : exprinlist COMMA expression
                  | exprinlist SEMICOLON expression
                  | exprinlist NEWLINE expression'''
    if p[2] in ['SEMICOLON', 'NEWLINE']:
    #if p[2] in ';\n':
        p[0] = '%s, OMPCSEMI, %s'%(p[1], p[3])
    else:
        p[0] = '%s, %s'%(p[1], p[3])

def p_expr_inlist2(p):
    '''exprinlist : exprinlist expression'''
    p[0] = '%s, %s'%(p[1], p[2])

def p_statement_empty(p):
    '''statement : empty'''
    p[0] = ''
    #_print('')

def p_expression_inlist_empty(p):
    "exprinlist : empty"
    p[0] = p[1]

def p_empty(p):
    "empty : "
    p[0] = ''

_pinlist = False
def p_expr_inlist_2(p):
    '''exprinlist : expression'''
    global _pinlist
    _pinlist = True
    p[0] = p[1]

def p_expression_binop(p):
    '''expression : expression '+' expression
                  | expression '-' expression
                  | expression '*' expression
                  | expression '/' expression
                  | expression '^' expression
                  | expression DOTTIMES expression
                  | expression DOTDIVIDE expression
                  | expression DOTPOWER expression
                  | expression NOTEQUAL expression
                  | expression ISEQUAL expression
                  | expression LESS expression
                  | expression GREATER expression
                  | expression LESSEQUAL expression
                  | expression GREATEREQUAL expression
                  | expression ELAND expression
                  | expression ELOR expression
                  | expression AND expression
                  | expression OR expression'''
    if p[2] == '+'  : p[0] = '%s + %s'%(p[1], p[3])
    elif p[2] == '-'  : p[0] = '%s - %s'%(p[1], p[3])
    elif p[2] == '*'  : p[0] = '%s * %s'%(p[1], p[3])
    elif p[2] == '/'  : p[0] = '%s / %s'%(p[1], p[3])
    elif p[2] == '^'  : p[0] = '%s ** %s'%(p[1], p[3])
    elif p[2] == '.*'  : p[0] = '%s *elmul* %s'%(p[1], p[3])
    elif p[2] == './'  : p[0] = '%s /eldiv/ %s'%(p[1], p[3])
    elif p[2] == '.^'  : p[0] = '%s **elpow** %s'%(p[1], p[3])
    # conditional and logical
    elif p[2] == '~='  : p[0] = '%s != %s'%(p[1], p[3])
    elif p[2] == '=='  : p[0] = '%s == %s'%(p[1], p[3])
    elif p[2] == '<'  : p[0] = '%s < %s'%(p[1], p[3])
    elif p[2] == '>'  : p[0] = '%s > %s'%(p[1], p[3])
    elif p[2] == '<='  : p[0] = '%s <= %s'%(p[1], p[3])
    elif p[2] == '>='  : p[0] = '%s >= %s'%(p[1], p[3])
    elif p[2] == '&'  : p[0] = 'logical_and(%s, %s)'%(p[1], p[3])
    elif p[2] == '|'  : p[0] = 'logical_or(%s, %s)'%(p[1], p[3])
    elif p[2] == '&&'  : p[0] = '%s and %s'%(p[1], p[3])
    elif p[2] == '||'  : p[0] = '%s or %s'%(p[1], p[3])

def p_expression_not(p):
    "expression : NOT expression"
    p[0] = 'not %s'%p[2]

def p_expression_uminus(p):
    "expression : '-' expression %prec UMINUS"
    p[0] = '-%s'%p[2]

def p_expression_uplus(p):
    "expression : '+' expression %prec UPLUS"
    p[0] = p[2]

def p_expression_group(p):
    "expression : LPAREN exprlist RPAREN"
    p[0] = '(%s)'%p[2]

def p_expression_empty_group(p):
    "expression : NAME LPAREN RPAREN"
    p[0] = '%s()'%p[1]

def p_expr_mcat(p):
    'expression : exprmcat'
    #if p[1] == '[]'
    p[0] = 'mcat(%s)'%p[1]
    
def p_expression_list(p):
    "exprmcat : LBRACKET exprinlist RBRACKET"
    global _pinlist
    _pinlist = False
    p[0] = '[%s]'%p[2]

def p_expression_cell(p):
    "expression : LCURLY exprinlist RCURLY"
    global _pinlist
    _pinlist = False
    p[0] = 'mcellarray([%s])'%p[2]

def p_expression_conjtranspose(p):
    'expression : expression CONJTRANS'
    p[0] = '%s.cT'%p[1]

def p_expression_transpose(p):
    'expression : expression TRANS'
    p[0] = '%s.T'%p[1]

def p_expression_string(p):
    "expression : STRING"
    p[0] = p[1]

def p_expression_indexflat(p):
    "indexflat : LPAREN ':' RPAREN"
    p[0] = '(mslice[:])'

def p_expr_flatslice(p):
    'expression : ":"'
    p[0] = 'mslice[:]'
    
def p_expression_sub_flat(p):
    "expression : NAME indexflat"
    p[0] = '%s%s'%(p[1], p[2])

def p_expression_sub(p):
    "name_sub : NAME LPAREN exprlist RPAREN"
    p[0] = '%s(%s)'%(p[1], p[3])

def p_name_attr2(p):
    "name_attr : name_attr '.' NAME"
    p[0] = '%s.%s'%(p[1], p[3])

def p_name_attr(p):
    "name_attr : NAME"
    p[0] = '%s'%p[1]

def p_expression_attr(p):
    "expression : name_attr"
    p[0] = p[1]

def p_expression_sub2(p):
    "name_sub : NAME LCURLY exprinlist RCURLY"
    p[0] = '%s(%s)'%(p[1], p[3])

def p_expression_items(p):
    "expression : name_sub"
    p[0] = '%s'%p[1]

def p_expression_slice(p):
    """slice : expression ':' expression ':' expression 
             | expression ':' expression"""
    if len(p) == 6:
        p[0] = '%s:%s:%s'%(p[1],p[3],p[5])
    else:
        p[0] = '%s:%s'%(p[1],p[3])

def p_expression_mslice(p):
    "expression : slice"
    p[0] = 'mslice[%s]'%p[1]

def p_expression_number(p):
    "expression : NUMBER"
    p[0] = p[1]

def p_expression_name(p):
    "expression : NAME"
    p[0] = p[1]
#     try:
#         p[0] = names[p[1]]
#     except LookupError:
#         print "Undefined name '%s'" % p[1]
#         p[0] = 0

# comments are done in preprocessing
# def p_comment(p):
#     "comment_text : COMMENT"
#     p[0] = '# %s'%p[1]
#     _print(p[0])

_more = False
def p_error(p):
    global _comment, _more, _pinlist
    if p:
        if p.value == 'NEWLINE' and _pinlist:
            _more = True
        else:
            print "Syntax error at '%s'" % p.value
            pass
    else:
        if _pinlist:
            _more = True
        else:
            print "Syntax error at EOF"

import yacc
yacc.yacc(debug=1)

xbuf = ''
def _myparse(x):
    global _more, xbuf
    ret = yacc.parse(xbuf + x)
    if _more:
        xbuf += x.strip()
        if not xbuf.endswith(';'):
            xbuf += ';'
        _more = False
    else:
        xbuf = ''
        more = False
    return ret

yacc.myparse = _myparse

def translate(source, fo=None):
    """MATLAB syntax parser and translator.
    
    translate(source[, fo=None])
    
    The function takes MATLAB source code as a string and translates this
    code into Python compatible syntax that relies on OMPClib. The function of
    the original script should be preserved.
    
    The only public function is the constructor.
    """
    py_str = ''
    fout = fo
    if fo is None:
        from cStringIO import StringIO
        foout = StringIO()
    fout.write( _ompcparse(source) )
    if fo is None:
        return fout.getvalue()
    return None

def main():
    """A simple console OMPC translator.
    """
    import sys
    LEXDEBUG = 0
    if len(sys.argv) > 1:
        LEXDEBUG = 1
    
    while 1:
        try:
            s = raw_input('ompc> ') + '\n'
        except EOFError:
            break
        if not s: continue
        if LEXDEBUG:
            # Tokenize
            lex.input(s)
            while 1:
                tok = lex.token()
                if not tok: break      # No more input
                print tok
        if s.strip():
            yacc.myparse(s)


__all__ = ['lex', 'yacc', 'translate']

if __name__ == "__main__":
    main()
