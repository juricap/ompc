#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

# Based on the code from Jürgen Herman, the following changes where made:
#
# Mike Brown <http://skew.org/~mike/>:
# - make script callable as a CGI and a Apache handler for .py files.
#
# Christopher Arndt <http://chrisarndt.de>:
# - make script usable as a module
# - use class tags and style sheet instead of <style> tags
# - when called as a script, add HTML header and footer
#
# Peter Jurica <http://juricap.com>
# - make it less verbose
# - Gedit's oblivion, Vim's darskspectrum color scheme
# - highlight function name
#
# TODO:
#
# - parse script encoding and allow output in any encoding by using unicode
#   as intermediate

__version__ = '0.51'
__date__ = '2008-09-10'
__license__ = 'GPL'
__author__ = 'Peter Jurica, Jürgen Hermann, Mike Brown, Christopher Arndt'


import string, sys, cStringIO, cgi
import keyword, token, tokenize

_KEYWORD  = token.NT_OFFSET + 1
_TEXT     = token.NT_OFFSET + 2
_FUNCTION = token.NT_OFFSET + 3

_css_classes = {
    token.NUMBER:       'number',
    token.OP:           'operator',
    token.STRING:       'string',
    tokenize.COMMENT:   'comment',
    token.NAME:         'name',
    token.ERRORTOKEN:   'error',
    _KEYWORD:           'keyword',
    _TEXT:              'text',
    _FUNCTION:          'function',
}

STYLE_OBLIVION = """\
pre.code { font-style: Lucida,"Courier New"; background-color: #2e3436; }
    .line_number { color: #555753; }
    .number { color: #fce94f; }
    .operator { color: #ffffff; }
    .string { color: #fce94f; }
    .comment { color: #888a85; }
    .function { color: #729fcf; }
    .name { color: #eeeeec; }
    .error { color: red; border: solid 1.5pt #FF0000; }
    .keyword { color: #ffffff; font-weight: bold; }
    .text { color: #fce94f; }
"""

STYLE = STYLE_OBLIVION

template = """
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"
  "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
  <title>%(title)s</title>
  <meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
  <meta name="Generator" content="colorpy.py">
  <style type="text/css">
    %(style)s
  </style>
</head>
<body>
%(body)s
</body>
</html>
"""

class Parser:
    def __init__(self, raw, out=sys.stdout):
        self.raw = string.strip(string.expandtabs(raw))
        self.out = out
        self._function = False
        self._format()
    
    def _print(self, x):
        self.out.write( x )
    
    def getstyle(self):
        return STYLE
    
    def getbody(self):
        return self.out.getvalue()
    
    def _format(self):
        """ Parse and send the colored source.
        """
        # store line offsets in self.lines
        self.lines = [0, 0]
        pos = 0
        while 1:
            pos = string.find(self.raw, '\n', pos) + 1
            if not pos: break
            self.lines.append(pos)
        self.lines.append(len(self.raw))
        
        # parse the source and write it
        self.pos = 0
        text = cStringIO.StringIO(self.raw)
        self._print('<pre class="code">\n')
        try:
            tokenize.tokenize(text.readline, self._token_cb)
        except tokenize.TokenError, ex:
            msg = ex[0]
            line = ex[1][0]
            self._print("<h3>ERROR: %s</h3>%s\n" % (
                msg, self.raw[self.lines[line]:]))
        self._print('\n</pre>')
    
    def _token_cb(self, toktype, toktext, (srow,scol), (erow,ecol), line):      
        # calculate new positions
        oldpos = self.pos
        newpos = self.lines[srow] + scol
        self.pos = newpos + len(toktext)
        
        # handle newlines
        if toktype in [token.NEWLINE, tokenize.NL]:
            self._print('\n')
            return
        
        # send the original whitespace, if needed
        if newpos > oldpos:
            self._print(self.raw[oldpos:newpos])
        
        # skip indenting tokens
        if toktype in [token.INDENT, token.DEDENT]:
            self.pos = newpos
            return
        
        # map token type to a color group
        if token.LPAR <= toktype and toktype <= token.OP:
            toktype = token.OP
        elif toktype == token.NAME:
            if keyword.iskeyword(toktext):
                toktype = _KEYWORD
                if toktext in ['class', 'def']:
                    self._function = True
            elif self._function:
                toktype = _FUNCTION
                self._function = False
        css_class = _css_classes.get(toktype, 'text')
        
        # send text
        self._print('<span class="%s">' % (css_class,))
        self._print(cgi.escape(toktext))
        self._print('</span>')

def colorize_file(file=None, outstream=sys.stdout):    
    from os.path import basename
    source = open(file, 'U').read()
    filename = basename(file)
    from cStringIO import StringIO
    tout = StringIO()
    p = Parser(source, out=tout)
    outstream.write(template % \
        {'title': filename, 'style': STYLE_OBLIVION,
         'body' : tout.getvalue() })

if __name__ == "__main__":
    colorize_file(__file__)
