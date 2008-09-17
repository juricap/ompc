
import os, sys, re
from src.ompcply import lex, yacc, _gettabs, _print3000, _reset

import wsgiref.handlers, logging
from google.appengine.api import users
from google.appengine.ext import webapp
from src.functions import path2params, DjangoHandler
from google.appengine.ext import webapp
from google.appengine.ext import db
from google.appengine.ext.webapp.util import run_wsgi_app
from google.appengine.api import mail

def mysub(x):
    f, t = x.span()
    return 'x'*(t-f)

def m2py(data, outfile):
    from src.ompcply import translate
    _reset()
    translate(data, outfile)
        
class M2PyHandler(webapp.RequestHandler, DjangoHandler):
    def get(self):
        user = users.get_current_user()
        d = {'status':self.get_status(user)}
        self.render_to_response('templates/m2py.html', d)

    def post(self):
        f = self.request.get('mfile')
        t = self.request.get('mtext')
        c = self.request.get('ctext') == 'on'
        user = users.get_current_user()
        d = {'status':self.get_status(user), 'message': ''}
        content = f or t or None
        if len(content) > 5000:
            d['message'] = """<p>Your submission was truncated to 5,000 characters.
            If your intentions are not bad and there is a possibility that 
            the error you are aubmitting is beyond the 5,000 characters 
            please copy/paste the problematic section directly into 
            the textarea below.</p>"""
            mail.send_mail(sender="juricap@gmail.com",
              to="juricap@juricap.com",
              subject="ompc misuse",
              body="somebody (%s) tried to upload %d bytes."%(user, len(content)))
            logging.error("somebody (%s) tried to upload %d bytes."%(user, len(content)))
            
        if not content:
            self.render_to_response('templates/m2py.html', d)
        else:
            from StringIO import StringIO
            fout = StringIO()
            #logging.info(repr(f))
            log = CompileLog(owner=user, content=content[:5000])
            try:
                m2py(content, fout)
                log.ompc_converted = True
                #d['message'] += '<p>OMPC processed the file without error.</p>'
            except:
                log.ompc_converted = False
                logging.error("Couldn't convert M code")
                d['message'] += "<p>OMPC didn't like some of your syntax.</p>"
            
            py_content = fout.getvalue()
            try:
                co = compile(py_content, '__main__', 'exec')
                log.py_parsed = True
                #d['message'] += '<p>Python liked the OMP.</p>'
            except SyntaxError:
                log.py_parsed = False
                logging.error("Couldn't parse Python code")
                d['message'] += "<p>OMPC generated Python code that Python doesn't like.</p>"
            d['message'] += "<p>Thank you, you are making OMPC better, unless you've just uploaded some garbage :-).</p>"

            log.put()
            
            if c:
                cout = StringIO()
                from src.colorpy import Parser
                p = Parser(py_content, cout)
                d['style'] = p.getstyle()
                d['source'] = p.getbody()
                self.render_to_response('templates/m2py.html', d)
            else:
                self.response.headers['Content-Type'] = 'text/plain'
                self.response.out.write(py_content)

class CompileLog(db.Model):
    owner = db.UserProperty()
    content = db.TextProperty(required=True)
    ompc_converted = db.BooleanProperty()
    py_parsed = db.BooleanProperty()

def main():
    application = webapp.WSGIApplication([('/m2py', M2PyHandler),
                                          ('/', M2PyHandler)],
                                         debug=True)
    run_wsgi_app(application)

if __name__=='__main__':
    main()
