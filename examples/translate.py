"""Usage: translate.py mfile.m

Uses OMPC online compiler at http://ompclib.appspot.com/m2py to translate 
MATLAB(R) n-files into Python compatible code.
"""

import sys
if len(sys.argv) < 2:
    print 
    sys.exit(0)

import urllib, httplib

fname = sys.argv[1]
mcode = file(fname, 'rb')
params = urllib.urlencode({'mtext': mcode.read()})
headers = {"Content-type": "application/x-www-form-urlencoded",
           "Accept": "text/plain"}
conn = httplib.HTTPConnection("ompclib.appspot.com")
conn.request("POST", "/m2py", params, headers)
response = conn.getresponse()

if response.status == 200:
    print >>sys.stderr, 'Server contacted'
else:
    print >>sys.stderr, 'Cannot connect to the server!'
    print >>sys.stderr, 'Response code:', response.status
    print >>sys.stderr, 'Reason:       ', response.reason

data = response.read()
print data
conn.close()
