
from google.appengine.api import users
from google.appengine.ext.webapp import template

class DjangoHandler:
    """A RequestHandler mixin that emulates, Django's render_to_response."""
    def render_to_response(self, tname, template_values):
        from os.path import join, dirname
        path = join(dirname(__file__),'..', tname)
        self.response.out.write(template.render(path, template_values))
    def get_status(self,user):
        if user:
            return 'Signed in as %s| <a href="%s">Logout</a>' \
                      %(user.nickname(),
                        users.create_logout_url(self.request.uri))
        else:
            return 'Anonymous guest | <a href="%s">Sign in</a>' \
                      %users.create_login_url(self.request.uri)

def path2params(name, pth, num=None):
    res = None
    try: res = pth.strip('/').split('/')
    except: res = pth
    if res[0] == name: res.pop(0)
    if num == 1: return res[0]
    elif len(res) < num: res += [None]*(num-len(res))
    return res[:num]
