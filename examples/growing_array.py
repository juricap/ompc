
import ompc

class mcellarray(list):
    def __setitem__(self,i,v):
        if i >= len(self):
            self.extend([None]*(i-len(self)) + [v])

m = mcellarray()
tic()
for i in xrange(100000): m[i] = 12
toc()
