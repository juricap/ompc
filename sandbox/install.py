
BB_URL = 'https://www.bitbucket.org/juricap/ompc/get/tip.gz'
import urllib, tarfile, os
zf = urllib.urlretrieve(BB_URL, 'ompc.tar.gz')

zf = tarfile.open(zf[0])
zf.extractall()
dname = zf.getnames()[0].split('/')[0]
zf.close()
