#!/usr/bin/env python
#
# Copyright 2008 Peter Jurica. All Rights Reserved.
# See the file LICENSE.

'''The setup and build script for OMPC.'''

__author__ = 'juricap@gmail.com'
__version__ = '1.0-beta'


from distutils.sysconfig import get_python_lib; print 

# The base package metadata to be used by both distutils and setuptools
METADATA = dict(
  name = "OMPC",
  version = __version__,
  author='Peter Jurica',
  author_email='juricap@gmail.com',
  long_description='MATLAB to Python syntax adapting compiler.',
  license='BSD License',
  url='http://ompc.juricap.com',
  keywords='ompc matlab numerical math science',
  packages=['ompc', 'ompclib', 'ompceg', ''],
  package_dir={'':'.'},
  package_data={'ompceg': ['test.*'],
                'ompc' : ['ompcply.test'],
                '': ['licenses/ply/*', 'test.py',
                     'ompc.cfg', 'LICENSE', 'README']},
  data_files=[(get_python_lib(), ['sandbox/ompc.pth'])],
)

def pysetup():
    # Build the long_description from the README and CHANGES
    METADATA['long_description'] = '\n'.join([open('README').read(),])
    # Use setuptools if available, otherwise fallback and use distutils
    try:
        import setuptools
        setuptools.setup(**METADATA)
    except ImportError:
        import distutils.core
        distutils.core.setup(**METADATA)

if __name__ == '__main__':
    pysetup()
