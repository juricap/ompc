#!/usr/bin/env python
#
# Copyright 2008 Peter Jurica. All Rights Reserved.
# See the file LICENCE.

'''The setup and build script for the python-twitter library.'''

__author__ = 'juricap@gmail.com'
__version__ = '1.0'


# The base package metadata to be used by both distutils and setuptools
METADATA = dict(
  name = "OMPC",
  version = __version__,
  author='Peter Jurica',
  author_email='juricap@gmail.com',
  long_description='MATLAB to Python syntax adapting comiler.',
  license='BSD License',
  url='http://ompc.juricap.com',
  keywords='ompc matlab numerical math science',
)

# Extra package metadata to be used only if setuptools is installed
SETUPTOOLS_METADATA = dict(
  include_package_data = True,
  classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: Apache Software License',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Communications :: Chat',
    'Topic :: Internet',
  ],
  test_suite = 'OMPC.suite',
)

def pysetup():
    # Build the long_description from the README and CHANGES
    METADATA['long_description'] = '\n'.join([open('README').read(),])

    # Use setuptools if available, otherwise fallback and use distutils
    try:
        import setuptools
        METADATA.update(SETUPTOOLS_METADATA)
        setuptools.setup(**METADATA)
    except ImportError:
        import distutils.core
        distutils.core.setup(**METADATA)


if __name__ == '__main__':
    pysetup()
