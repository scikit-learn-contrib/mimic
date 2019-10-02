#! /usr/bin/env python
"""A template for scikit-learn compatible packages."""

import codecs
import os
from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('mimic', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'mimiccalib'
DESCRIPTION = 'mimic is a calibration method in binary classification.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Pin-Ju Tien'
MAINTAINER_EMAIL = 'pinju.tien@gmail.com'
URL = 'https://github.com/scikit-learn-contrib/mimic'
DOWNLOAD_URL = 'https://github.com/scikit-learn-contrib/mimic'
LICENSE = 'new BSD'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'scikit-learn', 'matplotlib', 'pandas']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.5']

setup(name=DISTNAME,
      packages=find_packages(exclude=['tests*']),
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      # zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      install_requires=INSTALL_REQUIRES)
