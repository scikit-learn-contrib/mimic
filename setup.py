#! /usr/bin/env python
"""A template for scikit-learn compatible packages."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('mimic', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'mimic'
DESCRIPTION = 'mimic is a calibration method in binary classification.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Pin-Ju Tien'
MAINTAINER_EMAIL = 'pinju.tien@gmail.com'
URL = 'https://github.com/scikit-learn-contrib/mimic'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/scikit-learn-contrib/mimic'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'scikit-learn', 'matplotlib']
setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      # zip_safe=False,  # the package can run out of an .egg file
      # classifiers=CLASSIFIERS,
      # packages=find_packages(),
      install_requires=INSTALL_REQUIRES)
