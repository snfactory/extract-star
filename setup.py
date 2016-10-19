#!/usr/bin/env python

from distutils.core import setup
import os

setup(name='extract_star',
      version='0.1.0.dev',
      description="Fit a 3-d PSF on a SNFactory data cube",
      author = "SNFactory",
      author_email="kylebarbary@gmail.com",
      packages = ['extract_star', 'extract_star.extern'],
      # note: there are more scripts in `scripts`.
      scripts=[os.path.join('scripts', 'extract-star')]
)
