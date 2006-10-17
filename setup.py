#!/usr/bin/env python
######################################################################
## Filename:      setup.py
## Version:       $Id$
## Description:   pySNIFS setup script
## Author:        Yannick Copin <ycopin@ipnl.in2p3.fr>
## Created at:    Thu Apr 13 13:47:49 2006
## Modified at:   Tue Oct 17 12:08:17 2006
## Modified by:   Yannick Copin <ycopin@ipnl.in2p3.fr>
######################################################################

from distutils.core import setup
from re import search
import sys, os

__author__ = '$Author$'
__version__ = '$Name$'

if not hasattr(sys, 'version_info') or sys.version_info < (2,3,0,'alpha',0):
    raise SystemExit, "Python 2.3 or later required to build pySNIFS."

def dolocal():
    """Adds a command line option --local=<install-dir> which is an
    abbreviation for 'put all of pySNIFS in <install-dir>/pySNIFS'."""
    if "--help" in sys.argv:
        print >>sys.stderr
        print >>sys.stderr, " options:"
        print >>sys.stderr, "--local=<install-dir>    same as --install-lib=<install-dir>"
    for a in sys.argv:
        if a.startswith("--local="):
            dir = a.split("=")[1]
            sys.argv.extend([
                "--install-lib="+dir,
                ])
            sys.argv.remove(a)


def main():
    dolocal()
    name = 'pySNIFS'
    try:
        # Taggued version: pySNIFS-M-m
        majmin = search('\$Name$' % name, __version__).groups()
        version = '.'.join(majmin)
    except AttributeError:
        # Developer's version
        version = 'developer $Revision$'
    
    setup(name = name,
          version = version,
          description = "SNIFS data handling and processing package",
          author = "E. Pecontal",
          author_email = "pecontal@obs.univ-lyon1.fr",
          platforms = ["Linux"],
          py_modules = ['pySNIFS', 'pySNIFS_fit'],
          package_dir={'':'lib'},
          scripts=[os.path.join('apps','extract_star','extract_star.py')],
          #packages=['pySNIFS','extract_star','cube_explorer'],
          #package_dir={'pySNIFS':'lib'}
          )


if __name__ == "__main__":
    main()

