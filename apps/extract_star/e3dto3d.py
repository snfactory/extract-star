#!/usr/bin/env python
##############################################################################
## Filename:      e3dto3d.py
## Version:       $Revision$
## Description:   Standard star spectrum extraction
## Author:        $Author$
## $Id$
##############################################################################

__author__ = "Y. Copin <y.copin@ipnl.in2p3.fr>"
__version__ = '$Id$'

import os
import optparse
import pySNIFS

usage = "usage: [%prog] [-o 3d.fits] euro3d.fits"

parser = optparse.OptionParser(usage, version=__version__)

parser.add_option("-o", "--out", type="string",
                  help="Output NAXIS=3 FITS cube.")

opts,args = parser.parse_args()
filename = args[0]

if not opts.out:
    opts.out = '3d_'+os.path.basename(filename)

cube = pySNIFS.SNIFS_cube(filename)
cube.WR_3d_fits(opts.out)
