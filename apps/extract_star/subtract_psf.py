#!/usr/bin/env python
# -*- coding: utf-8 -*-
##############################################################################
## Filename:      es_psf.py
## Version:       $Revision$
## Description:   Generate and subtract extract_star PSF cube
## Author:        $Author$
## $Id$
##############################################################################

"""Generate (and subtract) extract_star PSF cube from extracted spectrum"""

__author__ = "Y. Copin <y.copin@ipnl.in2p3.fr>"
__version__ = '$Id$'

import os
import numpy as N
from pySnurp import Spectrum
import pySNIFS
import libExtractStar as libES

SpxSize = 0.43      # Spaxel size in arcsec (check SNIFS_cube.spxSize)

if __name__ == '__main__':

    import optparse

    usage = "Usage: [%prog] [options] inspec.fits"
    parser = optparse.OptionParser(usage, version=__version__)

    parser.add_option("-r", "--ref", 
                      help="Reference datacube")
    parser.add_option("-s", "--sky",
                      help="Sky spectrum to be removed from output datacube")
    parser.add_option("-o", "--out",
                      help="Output point-source subtracted datacube")

    parser.add_option("-k", "--keep", action="store_true",
                      help="Save point-source datacube (with --psfname)", 
                      default=False)
    parser.add_option("--psfname", 
                      help="Name of point-source datacube [psf_refcube]", 
                      default=None)

    parser.add_option("-n", "--nosubtract", 
                      dest="subtract", action="store_false",
                      help="Do *not* subtract point-source from datacube",
                      default=True)

    opts,args = parser.parse_args()

    if not opts.ref:
        parser.error("Reference cube not specified")
    if opts.subtract and not opts.out:
        parser.error("Name for output point-source subtracted cube "
                     "not specified")
    if opts.psfname is None: # Default name for output PSF cube
        opts.psfname = 'psf_' + os.path.basename(opts.ref)
    else:                    # Assume that user wants to keep the PSF...
        opts.keep = True

    # Input spectrum
    print "Opening input spectrum %s" % args[0]
    spec = Spectrum(args[0])
    print spec

    # Reference/input cube
    print "Opening reference cube %s" % opts.ref
    try:                        # Try Euro3D
        cube = pySNIFS.SNIFS_cube(e3d_file=opts.ref)
        cube.writeto = cube.WR_e3d_file
        cubetype = "Euro3D"
    except ValueError:          # Try 3D
        cube = pySNIFS.SNIFS_cube(fits3d_file=opts.ref)
        cube.writeto = cube.WR_3d_fits
        cubetype = "3D"
    print "  %s, %d slices [%.2f-%.2f], %d spaxels" % \
        (cubetype, cube.nslice, cube.lstart, cube.lend, cube.nlens)

    # Check spectral samplings are coherent
    assert (spec.npts,spec.start,spec.step) == \
        (cube.nslice,cube.lstart,cube.lstep), \
        "Incompatible spectrum and reference cube"

    lrange = not spec._hdr.has_key('ES_LMIN') \
             and libES.get_slices_lrange(cube) or ()
    # Read PSF function name and parameters from spectrum header
    psf_fn = libES.read_psf_name(spec._hdr)
    psf_ctes = [cube.spxSize] + \
               libES.read_psf_ctes(spec._hdr, lrange) # [lref,aDeg,eDeg]
    psf_param = libES.read_psf_param(spec._hdr, lrange)

    cube.x = cube.i - 7         # x in spaxel
    cube.y = cube.j - 7         # y in spaxel
    model = psf_fn(psf_ctes, cube)

    # The PSF parameters are only the shape parameters. We set the
    # intensity of each slice to spectrum values
    psf = model.comp(N.concatenate((psf_param, N.ones(cube.nslice))), 
                     normed=True) # nslice,nlens
    sig = psf * spec.y.reshape(-1,1)
    var = (psf**2) * spec.v.reshape(-1,1)

    if opts.subtract:           # Save point-source subtracted cube
        cube.data -= sig
        cube.var += var
        if opts.sky:
            sky = Spectrum(opts.sky)
            if sky.readKey('ES_SDEG') >= 1:
                raise NotImplementedError('skyDeg>0 subtraction '
                                          'is not yet implemented')
            # from arcsec^-2 into spaxels^-1
            cube.data -= sky.y.reshape(-1,1) * SpxSize**2
            cube.var  += sky.v.reshape(-1,1) * SpxSize**4

        print "Saving point-source subtracted %s cube %s" % (cubetype, opts.out)
        cube.writeto(opts.out)

    if opts.keep:               # Save PSF cube
        cube.data = sig
        cube.var = var
        print "Saving point-source %s cube %s" % (cubetype, opts.psfname)
        cube.writeto(opts.psfname)
