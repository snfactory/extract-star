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

import os, re
import numpy as N
from pySnurp import Spectrum
import pySNIFS
import libExtractStar as libES

def read_psf_name(hdr):
    """Return PSF function name read (or guessed) from header."""

    assert hdr['ES_METH']=='psf', \
        "PSF reconstruction only works for PSF spectro-photometry"

    try:
        psfname = hdr['ES_PSF']
    except KeyError:
        efftime = hdr['EFFTIME']
        print "WARNING: cannot read 'ES_PSF' keyword, " \
            "guessing from EFFTIME=%.0fs" % efftime
        # Assert it's an old correlation PSF (i.e. 'long' or 'short')
        psfname = (efftime > 12.) and 'long' or 'short'
    
    # Convert PSF name (e.g. 'short red') to PSF function name
    # ('ShortRed_ExposurePSF')
    fnname = ''.join(map(str.capitalize,psfname.split())) + '_ExposurePSF'
    print "PSF name: %s [%s]" % (psfname, fnname)

    return fnname


def read_psf_ctes(hdr):
    """Read PSF constants [lbda_ref,alphaDeg,ellDeg] from header."""

    lref = hdr['ES_LREF']       # Reference wavelength [A]
    # Count up alpha/ell coefficients (ES_Ann/ES_Enn) to get the
    # polynomial degrees
    adeg = len([ k for k in hdr.keys() 
                 if re.match('ES_A\d+$',k) is not None ]) - 1
    edeg = len([ k for k in hdr.keys() 
                 if re.match('ES_E\d+$',k) is not None ]) - 1
    print "PSF constants: lref=%.2fA, alphaDeg=%d, ellDeg=%d" % (lref,adeg,edeg)

    return [lref,adeg,edeg]


def read_psf_param(hdr):
    """Read (7+ellDeg+alphaDeg) PSF parameters from header:
    delta,theta,x0,y0,PA,e0,...en,a0,...an."""

    airmass = hdr['ES_AIRM']    # Effective airmass
    parang = hdr['ES_PARAN']    # Effective parallactic angle [deg]
    delta = N.tan(N.arccos(1/airmass)) # ADR intensity
    theta = parang/57.295779513082323  # Parallactic angle [rad]

    x0 = hdr['ES_XC']           # Reference position [spx]
    y0 = hdr['ES_YC']
    pa = hdr['ES_XY']           # (Nearly) position angle

    # Polynomial coeffs in lr~ = lambda/LbdaRef - 1
    c_ell = [ v for k,v in hdr.items() if re.match('ES_E\d+$',k) is not None ]
    c_alp = [ v for k,v in hdr.items() if re.match('ES_A\d+$',k) is not None ]

    # Convert polynomial coeffs from lr~ = lambda/LbdaRef - 1 = a+b*lr
    # back to lr = (2*lambda + (lmin+lmax))/(lmax-lmin)
    lmin = hdr['CRVAL1']                      # Start
    lmax = lmin + hdr['NAXIS1']*hdr['CDELT1'] # End
    lref = hdr['ES_LREF']       # Reference wavelength [A]
    a = (lmin+lmax) / (2*lref) - 1
    b = (lmax-lmin) / (2*lref)
    ecoeffs = libES.polyConvert(c_ell, trans=(a,b), backward=True).tolist()
    acoeffs = libES.polyConvert(c_alp, trans=(a,b), backward=True).tolist()

    print "PSF parameters: airmass=%.3f, parangle=%.1fdeg, " \
        "refpos=%.2fx%.2f spx" % (airmass,parang,x0,y0)

    return [delta,theta,x0,y0,pa] + ecoeffs + acoeffs


if __name__ == '__main__':

    import optparse

    usage = "Usage: [%prog] [options] inspec.fits"
    parser = optparse.OptionParser(usage, version=__version__)

    parser.add_option("-r", "--ref", 
                      help="Reference datacube")
    parser.add_option("-o", "--out", 
                      help="Output point-source subtracted datacube")

    parser.add_option("-k", "--keep", action="store_true",
                      help="Save point-source datacube [psf_refcube]", 
                      default=False)
    parser.add_option("-n", "--nosubtract", 
                      dest="subtract", action="store_false",
                      help="Do *not* subtract point-source from datacube",
                      default=True)

    opts,args = parser.parse_args()

    if opts.subtract and not opts.out:
        parser.error("Name for output point-source subtracted cube "
                     "not specified")

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

    # Read PSF function name and parameters from spectrum header
    psf_fn = eval('libES.'+read_psf_name(spec._hdr))
    psf_ctes = [cube.spxSize] + read_psf_ctes(spec._hdr) # [lref,aDeg,eDeg]
    psf_param = read_psf_param(spec._hdr)

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
        print "Saving point-source subtracted %s cube %s" % (cubetype, opts.out)
        cube.writeto(opts.out)

    if opts.keep:               # Save PSF cube
        cube.data = sig
        cube.var = var
        outname = 'psf_' + os.path.basename(opts.ref)
        print "Saving point-source %s cube %s" % (cubetype, outname)
        cube.writeto(outname)

