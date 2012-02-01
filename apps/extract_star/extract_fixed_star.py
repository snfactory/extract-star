#!/usr/bin/env python 
# Version:     $Revision$
# Description: 
# Creator:     Rui Pereira <rui.pereira@in2p3.fr>
# Created:     27/01/2012 15:14
# CVS info:    $Id$
"""
3D fixed PSF/aperture extractor
"""

__author__ = 'Rui Pereira <rui.pereira@in2p3.fr>'
__version__ = '$Revision$'

import os
import numpy as N
from pySnurp import Spectrum
import pySNIFS
import libExtractStar as libES

if __name__ == '__main__':

    import optparse

    usage = "Usage: [%prog] [options] inspec.fits"
    parser = optparse.OptionParser(usage, version=__version__)

    parser.add_option("-c", "--cube",
                      help="Datacube")
    parser.add_option("-s", "--spec",
                      help="Reference spectrum for PSF parameters")
    parser.add_option("-o", "--out",
                      help="Output spectrum")
    parser.add_option("-m", "--method",
                      help="Extraction method (psf|optimal|aperture|subaperture) ['%default']",
                      default="psf")
    parser.add_option("-r", "--radius", type="float",
                      help="Aperture radius for non-PSF extraction " \
                           "(>0: in arcsec, <0: in seeing sigma) [%default]",
                      default=-5.)

    opts,args = parser.parse_args()

    # Input spectrum
    print "Opening input spectrum %s" % opts.spec
    spec = Spectrum(opts.spec)
    print spec

    # Reference/input cube
    print "Opening cube %s" % opts.cube
    try:                        # Try Euro3D
        cube = pySNIFS.SNIFS_cube(e3d_file=opts.cube)
        cubetype = "Euro3D"
    except ValueError:          # Try 3D
        cube = pySNIFS.SNIFS_cube(fits3d_file=opts.cube)
        cubetype = "3D"
    print "  %s, %d slices [%.2f-%.2f], %d spaxels" % \
        (cubetype, cube.nslice, cube.lstart, cube.lend, cube.nlens)

    # Check spectral samplings are coherent
    assert (spec.npts,spec.start,spec.step) == \
        (cube.nslice,cube.lstart,cube.lstep), \
        "Incompatible spectrum and cube"

    lrange = ()
    if not spec._hdr.has_key('ES_LMIN'):
        # get sliced cube
        nmeta = 12 # default for all old prods
        imin = 10                           # 1st slice [px]
        imax = cube.nslice - 10        # Last slice [px]
        istep = (imax-imin)//nmeta     # Metaslice thickness [px]
        imax = imin + nmeta*istep
        slices = [imin,imax,istep]
        try:
            slice_cube = pySNIFS.SNIFS_cube(e3d_file=opts.cube, slices=slices)
        except ValueError:
            slice_cube = pySNIFS.SNIFS_cube(fits3d_file=opts.cube, slices=slices)
        lrange = (slice_cube.lstart, slice_cube.lend)

    # Read PSF function name and parameters from spectrum header
    psf_fn = libES.read_psf_name(spec._hdr)
    psf_ctes = [cube.spxSize] + libES.read_psf_ctes(spec._hdr, lrange) # [lref,aDeg,eDeg]
    psf_param = libES.read_psf_param(spec._hdr, lrange)

    # Compute aperture radius
    if opts.method == 'psf':
        radius = None
    else:
        # TODO: seeing based aperture
        print 'go fetch the seeing somewhere!'
        radius = opts.radius < 0 and -opts.radius * spec.readKey('SEEING')/2.355 or opts.radius # [sigma] or [arcsec]

    lbda,spec,var = libES.extract_spec(cube, psf_fn, psf_ctes, psf_param,
                                       method=opts.method,
                                       radius=radius, verbosity=2)

    import IPython
    IPython.embed()
