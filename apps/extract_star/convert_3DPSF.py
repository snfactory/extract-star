#!/usr/bin/env python
##############################################################################
## Filename:      convert_3DPSF.py
## Version:       $Revision$
## Description:   Convert 3D PSF simulated by GR to Euro3D cube
##                suitable for extract_star
## Author:        $Author$
## $Id$
##############################################################################

import numpy as N
import os, sys

# Add IFU_py_io python wrapper to path
wrapper = os.path.join(os.environ['IFU_PATH'],
                       "IFU_C_iolibs-%s" % os.environ['IFU_VERS'],
                       "python_wrapper")
sys.path.append(wrapper)

import IFU_py_io as IFU
import pyfits                           # getheader

def read_3Dcube(name, sigext=0, varext=1):
    """Read FITS cube (NAXIS=3) from file name, using extension sigext
    (varext) for signal (variance)."""

    IFU.set_user_dataformat()

    sig = IFU.IMAGE3D()
    IFU.open_cube(sig, name + '[%d]' % sigext, 'i')
    sig.signal = IFU.init_cube_array(sig)

    var = IFU.IMAGE3D()
    IFU.open_cube(var, name + '[%d]' % varext, 'i')
    sig.variance = IFU.init_cube_array(var)

    sig.lbda = N.arange(sig.nz)*sig.stepz + sig.startz

    # DO NOT IFU.close_cube sig and var before we dont need them anymore!
    return sig


def crop_PSF(psf, center=(15,15), hsize=(7,7), wrange=None):
    """Crop input PSF spatially (FoV centered on center of size 2*hsize+1) and
    spectrally (wrange=lmin,lmax)."""

    if wrange is not None:
        lmin,lmax = wrange
        good = ((psf.lbda>=lmin) & (psf.lbda<=lmax)).nonzero()[0].tolist()
        psf.lbda = psf.lbda[good]
        psf.signal = psf.signal[good]
        psf.variance = psf.variance[good]

    # Extract central part of PSF: PSF is centered on center, and we want a
    # final FoV of 2*hsize+1
    x0,y0 = center
    dx,dy = hsize
    psf.signal = psf.signal[:,y0-dy:y0+dy+1,x0-dx:x0+dx+1]
    psf.variance = psf.variance[:,y0-dy:y0+dy+1,x0-dx:x0+dx+1]


def create_E3Dcube(name, signal, lbda=(0,1), variance=None,
                   coords=None, spxSize=0.43, desc=None):
    """Create Euro3D cube name from signal array (and variance array if
    any). Wavelength axis is defined with lbda=start,step, and spatial
    coordinates from coords (or automatically with spxSize). Copy non-std
    keywords from desc."""

    nw,ny,nx = signal.shape
    ssig = N.asarray(signal).reshape(nw,-1)
    if variance is not None:
        assert variance.shape == signal.shape
        vvar = N.asarray(variance).reshape(nw,-1)
    else:
        vvar = N.zeros(ssig.shape)

    # Spaxels and coordinates
    group = IFU.GROUP()                 # Single group
    spx = IFU.SPAXEL()
    spx.group = 0

    if coords is not None:
        x,y = coords
        assert x.shape == y.shape == (ny,nx)
    else:
        x,y = N.meshgrid(N.arange(nx),N.arange(ny)) # (ny,nx)
        x = (x - nx//2) * spxSize
        y = (y[::-1] - ny//2) * spxSize

    xx = x.ravel()
    yy = y.ravel()

    start,step = lbda                   # Start and step of wavelength ramp

    # Create output Euro3D cube
    ifu_default_fmt = os.environ['IFU_DEFAULT_FMT']
    os.environ['IFU_DEFAULT_FMT'] = 'euro3d' # Switch to Euro3D
    IFU.set_user_dataformat()
    os.environ['IFU_DEFAULT_FMT'] = ifu_default_fmt # Switch back to previous
    
    cube = IFU.E3D_file()
    ident= ""
    units = ""
    IFU.create_E3D_file(cube, name, nw, start, step, IFU.FLOAT, ident, units)

    # Fill in spectra
    specId = N.arange(nx*ny)
    for i in specId:
        spx.specId = i
        spx.xpos,spx.ypos = xx[i],yy[i]

        sigspec = IFU.SPECTRUM()
        IFU.init_new_E3D_spec(cube, sigspec, -1, -1)

        varspec = IFU.SPECTRUM()
        IFU.init_new_E3D_spec(cube, varspec, -1, -1)
        
        for l in N.arange(nw): # Is it possible to set spectra from arrays?
            IFU.WR_spec(sigspec, l, ssig[l,i])
            IFU.WR_qspec(sigspec, l, 0)

            IFU.WR_spec(varspec, l, vvar[l,i])
            IFU.WR_qspec(varspec, l, 0)
            
        IFU.put_E3D_row(cube, i, sigspec, varspec, 1, spx, 1, group)

    if desc:
        # Copy non-std keywords from desc
        IFU.CP_non_std_desc(desc, cube)

    return cube


if __name__ == '__main__':

    for psfName in sys.argv[1:]:
        outName = 'e3d_' + psfName

    # Open IMAGE3D cube
    channel = os.path.splitext(outName)[0][-1] # Last character
    print "Opening PSF %s [%c]" % (psfName, channel)
    psf = read_3Dcube(psfName, sigext=0, varext=1)
    print "  Shape:", psf.signal.shape, "Start,step:", psf.startz, psf.stepz

    # Wavelength truncation
    if channel == 'B':
        lmin,lmax = 3301.06,5200.30
    elif channel == 'R':
        lmin,lmax = 5054.25,9994.23
    else:
        raise ValueError("Unknown channel %c" % channel)
    print "Truncate PSF to wavelength range %.2f,%.2f" % (lmin,lmax)

    crop_PSF(psf, wrange=(lmin,lmax))
    print "Final PSF shape:", psf.signal.shape

    print "Creating output cube %s [%c]" % (outName, channel)
    cube = create_E3Dcube(outName, psf.signal, lbda=(psf.startz,psf.stepz),
                          variance=psf.variance, coords=None, desc=psf)

    # Specific keywords (handled w/ pyfits)
    hdr = pyfits.getheader(psfName)
    if not hdr.has_key('EFFTIME'):
        # Copy it from EXPTIME
        print "Copying EFFTIME keyword from EXPTIME"
        IFU.WR_desc(cube, 'EFFTIME', IFU.FLOAT,1, hdr['EXPTIME'])

    IFU.close_E3D_file(cube)


