#!/usr/bin/env python
##############################################################################
## Filename:      extractHost.py
## Version:       $Revision$
## Description:   Extract host spectrum from reference cube
## Author:        $Author$
## $Id$
##############################################################################

__version__ = "$Id$"
__author__ = "Y. Copin <y.copin@ipnl.in2p3.fr>"

from pySNIFS import SNIFS_cube
import numpy as N
from matplotlib.mlab import prctile
import pyfits
import optparse

def decipher2floats(parser, option):
    """Decipher option 'f1,f2'."""

    opt = eval("parser.values.%s" % option) # values has no __getattribute__
    try:
        f1,f2 = [ float(x) for x in opt.split(',') ]
    except:
        parser.error("Cannot decipher %s option '%s'" % (option,opt))

    return f1,f2

def sumSpectra(cube, idx):
    """Sum spectra (and variance) for nonzero idx."""

    i,j = idx.nonzero()
    nos = [ cube.get_no(jj,ii) for ii,jj in zip(i,j) ]
    s = N.array([ cube.spec(no) for no in nos ])
    v = N.array([ cube.spec(no, var=True) for no in nos ])
    spec = s.sum(axis=0)/len(nos)
    var = v.sum(axis=0)/(len(nos)**2)

    return spec,var

def dumpSpectrum(outname, y, hdr=None, **kwargs):
    """Dump array y as spectrum."""

    phdu = pyfits.PrimaryHDU(y, header=hdr)
    if hdr.get('EXTNAME') == 'E3D_DATA':
        del phdu.header['EXTNAME']
        phdu.header.update('CRVAL1', hdr['CRVALS'], after='NAXIS1')
        phdu.header.update('CDELT1', hdr['CDELTS'], after='CRVAL1')
        phdu.header.update('CRPIX1', hdr['CRPIXS'], after='CDELT1')
    elif hdr.get('NAXIS') == 3:
        phdu.header.update('CRVAL1', hdr['CRVAL3'], after='NAXIS1')
        phdu.header.update('CDELT1', hdr['CDELT3'], after='CRVAL1')
        phdu.header.update('CRPIX1', hdr['CRPIX3'], after='CDELT1')
    hdulist = pyfits.HDUList([phdu])
    # Add extension(s) if any
    for i,key in enumerate(kwargs):
        ima = pyfits.ImageHDU(kwargs[key], header=phdu.header, name=key.upper())
        hdulist.append(ima)
    hdulist.writeto(outname, clobber=True)


if __name__ == '__main__':

    usage = "usage: [%prog] [options] e3d_galaxy.fits"

    parser = optparse.OptionParser(usage, version=__version__)
    parser.add_option("-o", "--out", type="string",
                      help="Output host spectrum [host_X.fits]")
    parser.add_option("-r", "--range", type="string",
                      help="Reconstructed image wavelength range [%default]",
                      default='auto')
    parser.add_option("-s", "--sky", type="string",
                      help="Sky percentiles [%default]",
                      default="1,25")
    parser.add_option("-g", "--gal", type="string",
                      help="Galaxy percentiles [%default]",
                      default="75,99")
    parser.add_option("-p", "--plot", action='store_true',
                      help="Plot flag.")

    opts,args = parser.parse_args()
    if len(args)!=1:
        parser.error("No or too many arguments")
    else:
        cubename = args[0]

    try:
        pyfits.getval(cubename, 'EURO3D', 0) # Check for EURO3D keyword
        isE3D = True
        cube = SNIFS_cube(e3d_file=cubename)
    except:                     # Fits cube from DDT
        isE3D = False 
        cube = SNIFS_cube(fits3d_file=cubename)

    X = cube.e3d_data_header['CHANNEL'][0].upper()

    if opts.out is None:
        opts.out = "host_%c.fits" % X
    if opts.range=='auto':
        if X=='B':
            lmin,lmax = 3700,4100 # Around [OII] abd Balmer at z=0.01-0.03
        elif X=='R':
            lmin,lmax = 6400,6800 # Around Halpha at z=0.01-0.03
        else:
            raise ValueError("Unknown channel '%s'" % X)
    else:
        lmin,lmax = decipher2floats(parser, 'range')
        try:
            lmin,lmax = [ float(x) for x in opts.range.split(',') ]
        except:
            parser.error("Cannot decipher range option '%s'" % opts.range)
    try:
        csmin,csmax = [ float(x) for x in opts.sky.split(',') ]
    except:
        parser.error("Cannot decipher sky option '%s'" % opts.sky)
    try:
        cgmin,cgmax = [ float(x) for x in opts.gal.split(',') ]
    except:
        parser.error("Cannot decipher gal option '%s'" % opts.gal)

    # Image reconstruction
    print "Reconstructed image: %.1f,%.1f" % (lmin,lmax)
    ima = cube.slice2d([lmin,lmax]) # May includes NaN's

    # Selection on flux
    fsmin,fsmax,fgmin,fgmax = prctile(ima[N.isfinite(ima)], 
                                      p=(csmin,csmax,cgmin,cgmax))

    # Sky region
    skyIdx = (fsmin<=ima) & (ima<=fsmax)
    isky,jsky = skyIdx.nonzero()
    print "Sky [%.0f%%-%.0f%%]: %f,%f (%d spx)" % \
        (csmin,csmax,fsmin,fsmax,len(isky))
    # Galaxy region
    galIdx = (fgmin<=ima) & (ima<=fgmax)
    igal,jgal = galIdx.nonzero()
    print "Gal [%.0f%%-%.0f%%]: %f,%f (%d spx)" % \
        (cgmin,cgmax,fgmin,fgmax,len(igal))

    lbda = cube.lbda
    skySpec,skyVar = sumSpectra(cube, skyIdx) # Sky spectrum
    galSpec,galVar = sumSpectra(cube, galIdx) # Galaxy + sky spectrum
    resSpec = galSpec - skySpec               # Galaxy spectrum
    resVar = galVar + skyVar

    # Save host spectrum (along w/ variance)
    print "Host spectrum in '%s' (w/ VARIANCE extension)" % opts.out
    if isE3D:
        cubeHdr = pyfits.getheader(cubename, extname='E3D_DATA')
    else: 
        cubeHdr = pyfits.getheader(cubename)
    dumpSpectrum(opts.out, resSpec, hdr=cubeHdr, variance=resVar)

    if opts.plot:
        import matplotlib.pyplot as P

        fig = P.figure(figsize=(12,6))

        title = "%s [%s]" % (cube.e3d_data_header.get('OBJECT','Unknown'),
                             cube.e3d_data_header.get('FILENAME','Unknown'))
        ax = fig.add_subplot(1,2,1, title=title,
                             xlabel='I [spx]', ylabel='J [spx]')
        ax.imshow(ima, vmin=fsmin, vmax=fgmax, extent=(-7.5,7.5,-7.5,7.5))
        ax.plot(jsky-7,isky-7,'rs')
        ax.plot(jgal-7,igal-7,'bo')
        for i,j,no in zip(cube.i,cube.j,cube.no):
            ax.text(i-7,j-7,str(no), size='x-small',
                    horizontalalignment='center', verticalalignment='center')

        ax1 = fig.add_subplot(2,2,2, xlabel="Wavelength [A]")
        ax2 = fig.add_subplot(2,2,4, xlabel="Wavelength [A]", sharex=ax1)
        ax1.plot(lbda, galSpec, 'b-', label='Galaxy+sky')
        ax1.plot(lbda, skySpec, 'r-', label='Sky')
        ax1.axvspan(lmin,lmax,fc='0.9',ec='0.8', label='_')
        ax1.legend()
        ax2.plot(lbda, resSpec, 'g-', label='Galaxy')
        ax2.legend()

        P.show()
