#!/usr/bin/env python
##############################################################################
## Filename:      extract_host.py
## Version:       $Revision$
## Description:   Extract host spectrum from reference cube
## Author:        $Author$
## $Id$
##############################################################################

"""Extract host galaxy spectrum from reference cube."""

__version__ = "$Id$"
__author__ = "Y. Copin <y.copin@ipnl.in2p3.fr>"

from pySNIFS import SNIFS_cube
import numpy as N
from matplotlib.mlab import prctile
import pyfits
import optparse
import os

def decipher2floats(parser, option):
    """Decipher option 'f1,f2'."""

    opt = eval("parser.values.%s" % option) # values has no __getattribute__
    try:
        return [ float(x) for x in opt.split(',') ]
    except:
        parser.error("Cannot decipher %s option '%s'" % (option,opt))


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
    if hdr.get('EXTNAME') == 'E3D_DATA': # E3d cube header
        del phdu.header['EXTNAME']
        phdu.header.update('CRVAL1', hdr['CRVALS'], after='NAXIS1')
        phdu.header.update('CDELT1', hdr['CDELTS'], after='CRVAL1')
        phdu.header.update('CRPIX1', hdr['CRPIXS'], after='CDELT1')
    elif hdr.has_key('CRVAL3'):         # Fits 3D cube header
        phdu.header.update('CRVAL1', hdr['CRVAL3'], after='NAXIS1')
        phdu.header.update('CDELT1', hdr['CDELT3'], after='CRVAL1')
        phdu.header.update('CRPIX1', hdr['CRPIX3'], after='CDELT1')
    hdulist = pyfits.HDUList([phdu])
    # Add extension(s) if any
    for i,key in enumerate(kwargs):
        ima = pyfits.ImageHDU(kwargs[key], name=key.upper())
        ima.header.update('CRVAL1', phdu.header['CRVAL1'])
        ima.header.update('CDELT1', phdu.header['CDELT1'])
        ima.header.update('CRPIX1', phdu.header['CRPIX1'])
        hdulist.append(ima)
    hdulist.writeto(outname, clobber=True)


def comp_histo(a, **kwargs):

    if 'bins' in kwargs:
        try:
            nbins = len(kwargs['bins'])
        except:
            nbins = kwargs['bins']
        print "Using default numpy histogram: nbins=%d" % nbins
        h,l = N.histogram(a, **kwargs)
    else:                               # Define optimal binning
        if 'range' in kwargs:
            vmin,vmax = kwargs['range']
        else:
            vmin,vmax = a.min(),a.max()
        # Freedman-Diaconis' choice for optimal bin width
        # (see http://en.wikipedia.org/wiki/Histogram)
        q1,q3 = prctile(a, p=(25.,75.))
        h = 2 * (q3-q1) / len(a)**(1./3.)
        nbins = round( (vmax-vmin)/h )
        print "Using Freedman-Diaconis histogram: nbins=%d" % nbins
        h,l = N.histogram(a, bins=nbins, **kwargs)

    h = N.concatenate((h,[h[-1]]))  # Complete h
    l = N.concatenate((l,[l[-1]+l[1]-l[0]])) # Not needed w/ new=True

    return h,l


def fill_hist(ax, x,y, **kwargs):
    """Fill in history plot."""

    # Compute hist plot coordinates
    n = len(x)
    xt = N.empty((2*n,), x.dtype)
    xt[0:-1:2], xt[1:-1:2], xt[-1] = x, x[1:], x[-1]
    yt = N.empty((2*n,), y.dtype)
    yt[0:-1:2], yt[1::2] = y, y

    # Fill in histogram
    #xs,ys = P.poly_between(xt, 0, yt)
    xs, ys = N.concatenate((xt,xt[::-1])),N.concatenate((yt,N.zeros_like(yt)))
    fl = ax.fill(xs, ys, **kwargs)

    return fl
        

if __name__ == '__main__':

    usage = "usage: [PYSHOW=1] %prog [options] e3d_galaxy.fits"

    parser = optparse.OptionParser(usage, version=__version__)
    parser.add_option("-o", "--out", type="string",
                      help="Output host spectrum [object_X.fits]")
    parser.add_option("-r", "--range", type="string",
                      help="Reconstructed image wavelength range [%default]",
                      default='auto')
    parser.add_option("-s", "--sky", type="string",
                      help="Sky percentiles [%default]",
                      default="1,25")
    parser.add_option("-g", "--gal", type="string",
                      help="Galaxy percentiles [%default]",
                      default="75,100")
    parser.add_option("-p", "--plot", action='store_true',
                      help="Plot flag.")

    opts,args = parser.parse_args()
    if len(args) != 1:
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
    obj = cube.e3d_data_header.get('OBJECT','unknown')
    filename = cube.e3d_data_header.get('FILENAME','unknown')
    if X not in ('B','R'):
        raise ValueError("Unknown channel '%s'" % X)
    if opts.out is None:
        path,name = os.path.split(cubename)
        opts.out = "host_" + name     # Includes extension .fits
    figname = os.path.splitext(opts.out)[0]+'.png'
    if opts.range=='auto':
        ranges = {'B':(3700,4100),    # Around [OII] and Balmer at z=0.01-0.03
                  'R':(5600,6800)}    # Almost sky-line free zone
        lmin,lmax = ranges[X]
    else:
        lmin,lmax = decipher2floats(parser, 'range')
    csmin,csmax = decipher2floats(parser, 'sky')
    cgmin,cgmax = decipher2floats(parser, 'gal')

    print "%s: %s [%s]" % (cubename, obj, filename)

    # Image reconstruction
    print "Reconstructed image: %.1f,%.1f" % (lmin,lmax)
    ima = cube.slice2d([lmin,lmax]) # May includes NaN's

    # Selection on flux
    fsmin,fsmax,fgmin,fgmax = prctile(ima[N.isfinite(ima)], 
                                      p=(csmin,csmax,cgmin,cgmax))

    # Sky region
    skyIdx = (fsmin<=ima) & (ima<=fsmax)
    isky,jsky = skyIdx.nonzero()
    print "Sky [%.0f%%-%.0f%%]: %g,%g (%d spx)" % \
        (csmin,csmax,fsmin,fsmax,len(isky))
    # Galaxy region
    galIdx = (fgmin<=ima) & (ima<=fgmax)
    igal,jgal = galIdx.nonzero()
    print "Gal [%.0f%%-%.0f%%]: %g,%g (%d spx)" % \
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
        cubeHdr = pyfits.getheader(cubename, 0)
    dumpSpectrum(opts.out, resSpec, hdr=cubeHdr, variance=resVar)

    if opts.plot:
        import matplotlib.pyplot as P

        fig = P.figure(figsize=(12,6))

        title = "%s [%s]" % (obj, filename)
        ax = fig.add_subplot(1,2,1, title=title,
                             xlabel='I [spx]', ylabel='J [spx]')
        ax.imshow(ima, vmin=fsmin, vmax=fgmax, extent=(-7.5,7.5,-7.5,7.5))
        ax.plot(jsky-7,isky-7,'rs', ms=3)
        ax.plot(jgal-7,igal-7,'bo', ms=3)
        #for i,j,no in zip(cube.i,cube.j,cube.no):
        #    ax.text(i-7,j-7,str(no), size='x-small',
        #            horizontalalignment='center', verticalalignment='center')

        ax0 = fig.add_subplot(3,2,2)
        ax1 = fig.add_subplot(3,2,4)
        ax2 = fig.add_subplot(3,2,6, xlabel="Wavelength [A]", sharex=ax1)

        # Flux histogram
        fx = ima[N.isfinite(ima)]
        h,l = comp_histo(fx)
        ax0.plot(l,h, color='g', ls='steps', label='_')
        fill_hist(ax0, l, h, fc='g', alpha=0.3)
        ax0.axvspan(fsmin,fsmax,fc='r', alpha=0.3) # Sky
        ax0.axvspan(fgmin,fgmax,fc='b', alpha=0.3) # Galaxy

        # Galaxy+sky and sky spectra
        ax1.plot(lbda, galSpec, 'b-', label='Galaxy+sky')
        ax1.plot(lbda, skySpec, 'r-', label='Sky')
        ax1.axvspan(lmin,lmax,fc='0.9',ec='0.8', label='_')
        ax1.legend()
        # Galaxy spectrum
        lgal, = ax2.plot(lbda, resSpec, 'g-', label='Galaxy')
        ax2.legend()
        ax2.set_xlim(lbda[0],lbda[-1])
        ax2.set_autoscale_on(False)

        print "Saving figure in", figname
        fig.savefig(figname)

        if os.environ.has_key('PYSHOW'):
            P.show()

