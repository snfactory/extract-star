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

import pySNIFS
import numpy as N
from matplotlib.mlab import prctile
import pyfits
import optparse
import os

CLIGHT = 299792.458

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


class LinesOII:
    """[OII] doublet is described by 2 independant gaussians +
    background"""

    name = "LinesOII"
    l1 = 3726.03                        # [OII] doublet
    l2 = 3728.73

    def __init__(self, cube):

        self.npar_ind = 0
        self.npar_cor = 6               # 1+z,sigma,I1,I2,bkgnd(d=1)
        self.npar = self.npar_ind*cube.nslice + self.npar_cor
        self.l = N.reshape(cube.lbda,cube.data.shape)

    def comp(self, param):
        """[OII] (2G) + bkgnd(polyn(d=1))"""

        self.param = param
        zp1,s,i1,i2,b0,b1 = param
        val = i1 * N.exp(-0.5*( (self.l - self.l1*zp1)/s )**2) + \
              i2 * N.exp(-0.5*( (self.l - self.l2*zp1)/s )**2)
        val+= b0 + (self.l/self.l1 - 1)*b1 # Background

        return val


class LinesNIIHa:
    """[NII]+Halpha complex is described by 1 gaussian for Ha + 2 correlated
    gaussians for [NII] + background"""

    name = "LinesNIIHa"
    lHa = 6562.80                       # Halpha
    lNII1 = 6547.96                     # [NII]
    lNII2 = 6583.34                     # [NII]
    rNII = 0.340                        # [NII]1/[NII]2

    def __init__(self, cube):

        self.npar_ind = 0
        self.npar_cor = 6               # 1+z,sigma,I(Ha),I([NII]),bkgnd(d=1)
        self.npar = self.npar_ind*cube.nslice + self.npar_cor
        self.l = N.reshape(cube.lbda,cube.data.shape)

    def comp(self, param):
        """Halpha(G1) + [NII](G2,G3) + bkgnd(polyn(d=1))"""

        self.param = param
        zp1,s,iH,iN,b0,b1 = param
        val = iH * N.exp(-0.5*( (self.l - self.lHa*zp1)/s )**2) # Halpha
        val+= ( N.exp(-0.5*( (self.l - self.lNII1*zp1)/s )**2) * self.rNII +
                N.exp(-0.5*( (self.l - self.lNII2*zp1)/s )**2) ) * iN # [NII]
        val+= b0 + (self.l/self.lHa - 1)*b1   # Background

        return val


def errorband(ax, x, y, dy, color='b', alpha=0.3, label='_nolegend_'):

    #xp,yp = M.poly_between(x, y-dy, y+dy) # matplotlib.mlab
    xp = N.concatenate((x,x[::-1]))
    yp = N.concatenate(((y+dy),(y-dy)[::-1]))
    poly = ax.fill(xp, yp, fc=color, ec=color, alpha=alpha,
                   zorder=2, label=label)
    return poly

def addRedshiftedLines(ax, z):

    lines = [('[OII]',   (3726.03,3728.73)),
             ('[NeIII]', (3868.69,)),
             ('He',      (3970.07,)),
             ('Hd',      (4101.73,)),
             ('Hg',      (4340.46,)),
             ('[OIII]',  (4363.15,)),
             ('HeII',    (4685.74,)),
             ('Hb',      (4861.32,)),
             ('[OIII]',  (4958.83,5006.77)),
             ('[NI]',    (5197.90,5200.39)),
             ('[OI]',    (6300.20,)),
             ('[NII]',   (6547.96,6583.34)),
             ('Ha',      (6562.80,)),
             ('[SII]',   (6716.31,6730.68))]
    
    lmin,lmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    y0 = ymax - (ymax-ymin)/5
    for name,lbdas in lines:
        for i,l in enumerate(lbdas):
            l *= (1+z)
            if not lmin<l<lmax: continue
            ax.axvline(l, ymin=0.2,ymax=0.7, c='0.7', label='_', zorder=1)
            if i==0:
                ax.text(l,y0,name, size='x-small',
                        horizontalalignment='center',
                        verticalalignment='center',
                        rotation='vertical')
        

if __name__ == '__main__':

    usage = "usage: [%prog] [options] e3d_galaxy.fits"

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
    parser.add_option("--nofit", action='store_true',
                      help="Do not try to estimate redshift.")
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
        cube = pySNIFS.SNIFS_cube(e3d_file=cubename)
    except:                     # Fits cube from DDT
        isE3D = False 
        cube = pySNIFS.SNIFS_cube(fits3d_file=cubename)

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
                  'R':(6400,6800)}    # Around Halpha at z=0.01-0.03
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

    if not opts.nofit: # Estimate redshift from [OII] or Halpha/[NII]

        import pySNIFS_fit

        # Convert array to spectrum on a restricted range
        if X=='B':
            g = (3700<=lbda) & (lbda<=4200) # OII from 1+z=1 to 1.13
            l0 = lbda[g][resSpec[g].argmax()]
            print "Fit [OII] doublet in %.0f,%.0f A" % (l0-50,l0+50)
            g = ((l0-50)<=lbda) & (lbda<=(l0+50))
        elif X=='R':
            g = (6560<=lbda) & (lbda<=7400) # Ha from 1+z=1 to 1.13
            l0 = lbda[g][resSpec[g].argmax()]
            print "Fit [NII],Ha complex in %.0f,%.0f A" % (l0-100,l0+100)
            g = ((l0-100)<=lbda) & (lbda<=(l0+100))

        x = lbda[g]
        bkg = N.median(resSpec[g])
        norm = resSpec[g].max() - bkg
        y = (resSpec[g] - bkg) / norm
        v = resVar[g] / norm**2
        spec = pySNIFS.spectrum(data=y, var=v, start=x[0], step=cube.lstep)

        if X=='B':                      # [OII] doublet
            funcs = [ LinesOII.name, ]
            # 1+z,sigma,I1,I2,bkgnd(d=1)
            zp1 = x[resSpec[g].argmax()] / LinesOII.l2
            params = [ [zp1, 3, 0.5, 0.5, 0, 0] ]
            bounds = [ [[1.0,1.13],[2,5],[0,1],[0,1],[None,None],[None,None]]]
            myfunc = {LinesOII.name:LinesOII}

        elif X=='R':                    # [NII]+Halpha complex
            funcs = [ LinesNIIHa.name, ]
            # 1+z,sigma,I(Ha),I([NII]),bkgnd(d=1)
            zp1 = x[resSpec[g].argmax()] / LinesNIIHa.lHa
            params = [ [zp1, 4, 1, 0.5, 0, 0] ]
            bounds = [ [[1.0,1.13],[2,5],[0.1,2],[0,1],[None,None],[None,None]]]
            myfunc = {LinesNIIHa.name:LinesNIIHa}

        print "Initial guess:", params

        # Actual fit
        model = pySNIFS_fit.model(data=spec, func=funcs,
                                  param=params, bounds=bounds, myfunc=myfunc)
        model.fit(msge=False, deriv=False)
        model.khi2 *= model.dof             # True Chi2

        print "Adjusted parameters:", model.fitpar
        print "Chi2 (DoF=%d): %.2f" % (model.dof, model.khi2)

        #z = model.fitpar[0]/6562.80 - 1
        zsys = model.fitpar[0] - 1
        print "Estimated redshift: %.5f (%.2f km/s), Sigma: %f A" % \
            (zsys,zsys*CLIGHT,model.fitpar[1])

    makeMap = False and not opts.nofit
    if makeMap:

        zmap = N.zeros_like(ima) * N.nan # Redshift
        params = [ model.fitpar ] # Use global fit result as initial guess
        for ino,ii,ij in zip(cube.no,cube.i,cube.j):
            print "Spx #%03d at %02dx%02d:" % (ino,ii,ij),
            y = cube.spec(no=ino)[g]
            ibkg = N.median(y)
            inorm = y.max() - ibkg
            y = ( y - ibkg ) / inorm
            v = cube.spec(no=ino, var=True)[g] / inorm**2
            ispec = pySNIFS.spectrum(data=y, var=v, start=x[0], step=cube.lstep)
            imodel = pySNIFS_fit.model(data=ispec, func=funcs,
                                      param=params,bounds=bounds,myfunc=myfunc)
            imodel.fit(msge=False, deriv=False)
            imodel.khi2 *= imodel.dof   # True Chi2
            #print "   Fitpar:", imodel.fitpar
            z = imodel.fitpar[0] - 1
            print "Chi2 (DoF=%d)=%.2f, v=%+.2f km/s" % \
                (imodel.dof, imodel.khi2, (z-zsys)*CLIGHT)
            zmap[ii,ij] = z
        
        # Could now compute flux-weighted redshift

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

        ax1 = fig.add_subplot(2,2,2)
        ax2 = fig.add_subplot(2,2,4, xlabel="Wavelength [A]", sharex=ax1)
        ax1.plot(lbda, galSpec, 'b-', label='Galaxy+sky')
        ax1.plot(lbda, skySpec, 'r-', label='Sky')
        ax1.axvspan(lmin,lmax,fc='0.9',ec='0.8', label='_')
        ax1.legend()
        lgal, = ax2.plot(lbda, resSpec, 'g-', label='Galaxy')
        errorband(ax2, lbda, resSpec, N.sqrt(resVar), color='g')
        ax2.legend()
        ax2.set_xlim(lbda[0],lbda[-1])
        ax2.set_autoscale_on(False)

        if not opts.nofit:
            #ax2.plot(x, model.eval(model.flatparam)*norm + bkg, 'b-')
            ax2.plot(x, model.evalfit()*norm + bkg, 'r-')
            addRedshiftedLines(ax2, zsys)
            lgal.set_label("z=%.5f" % zsys) # Add redshift
            ax2.legend()

        if makeMap:
            fig2 = P.figure(figsize=(6,6))
            axv = fig2.add_subplot(1,1,1, title=title,
                                   xlabel='I [spx]', ylabel='J [spx]')
            # Velocity map
            vmap = (zmap - zsys)*CLIGHT # Convert redshift to velocity
            vmin,vmax = prctile(vmap[N.isfinite(zmap)], p=(3,97))
            imv = axv.imshow(vmap, vmin=vmin, vmax=vmax,
                             extent=(-7.5,7.5,-7.5,7.5))
            cbv = fig2.colorbar(imv, ax=axv, shrink=0.9)
            cbv.set_label('Velocity [km/s]')
            # Intensity contours
            axv.contour(ima, vmin=fsmin, vmax=fgmax, colors='k',
                        extent=(-7.5,7.5,-7.5,7.5))

        print "Saving figure in", figname
        fig.savefig(figname)
        P.show()

