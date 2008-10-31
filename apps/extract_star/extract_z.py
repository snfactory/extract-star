#!/usr/bin/env python
##############################################################################
## Filename:      extract_z.py
## Version:       $Revision$
## Description:   Extract redshift from galaxy spectrum
## Author:        $Author$
## $Id$
##############################################################################

"""Extract redshift on galaxy spectrum from [OII] or Halpha/[NII]."""

__version__ = "$Id$"
__author__ = "Y. Copin <y.copin@ipnl.in2p3.fr>"

from pySnurp import Spectrum
from pySNIFS import spectrum as SNIFS_spectrum
import pySNIFS_fit
import numpy as N
from matplotlib.mlab import prctile
import optparse
import os

CLIGHT = 299792.458

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
    """Plot errorband between y-dy and y+dy."""

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

    usage = "usage: [PYSHOW=1] %prog [options] spec_galaxy.fits"

    parser = optparse.OptionParser(usage, version=__version__)
    parser.add_option("-p", "--plot", action='store_true',
                      help="Plot flag.")

    opts,args = parser.parse_args()
    if len(args) != 1:
        parser.error("No or too many arguments")
    else:
        specname = args[0]

    if opts.plot:
        path,name = os.path.split(specname)
        figname = 'z_'+os.path.splitext(name)[0]+'.png'

    s = Spectrum(specname)
    print s

    lbda = s.x
    resSpec = s.y
    assert s.hasVar, "Input spectrum has no variance extension."
    resVar = s.v

    X = s.readKey('CHANNEL')[0].upper() # B or R
    obj = s.readKey('OBJECT', 'Unknown')
    filename = s.readKey('FILENAME', 'Unknown')
    flxunits = s.readKey('FLXUNITS', 'counts')

    # Convert array to pySNIFS.spectrum on a restricted range
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
    spec = SNIFS_spectrum(data=y, var=v, start=x[0], step=s.step)

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
        raise NotImplementedError

        zmap = N.zeros_like(ima) * N.nan # Redshift
        params = [ model.fitpar ] # Use global fit result as initial guess
        for ino,ii,ij in zip(cube.no,cube.i,cube.j):
            print "Spx #%03d at %02dx%02d:" % (ino,ii,ij),
            y = cube.spec(no=ino)[g]
            ibkg = N.median(y)
            inorm = y.max() - ibkg
            y = ( y - ibkg ) / inorm
            v = cube.spec(no=ino, var=True)[g] / inorm**2
            ispec = SNIFS_spectrum(data=y, var=v, start=x[0], step=cube.lstep)
            imodel = SNIFS_fit.model(data=ispec, func=funcs,
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

        fig = P.figure(figsize=(12,4))
        fig.subplots_adjust(left=0.075, right=0.95)

        title = "%s [%s]" % (obj, filename)
        ax1 = fig.add_subplot(1,1.5,1, title=title,
                              xlabel='Wavelength [A]',
                              ylabel='Flux [%s]' % flxunits)
        ax2 = fig.add_subplot(1,3,3, 
                              xlabel='Wavelength [A]')

        # Galaxy spectrum
        lgal, = ax1.plot(lbda, resSpec, 'g-', label="z=%.5f" % zsys)
        #ax1.plot(x, model.evalfit()*norm + bkg, 'r-')
        addRedshiftedLines(ax1, zsys)
        ax1.set_xlim(lbda[0],lbda[-1])
        ax1.legend()

        # Zoom on adjusted line
        ax2.plot(x, resSpec[g], 'g-')
        errorband(ax2, x, resSpec[g], N.sqrt(resVar[g]), color='g', alpha=0.3)
        #ax2.plot(x, model.eval(model.flatparam)*norm + bkg, 'm-')
        ax2.plot(x, model.evalfit()*norm + bkg, 'r-')
        addRedshiftedLines(ax2, zsys)
        ax2.set_xlim(x[0],x[-1])

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

    if os.environ.has_key('PYSHOW'):
        P.show()
