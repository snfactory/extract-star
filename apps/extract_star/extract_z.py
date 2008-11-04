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

import pyfits
from pySnurp import Spectrum
from pySNIFS import spectrum as SNIFS_spectrum
import pySNIFS_fit
import numpy as N
from matplotlib.mlab import prctile
import optparse
import os

CLIGHT = 299792.458

def find_max(lbda, flux, lrange):

    lmin,lmax = lrange
    g = (lmin<=lbda) & (lbda<=lmax)

    return lbda[g][flux[g].argmax()]


class LinesBackground:
    """Polynomial background."""

    name = "LinesBackground"

    def __init__(self, params, cube):

        self.deg,self.lmin,self.lmax = params
        self.npar_ind = 0
        self.npar_cor = int(self.deg+1)
        self.npar = self.npar_ind*cube.nslice + self.npar_cor
        self.l = N.reshape(cube.lbda,cube.data.shape)
        self.x = (2*self.l - (self.lmin+self.lmax))/(self.lmax - self.lmin)
        self.parnames = [ 'b%d' % i for i in range(self.npar_cor) ]

    def comp(self, param):

        self.param = param
        # val = a0 + a1*x + a2*x**2 + ...
        val = self.param[-1]
        for par in self.param[-2::-1]:
            val = val*self.x + par
        return val

    def deriv(self, param):

        self.param = param
        # val = a0 + a1*x + a2*x**2 + ...
        grad = N.zeros((self.npar_cor,)+self.l.shape,'d')
        for i in range(self.npar_cor):
            grad[i] = self.x**i
        return grad


class LinesOII:
    """[OII] doublet is described by 2 independant gaussians"""

    name = "LinesOII"
    l1 = 3726.03                        # [OII] doublet
    l2 = 3728.73
    parnames = ['1+z','sigma','[OII]1','[OII]2']

    def __init__(self, cube):

        self.npar_ind = 0
        self.npar_cor = len(self.parnames) # 1+z,sigma,I1,I2
        self.npar = self.npar_ind*cube.nslice + self.npar_cor
        self.l = N.reshape(cube.lbda,cube.data.shape)

    def comp(self, param):

        self.param = zp1,s,i1,i2 = param
        val = i1 * N.exp(-0.5*( (self.l - self.l1*zp1)/s )**2) + \
              i2 * N.exp(-0.5*( (self.l - self.l2*zp1)/s )**2)
        return val

    def deriv(self, param):

        self.param = zp1,s,i1,i2 = param
        d1 = (self.l - self.l1*zp1) / s
        d2 = (self.l - self.l2*zp1) / s
        g1 = N.exp(-0.5*d1**2)
        g2 = N.exp(-0.5*d2**2)
        # val = i1*g1(zp1,s) + i2*g2(zp1,s)
        grad = N.zeros((self.npar_cor,)+self.l.shape,'d')
        grad[0] = i1*g1*self.l1*d1/s + i2*g2*self.l2*d2/s  # dval/dzp1
        grad[1] = i1*g1*d1**2/s + i2*g2*d2**2/s            # dval/ds
        grad[2] = g1                                       # dval/di1
        grad[3] = g2                                       # dval/di2

        return grad


class LinesNIIHa:
    """[NII],Halpha complex is described by 1 gaussian for Ha + 2
    correlated gaussians for [NII]."""

    name = "LinesNIIHa"
    lHa = 6562.80                       # Halpha
    lNII1 = 6547.96                     # [NII]1
    lNII2 = 6583.34                     # [NII]2
    rNII = 0.340                        # i[NII]1/i[NII]2
    parnames = ['1+z','sigma','Halpha','[NII]']

    def __init__(self, cube):

        self.npar_ind = 0
        self.npar_cor = len(self.parnames) # 1+z,sigma,I(Ha),I([NII])
        self.npar = self.npar_ind*cube.nslice + self.npar_cor
        self.l = N.reshape(cube.lbda,cube.data.shape)

    def comp(self, param):
        """Halpha(G1) + [NII](G2,G3)"""

        self.param = zp1,s,iH,iN = param
        val = iH * N.exp(-0.5*( (self.l - self.lHa*zp1)/s )**2) # Halpha
        val+= ( N.exp(-0.5*( (self.l - self.lNII1*zp1)/s )**2) * self.rNII +
                N.exp(-0.5*( (self.l - self.lNII2*zp1)/s )**2) ) * iN # [NII]
        return val

    def deriv(self, param):

        self.param = zp1,s,iH,iN = param
        dH = (self.l - self.lHa*zp1) / s
        d1 = (self.l - self.lNII1*zp1) / s
        d2 = (self.l - self.lNII2*zp1) / s
        gH = N.exp(-0.5*dH**2)
        g1 = N.exp(-0.5*d1**2)
        g2 = N.exp(-0.5*d2**2)
        # val = iH*gH(zp1,s) + iN*(r*g1(zp1,s) + g2(zp1,s))
        grad = N.zeros((self.npar_cor,)+self.l.shape,'d')
        grad[0] = iH * gH * self.lHa*dH/s + \
            iN * ( g1 * self.lNII1*d1/s * self.rNII + 
                   g2 * self.lNII2*d2/s ) # dval/dzp1
        grad[1] = iH * gH * dH**2/s + \
            iN * ( g1 * d1**2/s * self.rNII + 
                   g2 * d2**2/s )         # dval/ds
        grad[2] = gH                      # dval/diH
        grad[3] = g1*self.rNII + g2       # dval/diN

        return grad

    def flux(self, par):
        """Flux (and error) of Halpha line."""

        # par: 0:1+z, 1:sigma, 2:Halpha, 3:[NII]
        f = N.sqrt(2*N.pi)*par[1] * (par[2] + par[3]*(1+self.rNII))
        return f

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

def plot_correlation(ax, corr, parnames=None):

    im = ax.imshow(N.absolute(corr), 
                   norm=P.matplotlib.colors.LogNorm(),
                   origin='upper')
    if parnames:                # [['a','b'],['c','d'],...]
        names = reduce(lambda x,y:x+y, parnames) # ['a','b','c','d'...]
        names = [''] + names + ['']                # ???
        ax.set_xticklabels(names, rotation=45, size='smaller')
        ax.set_yticklabels(names, rotation=45, size='smaller')
    fig = ax.get_figure()
    cb = fig.colorbar(im, ax=ax, orientation='horizontal')
    cb.set_label("Correlation matrix")
    

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
    assert s.varname, "Input spectrum has no variance extension."
    resVar = s.v

    X = s.readKey('CHANNEL')[0].upper() # B or R
    obj = s.readKey('OBJECT', 'Unknown')
    filename = s.readKey('FILENAME', 'Unknown')
    flxunits = s.readKey('FLXUNITS', 'counts')
    print "Exposure %s: %s" % (filename, obj)

    # Convert array to pySNIFS.spectrum on a restricted range
    if X=='B':
        l0 = find_max(lbda, resSpec, (3700,4200)) # OII from 1+z=1 to 1.13
        print "Fit [OII] doublet around %.0f A (%.0f A window)" % (l0,100)
        lmin,lmax = l0-50,l0+50
        g = (lmin<=lbda) & (lbda<=lmax)
    elif X=='R':
        l0 = find_max(lbda, resSpec, (6560,7400)) # Ha from 1+z=1 to 1.13
        print "Fit [NII],Ha complex around %.0f A (%.0f A window)" % (l0,200)
        lmin,lmax = l0-100,l0+100
        g = (lmin<=lbda) & (lbda<=lmax)

    x = lbda[g]
    bkg = N.median(resSpec[g])
    norm = resSpec[g].max() - bkg
    y = (resSpec[g] - bkg) / norm
    v = resVar[g] / norm**2
    spec = SNIFS_spectrum(data=y, var=v, start=x[0], step=s.step)

    if X=='B':                      # [OII] doublet
        funcs = [ LinesOII.name, 
                  '%s;1,%f,%f' % (LinesBackground.name,lmin,lmax) ]
        # 1+z,sigma,I1,I2 + bkgnd(d=1)
        zp1 = x[resSpec[g].argmax()] / LinesOII.l2
        params = [ [zp1, 3, 0.5, 0.5], [0, 0] ]
        bounds = [ [[1.0,1.13],[2,5],[0,1],[0,1]],
                   [[None,None]]*2] # No constraints on background
        myfunc = {LinesOII.name:LinesOII, 
                  LinesBackground.name:LinesBackground}

    elif X=='R':                    # [NII]+Halpha complex
        funcs = [ LinesNIIHa.name, 
                  '%s;1,%f,%f' % (LinesBackground.name,lmin,lmax) ]
        # 1+z,sigma,IHa,I[NII] + bkgnd(d=1)
        zp1 = x[resSpec[g].argmax()] / LinesNIIHa.lHa
        params = [ [zp1, 4, 1, 0.5], [0, 0] ]
        bounds = [ [[1.0,1.13],[2,5],[0.1,2],[0,1]],
                   [[None,None]]*2] # No constraints on background
        myfunc = {LinesNIIHa.name:LinesNIIHa, 
                  LinesBackground.name:LinesBackground}

    # Actual fit
    model = pySNIFS_fit.model(data=spec, func=funcs,
                              param=params, bounds=bounds, myfunc=myfunc)

    parnames = [ model.func[i].parnames for i in range(len(funcs)) ]
    print "Adjusted parameters:", parnames
    print "Initial guess:", params

    model.fit(save=True, msge=False)
    model.khi2 *= model.dof             # True Chi2

    print "Status: %d, Chi2 (DoF=%d): %f" % \
          (model.status, model.dof, model.khi2)
    
    # Error computation
    hess = pySNIFS_fit.approx_deriv(model.objgrad, model.fitpar, order=5)
    cov = N.linalg.inv(hess)            # Covariance matrix
    dfitpar = N.sqrt(cov.diagonal())
    corr = cov/(dfitpar * dfitpar[:,N.newaxis]) # Correlation matrix

    print "Adjusted parameters (including normalization):"
    for par,val,dval in zip(reduce(lambda x,y:x+y, parnames),
                            model.fitpar, dfitpar):
        print "  %s = %f +/- %f" % (par,val,dval)

    # Detection level: flux(Ha) in units of sig(flux).

    zsys = model.fitpar[0] - 1
    dzsys = dfitpar[0]
    print "Estimated redshift: %f +/- %f (%.1f +/- %.1f km/s)" % \
        (zsys,dzsys,zsys*CLIGHT,dzsys*CLIGHT)
    print "Sigma: %.2f +/- %.2f A" % (model.fitpar[1],dfitpar[1])

    # Store results in input spectra (awkward way...)
    hdu = pyfits.open(specname, mode='update')
    hdu[0].header.update('CVSEXTZ',__version__)
    hdu[0].header.update('EXTZ_Z',zsys,"extract_z redshift")
    hdu[0].header.update('EXTZ_DZ',dzsys,"extract_z error on redshift")
    hdu[0].header.update('EXTZ_K2',model.khi2,"extract_z chi2")
    hdu[0].header.update('EXTZ_L',funcs[0],"extract_z lines")
    hdu.close()

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

    if opts.plot or os.environ.has_key('PYSHOW'):
        import matplotlib.pyplot as P

        fig = P.figure(figsize=(12,4))
        fig.subplots_adjust(left=0.075, right=0.95)

        title = "%s [%s]" % (obj, filename)
        ax1 = fig.add_subplot(1,2,1, title=title,
                              xlabel='Wavelength [A]',
                              ylabel='Flux [%s]' % flxunits)
        ax2 = fig.add_subplot(1,4,3, 
                              xlabel='Wavelength [A]')

        # Galaxy spectrum
        lgal, = ax1.plot(lbda, resSpec, 'g-', 
                         label="z=%.5f +/- %.1g" % (zsys,dzsys))
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

        sig = N.median(N.sqrt(resVar[g]))

        # Correlation matrix
        ax3 = fig.add_subplot(1,4,4)
        plot_correlation(ax3, corr, parnames=parnames)

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

        if opts.plot:
            print "Saving figure in", figname
            fig.savefig(figname)
        else:                   # PYSHOW
            P.show()
