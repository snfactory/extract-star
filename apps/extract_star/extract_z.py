#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import numpy as N
import pyfits

from pySnurp import Spectrum
from pySNIFS import spectrum as SNIFS_spectrum
from pySNIFS import SNIFS_cube
import pySNIFS_fit
from ToolBox.MPL import errorband

from matplotlib.mlab import prctile

import optparse
import os

CLIGHT = 299792.458             # km/s

def find_max(lbda, flux, lrange):

    lmin,lmax = lrange
    g = (lmin<=lbda) & (lbda<=lmax)

    if not lbda[g].any():
        raise ValueError("Reasearch range %.2f-%.2f incompatible with "
                         "wavelength domaine %.2f-%.2f" %
                         (lmin,lmax,lbda[0],lbda[-1]))

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

    def __str__(self):
        return "background [deg=%d]" % self.deg

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


class LinesOI:
    """[OI] night-sky line is described by 1 gaussian"""

    name = "[OI] night-sky line"
    l0 = 5577.34                        # [OI] night sky line [air]
    parnames = ['1+z','sigma','[OI]']

    def __init__(self, cube):

        self.npar_ind = 0
        self.npar_cor = len(self.parnames) # 1+z,sigma,I
        self.npar = self.npar_ind*cube.nslice + self.npar_cor
        self.l = N.reshape(cube.lbda,cube.data.shape)

    def __str__(self):
        return "[OI] line"

    def comp(self, param):

        self.param = zp1,sig,i = param
        return i * N.exp(-0.5*( (self.l - self.l0*zp1)/sig )**2)

    def deriv(self, param):

        self.param = zp1,sig,i = param
        d = (self.l - self.l0*zp1) / sig
        g = N.exp(-0.5*d**2)
        grad = N.zeros((self.npar_cor,)+self.l.shape,'d')
        grad[0] = i*g*self.l0*d/sig     # dval/dzp1
        grad[1] = i*g*d**2/sig          # dval/dsig
        grad[2] = g                     # dval/di

        return grad


class LinesOII:
    """[OII] doublet is described by 2 independant gaussians"""

    name = "[OII] doublet"
    l1 = 3726.03                        # [OII] doublet
    l2 = 3728.73
    parnames = ['1+z','sigma','[OII]1','[OII]2']

    def __init__(self, cube):

        self.npar_ind = 0
        self.npar_cor = len(self.parnames) # 1+z,sigma,I1,I2
        self.npar = self.npar_ind*cube.nslice + self.npar_cor
        self.l = N.reshape(cube.lbda,cube.data.shape)

    def __str__(self):
        return "[OII] doublet"

    def comp(self, param):

        self.param = zp1,sig,i1,i2 = param
        val = i1 * N.exp(-0.5*( (self.l - self.l1*zp1)/sig )**2) + \
              i2 * N.exp(-0.5*( (self.l - self.l2*zp1)/sig )**2)
        return val

    def deriv(self, param):

        self.param = zp1,sig,i1,i2 = param
        d1 = (self.l - self.l1*zp1) / sig
        d2 = (self.l - self.l2*zp1) / sig
        g1 = N.exp(-0.5*d1**2)
        g2 = N.exp(-0.5*d2**2)
        # val = i1*g1(zp1,sig) + i2*g2(zp1,sig)
        grad = N.zeros((self.npar_cor,)+self.l.shape,'d')
        grad[0] = i1*g1*self.l1*d1/sig + i2*g2*self.l2*d2/sig # dval/dzp1
        grad[1] = i1*g1*d1**2/sig + i2*g2*d2**2/sig           # dval/dsig
        grad[2] = g1                                          # dval/di1
        grad[3] = g2                                          # dval/di2

        return grad


class LinesNIIHa:
    """[NII],Halpha complex is described by 1 gaussian for Ha + 2
    correlated gaussians for [NII]."""

    name = "[NII]+Ha complex"
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

    def __str__(self):
        return "[NII],Ha complex"

    def comp(self, param):
        """Halpha(G1) + [NII](G2,G3)"""

        self.param = zp1,sig,iH,iN = param
        val = iH*N.exp(-0.5*( (self.l - self.lHa*zp1)/sig )**2) # Halpha
        val+= ( N.exp(-0.5*( (self.l - self.lNII1*zp1)/sig )**2) * self.rNII +
                N.exp(-0.5*( (self.l - self.lNII2*zp1)/sig )**2) ) * iN # [NII]
        return val

    def deriv(self, param):

        self.param = zp1,sig,iH,iN = param
        dH = (self.l - self.lHa*zp1) / sig
        d1 = (self.l - self.lNII1*zp1) / sig
        d2 = (self.l - self.lNII2*zp1) / sig
        gH = N.exp(-0.5*dH**2)
        g1 = N.exp(-0.5*d1**2)
        g2 = N.exp(-0.5*d2**2)
        # val = iH*gH(zp1,sig) + iN*(r*g1(zp1,sig) + g2(zp1,sig))
        grad = N.zeros((self.npar_cor,)+self.l.shape,'d')
        grad[0] = iH * gH * self.lHa*dH/sig + \
            iN * ( g1 * self.lNII1*d1/sig * self.rNII + 
                   g2 * self.lNII2*d2/sig ) # dval/dzp1
        grad[1] = iH * gH * dH**2/sig + \
            iN * ( g1 * d1**2/sig * self.rNII + 
                   g2 * d2**2/sig )         # dval/dsig
        grad[2] = gH                        # dval/diH
        grad[3] = g1*self.rNII + g2         # dval/diN

        return grad

    def flux(self, par, cov=None):
        """Flux (and error) of Halpha line."""

        # par: 0:1+z, 1:sigma, 2:Halpha, 3:[NII]
        f = N.sqrt(2*N.pi)*par[1] * (par[2] + par[3]*(1+self.rNII))
        if cov is not None:
            # Compute jacobian of f
            j = N.empty(3, dtype='d')
            j[0] = (par[2] + par[3]*(1+self.rNII))
            j[1] = par[1]
            j[2] = par[1] * (1+self.rNII)
            j *= N.sqrt(2*N.pi)
            c = cov[1:4,1:4]    # Select proper submatrix
            df = N.sqrt(N.dot(j, N.dot(c,j)))
            return f,df
        else:
            return f


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

def plot_correlation_matrix(ax, corr, parnames=None):

    npar = len(corr)
    im = ax.imshow(N.absolute(corr), 
                   norm=P.matplotlib.colors.LogNorm(),
                   origin='upper', extent=(-0.5,npar-0.5,-0.5,npar-0.5))
    if parnames:
        assert len(parnames)==npar
        ax.set_xticks(range(npar)) # Set the nb of ticks
        ax.set_xticklabels(parnames, rotation=45, size='smaller')
        ax.set_yticks(range(npar))
        ax.set_yticklabels(parnames, rotation=45, size='smaller')
    fig = ax.get_figure()
    cb = fig.colorbar(im, ax=ax, orientation='horizontal')
    cb.set_label("|Correlation matrix|")
    

if __name__ == '__main__':

    usage = "usage: [PYSHOW=1] %prog [options] spec|cube.fits"

    parser = optparse.OptionParser(usage, version=__version__)
    parser.add_option("-p", "--plot", action='store_true',
                      help="Plot flag.")
    parser.add_option("-e", "--emissionline", dest='line',
                      help="Emission line to be adjusted [%default]",
                      default='auto')

    opts,args = parser.parse_args()
    if len(args) != 1:
        parser.error("No or too many arguments")
    else:
        specname = args[0]

    try:
        pspec = Spectrum(specname)
        print pspec
        isSpec = True
        step = pspec.step
    except KeyError:
        cube = SNIFS_cube(specname)
        print "Cube %s: %d spx, %d px [%.2f-%.2f]" % \
              (specname, cube.nlens, cube.nslice, cube.lstart, cube.lend)
        isSpec = False
        step = cube.lstep

    if isSpec:                          # Input is a Spectrum
        lbda = pspec.x
        resSpec = pspec.y
        assert pspec.varname, "Input spectrum has no variance extension."
        resVar = pspec.v

        X = pspec.readKey('CHANNEL','X')[0].upper() # B or R
        obj = pspec.readKey('OBJECT', 'Unknown')
        filename = pspec.readKey('FILENAME', 'Unknown')
        flxunits = pspec.readKey('FLXUNITS', 'counts')
    else:                               # Input is a Cube
        lbda = cube.lbda
        resSpec = cube.data.mean(axis=1)
        resVar = N.sqrt(cube.var.sum(axis=1)/cube.nlens**2)
        
        X = cube.e3d_data_header.get('CHANNEL','X')[0].upper() # B or R
        obj = cube.e3d_data_header.get('OBJECT', 'Unknown')
        filename = cube.e3d_data_header.get('FILENAME', 'Unknown')
        flxunits = cube.e3d_data_header.get('FLXUNITS', 'counts')
        
    print "%s: %s" % (obj, filename)

    # Automatic mode: [OII] for B-channel, [NIIHa] for R-channel
    if (opts.line=='auto' and X=='B'):
        opts.line=='OII'
    elif (opts.line=='auto' and X=='R'):
        opts.line=='NIIHa'
    elif (opts.line=='auto' and X=='X'):
        parser.error("Unrecognized input channel")

    # Convert array to pySNIFS.spectrum on a restricted range
    if opts.line=='OII':
        l0 = find_max(lbda, resSpec, (3700,4200)) # OII from 1+z=1 to 1.13
        lmin,lmax = l0-50,l0+50
        print "Fit [OII] doublet around %.0f Å (%.0f Å window)" % (l0,100)
    elif opts.line=='NIIHa':
        l0 = find_max(lbda, resSpec, (6560,7400)) # Ha from 1+z=1 to 1.13
        lmin,lmax = l0-100,l0+100
        print "Fit [NII],Ha complex around %.0f Å (%.0f Å window)" % (l0,200)
    elif opts.line=='OI':
        l0 = find_max(lbda, resSpec, (5527,5627)) # OI at z=0
        lmin,lmax = l0-50,l0+50
        print "Fit [OI] sky line around %.0f Å (%.0f Å window)" % (l0,100)
    else:
        parser.error("Unknown line '%s'" % opts.line)

    g = (lmin<=lbda) & (lbda<=lmax)
    x = lbda[g]
    bkg = N.median(resSpec[g])
    norm = resSpec[g].max() - bkg
    y = (resSpec[g] - bkg) / norm
    v = resVar[g] / norm**2
    sspec = SNIFS_spectrum(data=y, var=v, start=x[0], step=step)

    if opts.line=='OII':                # [OII] doublet + background
        funcs = [ LinesOII.name, 
                  '%s;1,%f,%f' % (LinesBackground.name,lmin,lmax) ]
        # 1+z,sigma,I1,I2 + bkgnd(d=1)
        zp1 = x[resSpec[g].argmax()] / LinesOII.l2
        params = [ [zp1, 3, 0.5, 0.5], [0, 0] ]
        bounds = [ [[1.0,1.13],[2,5],[0,1],[0,1]],
                   [[None,None]]*2] # No constraints on background
        myfunc = {LinesOII.name: LinesOII, 
                  LinesBackground.name: LinesBackground}
    elif opts.line=='NIIHa':       # [NII]+Halpha complex + background
        funcs = [ LinesNIIHa.name, 
                  '%s;1,%f,%f' % (LinesBackground.name,lmin,lmax) ]
        # 1+z,sigma,IHa,I[NII] + bkgnd(d=1)
        zp1 = x[resSpec[g].argmax()] / LinesNIIHa.lHa
        params = [ [zp1, 4, 1, 0.5], [0, 0] ]
        bounds = [ [[1.0,1.13],[2,5],[0.1,2],[0,1]],
                   [[None,None]]*2] # No constraints on background
        myfunc = {LinesNIIHa.name: LinesNIIHa, 
                  LinesBackground.name: LinesBackground}
    elif opts.line=='OI':           # [OI] night sky line + background
        funcs = [ LinesOI.name, 
                  '%s;1,%f,%f' % (LinesBackground.name,lmin,lmax) ]
        # 1+z,sigma,I + bkgnd(d=1)
        zp1 = x[resSpec[g].argmax()] / LinesOI.l0
        params = [ [zp1, 4, 1], [0, 0] ]
        bounds = [ [[0.95,1.05],[2,5],[0.1,2]],
                   [[None,None]]*2] # No constraints on background
        myfunc = {LinesOI.name: LinesOI, 
                  LinesBackground.name: LinesBackground}

    # Actual fit
    model = pySNIFS_fit.model(data=sspec, func=funcs,
                              param=params, bounds=bounds, myfunc=myfunc)

    parnames = [ model.func[i].parnames for i in range(len(funcs)) ]
    flatparnames = reduce(lambda x,y:x+y, parnames)
    print "Adjusted parameters:", parnames
    print "Initial guess:", params

    model.fit(save=True, msge=False)
    model.khi2 *= model.dof             # True Chi2

    print "Status: %d, Chi2/DoF: %.1f/%d" % \
          (model.status, model.khi2, model.dof)
    
    # Quadratic errors, including correlations (tested against Minuit)
    hess = pySNIFS_fit.approx_deriv(model.objgrad, model.fitpar, order=3)
    cov = 2 * N.linalg.inv(hess)        # Covariance matrix (for chi2-fit)
    diag = cov.diagonal()
    if (diag<0).any():          # Error in fit
        model.status = 1
    dfitpar = N.sqrt(diag)
    corr = cov/N.outer(dfitpar,dfitpar) # Correlation matrix

    print "Adjusted parameters (including normalization):"
    for par,val,dval in zip(flatparnames, model.fitpar, dfitpar):
        print "  %s = %f ± %f" % (par,val,dval)

    #print "Correlation matrix:"
    #print N.array2string(corr, 79, 3)

    # Detection level: flux(Ha) in units of sig(flux).
    func = model.func[0]
    if hasattr(func,'flux'):
        f,df = func.flux(model.fitpar[:func.npar_cor],
                         cov=cov[:func.npar_cor,:func.npar_cor])
        nsig = f/df
        print "Detection level: %.1f-sigma (flux: %f ± %f)" % (nsig, f, df)
    else:
        print "WARNING: %s has no flux method, " \
            "cannot compute detection level" % func.name
        nsig = 0

    print "Sigma: %.2f ± %.2f Å" % (model.fitpar[1],dfitpar[1])

    if opts.line=='OI':
        print "Night-sky line [OI]=%.2f Å: " \
              "obs: %.2f ± %.2f Å, offset: %.2f Å" % \
              (LinesOI.l0,
               LinesOI.l0*model.fitpar[0], LinesOI.l0*dfitpar[0],
               LinesOI.l0*(model.fitpar[0]-1))

    # Mean redshift
    zsys0 = model.fitpar[0] - 1
    dzsys = dfitpar[0]
    #print "Estimated redshift: %f ± %f (%.1f ± %.1f km/s)" % \
    #    (zsys0,dzsys,zsys0*CLIGHT,dzsys*CLIGHT)

    if isSpec and opts.line!='OI':
        # Barycentric correction: amount to add to an observed radial
        # velocity to correct it to the solar system barycenter
        v = pspec.get_skycalc('baryvcor')       # Relative velocity [km/s]
        print "Barycentric correction: %f (%.1f ± 0.01 km/s)" % (v/CLIGHT,v)
        zsys = zsys0 + v/CLIGHT
        dzsys = N.hypot(dzsys, 0.01/CLIGHT) # Correction precision: 0.01 km/s
        print "Heliocentric redshift: %f ± %f (%.1f ± %.1f km/s)" % \
            (zsys,dzsys,zsys*CLIGHT,dzsys*CLIGHT)
    else:
        zsys = zsys0
    
    # Store results in input spectra (awkward way, but pySnurp is too dumb...)
    if model.status==0 and isSpec:
            hdu = pyfits.open(specname, mode='update', ignore_missing_end=True)
            hdu[0].header.update('CVSEXTZ',__version__)
            hdu[0].header.update('EXTZ_Z',zsys,
                                 "extract_z heliocentric redshift")
            hdu[0].header.update('EXTZ_DZ',dzsys,
                                 "extract_z error on redshift")
            hdu[0].header.update('EXTZ_K2',model.khi2,
                                 "extract_z chi2")
            hdu[0].header.update('EXTZ_NS',nsig,
                                 "extract_z detection level")
            hdu[0].header.update('EXTZ_L',funcs[0],
                                 "extract_z lines")
            hdu.close()

    if not isSpec:
        ima = cube.slice2d([0,cube.nslice],'p')
        zmap = ima * N.nan                        # Redshift map
        params = model.unflat_param(model.fitpar) # Use global fit result as initial guess
        for ino,ii,ij in zip(cube.no,cube.i,cube.j):
            print "Spx #%03d at %+2dx%+2d:" % (ino,ii-7,ij-7),
            y = cube.spec(no=ino)[g]
            ibkg = N.median(y)
            inorm = y.max() - ibkg
            y = ( y - ibkg ) / inorm
            v = cube.spec(no=ino, var=True)[g] / inorm**2
            ispec = SNIFS_spectrum(data=y, var=v, start=x[0], step=step)
            imodel = pySNIFS_fit.model(data=ispec, func=funcs,
                                       param=params, bounds=bounds, myfunc=myfunc)
            imodel.fit(msge=False, deriv=False)
            imodel.khi2 *= imodel.dof   # True Chi2
            #print "   Fitpar:", imodel.fitpar
            z = imodel.fitpar[0] - 1
            print "Chi2/DoF=%.1f/%d, v=%+.2f km/s" % \
                (imodel.khi2, imodel.dof, (z-zsys0)*CLIGHT)
            zmap[ij,ii] = z

    if opts.plot or os.environ.has_key('PYSHOW'):
        import matplotlib.pyplot as P

        fig = P.figure(figsize=(12,5))
        fig.subplots_adjust(left=0.075, right=0.95)
        title = "%s [%s]" % (obj, filename)
        fig.text(0.5,0.94, title,
                 fontsize='large', horizontalalignment='center')

        ax1 = fig.add_subplot(1,2,1, 
                              xlabel=u'Wavelength [Å]',
                              ylabel='Flux [%s]' % flxunits)
        ax2 = fig.add_subplot(1,4,3, 
                              xlabel=u'Wavelength [Å]')

        # Galaxy spectrum
        lgal, = ax1.plot(lbda, resSpec, 'g-', 
                         label=u"zHelio = %.5f ± %.1g" % (zsys,dzsys))
        if model.status==0:
            #ax1.plot(x, model.evalfit()*norm + bkg, 'r-')
            addRedshiftedLines(ax1, zsys)
        ax1.legend(prop=dict(size='x-small'))
        ax1.set_xlim(lbda[0],lbda[-1])
        ymin,ymax = ax1.get_ylim()
        ax1.set_ylim(min(0,ymin/10),ymax)

        # Zoom on adjusted line
        ax2.plot(x, resSpec[g], 'g-')
        errorband(ax2, x, resSpec[g], N.sqrt(resVar[g]), color='g', alpha=0.3)
        if model.status==0:
            ax2.plot(x, model.evalfit()*norm + bkg, 'r-')
            addRedshiftedLines(ax2, zsys)
            ax2.text(0.1,0.9,
                     "Chi2/DoF=%.1f/%d\nDetection: %.1f sigma" % \
                     (model.khi2, model.dof, nsig),
                     transform=ax2.transAxes, fontsize='small')
        else:
            ax2.plot(x, model.eval(model.flatparam)*norm + bkg, 'm-')
        ax2.set_xlim(x[0],x[-1])

        # Correlation matrix
        if model.status==0:
            ax3 = fig.add_subplot(1,4,4)
            plot_correlation_matrix(ax3, corr, parnames=flatparnames)

        if not isSpec:
            fig2 = P.figure(figsize=(6,6))
            axv = fig2.add_subplot(1,1,1, title=title,
                                   xlabel='I [spx]', ylabel='J [spx]',
                                   aspect='equal')
            # Velocity map
            vmap = (zmap - zsys0)*CLIGHT # Convert redshift to velocity
            vmin,vmax = prctile(vmap[N.isfinite(zmap)], p=(3,97))
            imv = axv.imshow(vmap, vmin=vmin, vmax=vmax,
                             extent=(-7.5,7.5,-7.5,7.5))
            cbv = fig2.colorbar(imv, ax=axv, shrink=0.9)
            cbv.set_label('Velocity [km/s]')
            # Intensity contours
            #axv.contour(ima, vmin=fsmin, vmax=fgmax, colors='k',
            #            extent=(-7.5,7.5,-7.5,7.5))
            for ino,ii,ij in zip(cube.no,cube.i,cube.j):
                axv.text(ii-7,ij-7,str(ino),
                         size='x-small', ha='center', va='center')

        if opts.plot:
            path,name = os.path.split(specname)
            figname = 'z_'+os.path.splitext(name)[0]+'.png'
            print "Saving emission-line figure in", figname
            fig.savefig(figname)
            if not isSpec:
                figname = 'v_'+os.path.splitext(name)[0]+'.png'
                print "Saving velocity-map in", figname
                fig2.savefig(figname)
        if os.environ.has_key('PYSHOW'):
            P.show()
