#!/usr/bin/env python
##############################################################################
## Filename:      extract_star.py
## Version:       $Revision$
## Description:   Standard star spectrum extraction
## Author:        $Author$
## $Id$
##############################################################################

"""
Primarily based on the point source extractor of Emmanuel Pecontal
(extract_star.py).  This version of extract_star replaces double
gaussian PSF profile by an empirical PSF profile (Gaussian + Moffat).
"""

__author__ = "C. Buton, Y. Copin, E. Pecontal"
__version__ = '$Id$'

import os
import sys
import optparse

import pyfits
import pySNIFS
import pySNIFS_fit

import scipy as S
from scipy import linalg as L
from scipy import interpolate as I 
from scipy.ndimage import filters as F

SpaxelSize = 0.43                       # Spaxel size in arcsec

# Definitions ================================================================

def print_msg(str, verbosity, limit=0):
    """Print message 'str' if verbosity level >= verbosity limit."""

    if verbosity >= limit:
        print str

def atmosphericIndex(lbda, P=616, T=2):
    """Compute atmospheric refractive index: lbda in angstrom, P
    in mbar, T in C, RH in %.

    Cohen & Cromer 1988 (PASP, 100, 1582) give P = 456 mmHg = 608 mbar and T =
    2C for Mauna Kea. However, SNIFS observations show an average recorded
    pression of 616 mbar.

    Further note that typical water abundances on Mauna Kea are close enough
    to zero not to significantly impact these calculations.
    """

    # Sea-level (P=760 mmHg, T=15C)
    iml2 = 1/(lbda*1e-4)**2             # lambda in microns
    n = 1 + 1e-6*(64.328 + 29498.1/(146-iml2) + 255.4/(41-iml2))

    # (P,T) correction
    P *= 0.75006168               # Convert P to mmHg: *= 760./1013.25
    n = 1 + (n-1) * P * \
        ( 1 + (1.049 - 0.0157*T)*1e-6*P ) / \
        ( 720.883*(1 + 0.003661*T) )
    
    return n

def fit_param_hdr(hdr, param, lbda_ref, cubename, skyDeg, khi2, alphaDeg,
                  method, radius, seeing):
    
    hdr.update('ES_VERS' ,__version__)
    hdr.update('ES_CUBE' ,cubename,'Input cube')
    hdr.update('ES_LREF' ,lbda_ref,'Lambda ref. [A]')
    hdr.update('ES_SDEG' ,skyDeg,  'Polynomial bkgnd degree')
    hdr.update('ES_KHI2' ,khi2,    'khi2')    
    hdr.update('ES_DELTA',param[0],'ADR power')
    hdr.update('ES_THETA',param[1],'ADR angle')
    hdr.update('ES_XC'   ,param[2],'xc @lbdaRef [spx]')
    hdr.update('ES_YC'   ,param[3],'yc @lbdaRef [spx]')
    hdr.update('ES_ELL'  ,param[4],'Ellipticity')
    hdr.update('ES_PA'   ,param[5],'Position angle')
    for i in xrange(alphaDeg + 1):
        hdr.update('ES_A%i' % i, param[6+i], 'Alpha coeff. a%i' % i)
    hdr.update('ES_METH', method, 'Extraction method')
    if method != 'PSF':
        hdr.update('ES_APRAD', radius, 'Aperture radius [sigma]')
    hdr.update('SEEING', seeing, 'Seeing [arcsec]')

def laplace_filtering(cube, cut=0.9999):

    lapl = F.laplace(cube.data/cube.data.mean())
    fdata = F.median_filter(cube.data, size=[1, 3])
    hist = pySNIFS.histogram(S.ravel(S.absolute(lapl)), nbin=100,
                             Max=100, cumul=True)
    threshold = hist.x[S.argmax(S.where(hist.data<cut, 0, 1))]
    print_msg("Laplace filter threshold: %f" % threshold, opts.verbosity, 2)

    return (S.absolute(lapl) <= threshold)

def comp_spec(cube, psf_fn, psf_ctes, psf_param, skyDeg=0,
              method='PSF', radius=5.):
    """Extract object and sky spectra from cube according to PSF."""

    if method not in ('PSF','aperture','optimal'):
        raise ValueError("Extraction method '%s' unrecognized" % method)

    if (cube.var>1e20).any(): 
        print "WARNING: discarding infinite variances in comp_spec"
        cube.var[cube.var>1e20] = 0
    if (cube.var<0).any():              # There should be none anymore
        print "WARNING: discarding negative variances in comp_spec"
        cube.var[cube.var<0] = 0

    # The PSF parameters are only the shape parameters. We set the intensity
    # of each slice to 1.
    param = S.concatenate((psf_param,[1.]*cube.nslice))

    # Rejection of bad points (YC: need some clarifications...)
    filt = laplace_filtering(cube)
    if (~filt).any():
        print "WARNING: discarding %d bad px in comp_spec" % \
              len((~filt).nonzero()[0])
    cube.var *= filt                    # Discard non-selected px

    # Linear fit: I*PSF + sky [ + a*x + b*y + ...]
    cube.x /= psf_ctes[0]               # x in spaxel
    cube.y /= psf_ctes[0]               # y in spaxel
    model = psf_fn(psf_ctes, cube)
    psf = model.comp(param, normed=True) # nslice x nlens

    npar_poly = int((skyDeg+1)*(skyDeg+2)/2) # Nb param. in polynomial bkgnd
    Z = S.zeros((cube.nslice,cube.nlens,npar_poly+1),'d')
    Z[:,:,0] = psf                      # Intensity
    Z[:,:,1] = 1                        # Constant background
    n = 2
    for d in xrange(1,skyDeg+1):
        for j in xrange(d+1):
            Z[:,:,n] = cube.x**(d-j) * cube.y**j # Structured background
            n=n+1

    # Weighting
    weight = S.sqrt(S.where(cube.var!=0, 1/cube.var, 0))
    X = (Z.T * weight.T).T
    b = weight*cube.data

    A = S.array([S.dot(x.T, x) for x in X])
    B = S.array([S.dot(x.T,bb) for x,bb in zip(X,b)])
    # Spec = nslice x Star,Sky,[slope_x...]
    Spec = S.array([L.solve(a,b) for a,b in zip(A,B)])

    C = S.array([L.inv(a) for a in A])
    # Var = nslice x Star,Sky,[slope_x...]
    Var = S.array([S.diag(c) for c in C])

    if method=='PSF':
        return cube.lbda,Spec,Var       # Nothing else to be done

    # Reconstruct background and subtract it from cube
    bkgnd = 0
    var_bkgnd = 0
    for d in xrange(1,skyDeg+2):
        bkgnd += (Z[:,:,d].T*Spec[:,d]).T
        var_bkgnd += (Z[:,:,d].T*Var[:,d]).T
    subData = cube.data - bkgnd         # Bkgnd subtraction (nslice x nlens)
    subVar = cube.var.copy()
    good = cube.var>0
    subVar[good] += var_bkgnd[good]     # Variance of bkgnd-sub. signal

    radius = radius / SpaxelSize        # Aperture radius in spaxels
    # Centroids [spx] (nslice)
    xc = psf_param[2] + \
         psf_param[0] * model.ADR_coef[:,0] * S.cos(psf_param[1])
    yc = psf_param[3] + \
         psf_param[0] * model.ADR_coef[:,0] * S.sin(psf_param[1])
    # Circular aperture (nslice x nlens)
    r = S.hypot((model.x.T - xc).T,(model.y.T - yc).T)
    inside = r < radius  # r<radius[:,S.newaxis] if radius is a (nslice,) vec.
    
    # Plain summation over aperture
    
    # For the moment, a spaxel is either 100% or 0% within the aperture (same
    # limitation as quick_extract). Potential development:
    # 1. compute spaxel fraction inside the aperture
    # 2. extrapolate missing flux if aperture is partially outside FoV
    #    from PSF fit

    if method == 'aperture':
        # Replace signal and variance estimates from plain sum
        Spec[:,0] = (subData*inside).sum(axis=1) 
        Var[:,0] = (subVar*inside).sum(axis=1)
        return cube.lbda,Spec,Var

    if method=='optimal':
        # One has to have a model of the variance. This can be estimated from
        # a simple 'photon noise + RoN' model on each slice: signal ~ alpha*N
        # (alpha = 1/flat-field coeff and N = photon counts) and variance ~ (N
        # + RoN**2) * alpha**2 = (signal/alpha + RoN**2) * alpha**2 =
        # alpha*signal + beta. This model disregards spatial component of
        # flat-field, which is supposed to be constant of FoV.
        coeffs = S.array([ S.polyfit(cube.data[s], cube.var[s], 1)
                           for s in xrange(cube.nslice) ])
        alpha = F.median_filter(coeffs[:,0], 10) # A bit of smoothing...
        beta = coeffs[:,1]
        # Model signal = Intensity*PSF + bkgnd
        # Model variance = alpha*signal + beta
        modsig = (Spec[:,0]*psf.T).T + bkgnd # nslice x nlens
        modvar = (alpha*modsig.T + beta).T

        s = 300                         # Slice under investigation
        import pylab
        pylab.plot(cube.lbda, 1/coeffs[:,0], 'b-')
        pylab.plot(cube.lbda, 1/alpha, 'g-')

        dx = (model.x.T - xc).T         # nslice x nlens
        dy = (model.y.T - yc).T
        rell = S.sqrt(dx**2 + psf_param[4]*dy**2 + 2*psf_param[5]*dx*dy)
        pylab.figure()
        pylab.plot(rell[s], cube.data[s], 'b+') # Data
        pylab.plot(rell[s], modsig[s], 'r+')  # Model
        pylab.plot(rell[s], bkgnd[s], 'c+')
        pylab.plot(rell[s], cube.var[s], 'g+')
        pylab.plot(rell[s], modvar[s], 'y+')
        pylab.semilogy()
        pylab.title("Slice %d" % s)
        pylab.show()
        raise NotImplementedError

    return cube.lbda,Spec,Var       # Nothing else to be done

def get_start(cube, psf_fn, skyDeg=0):
    
    npar_poly = int((skyDeg+1)*(skyDeg+2)/2) # Nb. param. in polynomial bkgnd
    npar_psf  = 7                       # Number of parameters of the psf

    cube_sky = pySNIFS.SNIFS_cube()
    cube_sky.x = cube.x 
    cube_sky.y = cube.y 
    cube_sky.i = cube.i 
    cube_sky.j = cube.j 
    cube_sky.nslice = 1 
    cube_sky.nlens = cube.nlens 

    cube_star = pySNIFS.SNIFS_cube()
    cube_star.x = cube.x
    cube_star.y = cube.y
    cube_star.i = cube.i
    cube_star.j = cube.j
    cube_star.nslice = 1
    cube_star.nlens = cube.nlens

    nslice = cube.nslice

    delta_vec = S.zeros(nslice, dtype='d')
    theta_vec = S.zeros(nslice, dtype='d')
    xc_vec    = S.zeros(nslice, dtype='d')
    yc_vec    = S.zeros(nslice, dtype='d')
    ell_vec   = S.zeros(nslice, dtype='d')
    PA_vec    = S.zeros(nslice, dtype='d')
    alpha_vec = S.zeros(nslice, dtype='d')
    int_vec   = S.zeros(nslice, dtype='d')
    khi2_vec  = S.zeros(nslice, dtype='d')
    sky_vec   = S.zeros((nslice,npar_poly), dtype='d')
    error_mat = S.zeros((nslice,npar_psf+npar_poly+1), dtype='d') # PSF+Intens.+Bkgnd
    
    n = 2                               # Nb of edge spx used for sky estimate
    if n>7:
        raise ValueError('The number of edge pixels should be less than 7')

    for i in xrange(nslice):
        cube_star.lbda = S.array([cube.lbda[i]])
        cube_star.data = cube.data[i, S.newaxis]
        cube_star.var  = cube.var[i,  S.newaxis]
        cube_sky.data  = cube.data[i, S.newaxis]
        cube_sky.var   = cube.var[i,  S.newaxis]
        if opts.verbosity >= 1:
            sys.stdout.write('\rSlice %2d/%d' % (i+1, nslice))
            sys.stdout.flush()
        print_msg("", opts.verbosity, 2)

        # Sky estimate (from FoV edge spx)
        ind = S.where((cube_sky.i<n) | (cube_sky.i>=15-n) | \
                      (cube_sky.j<n) | (cube_sky.j>=15-n))
        bkgnd = S.median(cube_sky.data.T[ind].squeeze())

        # Guess parameters for the current slice
        star_int = F.median_filter(cube_star.data[0] - bkgnd, 3)
        imax = star_int.max()           # Intensity
        xc = S.average(cube_star.x, weights=star_int) # Centroid
        yc = S.average(cube_star.y, weights=star_int)
        xc = S.clip(xc, -3.5,3.5)       # Put initial guess ~ in FoV
        yc = S.clip(yc, -3.5,3.5)

        # Filling in the guess parameter arrays (px) and bounds arrays (bx)
        p1 = [0., 0., xc, yc, 1., 0., 2.4, imax] # psf parameters
        b1 = [[None, None],             # delta
              [None, None],             # theta
              [None, None],             # xc
              [None, None],             # yc
              [0., None],               # ellipticity 
              [None, None],             # PA
              [0., None],               # alpha > 0
              [0., None]]               # Intensity > 0
        p2 = [bkgnd] + [0.]*(npar_poly-1) # Guess: Background=constant (>0)
        b2 = [[0,None]] + [[None,None]]*(npar_poly-1)
        print_msg("    Initial guess [PSF+bkgnd]: %s" % (p1+[p2[0]]),
                  opts.verbosity, 2)

        # Instanciating of a model class
        lbda_ref = cube_star.lbda[0]
        model_star = pySNIFS_fit.model(data=cube_star,
                                       func=['%s;%f,%f,%f' % \
                                             (psf_fn.__name__,
                                              SpaxelSize,lbda_ref,0), # a0=cte
                                             'poly2D;%d' % skyDeg],
                                       param=[p1,p2],
                                       bounds=[b1,b2],
                                       myfunc={psf_fn.__name__:psf_fn})

        # Fit of the current slice
        model_star.fit(maxfun=400, msge=int(opts.verbosity >= 3))

        # Error computation
        hess = pySNIFS_fit.approx_deriv(model_star.objgrad,
                                        model_star.fitpar,order=2)

        if model_star.fitpar[4]>0 and \
               model_star.fitpar[6]>0 and model_star.fitpar[7]>0: 
            cov = S.linalg.inv(hess[2:,2:]) # Discard 1st 2 lines (unfitted)
            errorpar = S.concatenate(([0.,0.], S.sqrt(cov.diagonal())))
        else:
            # Set error to 0 if alpha, intens. or ellipticity is 0. 
            errorpar = S.zeros(9)

        # Storing the result of the current slice parameters
        delta_vec[i] = model_star.fitpar[0]
        theta_vec[i] = model_star.fitpar[1]
        xc_vec[i]    = model_star.fitpar[2]
        yc_vec[i]    = model_star.fitpar[3]
        ell_vec[i]   = model_star.fitpar[4]
        PA_vec[i]    = model_star.fitpar[5]
        alpha_vec[i] = model_star.fitpar[6]
        int_vec[i]   = model_star.fitpar[7]
        sky_vec[i]   = model_star.fitpar[8]
        khi2_vec[i]  = model_star.khi2
        error_mat[i] = errorpar
        print_msg("    Fit result [PSF+bkgnd]: %s" % \
                  model_star.fitpar, opts.verbosity, 2)

    return (delta_vec,theta_vec,xc_vec,yc_vec,ell_vec,PA_vec,
            alpha_vec,int_vec,sky_vec,khi2_vec,error_mat)

def create_log_file(filename,delta,theta,xc,yc,ell,PA,alpha,
                    khi2,khi3D,model,nslice):

    def strParam(i,param,j=5):
        if len(S.shape(param))==1:
            if param[i]>0:
                return '  '+str(param[i])[:j]+' '
            elif param[i]==0:
                return '  '+'0.000'+' '
            else:
                return '  '+str(param[i])[:j]+' '

    logfile = open(filename,'w')
    logfile.write('extract_star.py: result file for %s\n\n' % \
                  os.path.basename(opts.input))
    logfile.write('slice    delta   theta    xc       yc     ell     PA    %s    khi2\n' % \
                  ''.join(['  a%d    ' % i for i in S.arange(alphaDeg+1)]))

    for n,khi in enumerate(khi2):
        logfile.write('%2s/%2s'%(n+1,nslice)+' :'\
                      +strParam(n,delta)\
                      +strParam(n,theta)\
                      +strParam(n,xc)\
                      +strParam(n,yc)\
                      +strParam(n,ell)\
                      +strParam(n,PA,)\
                      +strParam(n,alpha)\
                      +('  '+'0.000'+' ')*alphaDeg
                      +'  '+str(khi)[:9]\
                      +'\n')
    logfile.write('\n\n'+'3Dfit :'\
                  +strParam(0,model.fitpar)\
                  +strParam(1,model.fitpar)\
                  +strParam(2,model.fitpar)\
                  +strParam(3,model.fitpar)\
                  +strParam(4,model.fitpar)\
                  +strParam(5,model.fitpar)\
                  +''.join([ strParam(6+m,model.fitpar)
                             for m in xrange(alphaDeg+1) ])\
                  +'  '+str(khi3D)[:9]\
                  +'\n')
    logfile.close()

def build_sky_cube(cube, sky, sky_var, skyDeg):

    nslices   = len(sky)
    npar_poly = int((skyDeg+1)*(skyDeg+2)/2)
    poly      = pySNIFS_fit.poly2D(skyDeg,cube)
    cube2     = pySNIFS.zerolike(cube)
    cube2.x   = (cube2.x)**2
    cube2.y   = (cube2.y)**2
    poly2     = pySNIFS_fit.poly2D(skyDeg,cube2)
    param     = S.zeros((nslices,npar_poly),'d')
    vparam    = S.zeros((nslices,npar_poly),'d')
    for i in xrange(nslices):
        param[i,:] = sky[i].data
        vparam[i,:] = sky_var[i].data
    data = poly.comp(param)
    var = poly2.comp(vparam)
    bkg_cube = pySNIFS.zerolike(cube)
    bkg_cube.data = data
    bkg_cube.var = var
    bkg_spec = bkg_cube.get_spec(no=bkg_cube.no)

    return bkg_cube,bkg_spec

# PSF classes ================================================================

class ExposurePSF:
    """
    Empirical PSF 3D function used by the L{model} class.
    """
    
    def __init__(self, psf_ctes, cube):
        """
        Initiating the class.
        @param psf_ctes: Internal parameters (pixel size in cube spatial unit,
                       reference wavelength and polynomial degree of alpha). A
                       list of three numbers.
        @param cube: Input cube. This is a L{SNIFS_cube} object.
        """
        self.pix      = psf_ctes[0]
        self.lbda_ref = psf_ctes[1]
        self.alphaDeg = int(psf_ctes[2])
        self.npar_ind = 1               # Intensity
        self.npar_cor = 7 + self.alphaDeg # PSF parameters
        self.npar = self.npar_ind*cube.nslice + self.npar_cor
        self.x = S.resize(cube.x, (cube.nslice,cube.nlens)) # nslice x nlens
        self.y = S.resize(cube.y, (cube.nslice,cube.nlens))
        self.l = S.resize(cube.lbda, (cube.nlens,cube.nslice)).T # nlens x nslice
        self.n_ref = atmosphericIndex(self.lbda_ref)
        # ADR in spaxels
        self.ADR_coef = 206265*(atmosphericIndex(self.l) - self.n_ref) / SpaxelSize

    def comp(self, param, normed=False):
        """
        Compute the function.
        @param param: Input parameters of the polynomial. A list of numbers:
                - C{param[0:6+m]} : The n parameters of the PSF shape
                     - C{param[0]}: Atmospheric dispersion power
                     - C{param[1]}: Atmospheric dispersion position angle
                     - C{param[2]}: X center at the reference wavelength
                     - C{param[3]}: Y center at the reference wavelength
                     - C{param[4]}: Ellipticity
                     - C{param[5]}: Position angle
                     - C{param[m]}: Moffat radius (m coefficients corresponding to the
                                    polynomial degree of alpha + 1)
                - C{param[6+m:]}  : Intensity parameters (one for each slice in the cube).
        """

        self.param = S.asarray(param)

        # ADR params
        delta = self.param[0]
        theta = self.param[1]
        xc    = self.param[2]
        yc    = self.param[3]
        x0 = xc + delta*self.ADR_coef*S.cos(theta)
        y0 = yc + delta*self.ADR_coef*S.sin(theta)

        # Other params
        ell = self.param[4]
        PA  = self.param[5]
        alphaCoeffs = self.param[6:7+self.alphaDeg]

        # aliases + correlations params (fixed)
        lbda_rel = self.l / self.lbda_ref - 1  
        alpha = 0
        for i,a in enumerate(alphaCoeffs):
            alpha += a * lbda_rel**i

        s1,s0,b1,b0,e1,e0 = self.corrCoeffs
        sigma = s0 + s1*alpha
        beta  = b0 + b1*alpha
        eta   = e0 + e1*alpha

        dx  = self.x - x0
        dy  = self.y - y0
        r2 = dx**2 + ell*dy**2 + 2*PA*dx*dy
        gaussian = S.exp(-r2/2/sigma**2)
        moffat = (1 + r2/alpha**2)**(-beta)

        # function
        val = self.param[self.npar_cor:,S.newaxis] * (moffat + eta*gaussian)

        # The 3D psf model is not normalized to 1 in integral. The result must
        # be renormalized by (2*eta*sigma**2 + alpha**2/(beta-1)) *
        # S.pi/sqrt(ell)
        if normed:
            val /= S.pi*( 2*eta*sigma**2 + alpha**2/(beta-1) )/S.sqrt(ell)

        return val

    def deriv(self, param):
        """
        Compute the derivative of the function with respect to its parameters.
        @param param: Input parameters of the polynomial.
                      A list numbers (see L{SNIFS_psf_3D.comp}).
        """

        self.param = S.asarray(param)

        # ADR params
        delta = self.param[0]
        theta = self.param[1]
        xc    = self.param[2]
        yc    = self.param[3]
        costheta = S.cos(theta)
        sintheta = S.sin(theta)
        x0 = xc + delta*self.ADR_coef*costheta
        y0 = yc + delta*self.ADR_coef*sintheta
        
        # Other params
        ell = self.param[4]
        PA  = self.param[5]
        alphaCoeffs = self.param[6:7+self.alphaDeg]

        # aliases + correlations params (fixed)
        lbda_rel = self.l / self.lbda_ref - 1
        alpha = 0
        for i,a in enumerate(alphaCoeffs):
            alpha += a * lbda_rel**i

        s1,s0,b1,b0,e1,e0 = self.corrCoeffs
        sigma = s0 + s1*alpha
        beta  = b0 + b1*alpha
        eta   = e0 + e1*alpha

        dx = self.x - x0
        dy = self.y - y0
        r2 = dx**2 + ell*dy**2 + 2*PA*dx*dy
        gaussian = S.exp(-r2/2/sigma**2)
        ea = 1 + r2/alpha**2
        moffat = ea**(-beta)
        da0 = e1*gaussian + eta*r2*s1*gaussian/sigma**3 + \
              moffat*( -b1*S.log(ea) + 2*beta*r2/(ea*alpha**3) )

        # Derivatives
        grad = S.zeros((self.npar_cor+self.npar_ind,)+self.x.shape,'d')
        tmp = eta*gaussian/sigma**2 + 2*beta*moffat/ea/alpha**2
        grad[2] = tmp*(    dx + PA*dy)
        grad[3] = tmp*(ell*dy + PA*dx)
        grad[0] =       self.ADR_coef*(costheta*grad[2] + sintheta*grad[3])
        grad[1] = delta*self.ADR_coef*(costheta*grad[3] - sintheta*grad[2])
        grad[4] = -tmp/2 * dy**2
        grad[5] = -tmp   * dx*dy
        for i in xrange(self.alphaDeg + 1):
            grad[6+i] = da0 * lbda_rel**i
        grad[self.npar_cor] = moffat + eta*gaussian
        
        grad[:self.npar_cor] *= self.param[S.newaxis,self.npar_cor:,S.newaxis]

        return grad
    
    def _HWHM_fn(self, r, alphaCoeffs, lbda):
            """Half-width at half maximum function (=0 at HWHM)."""
            lbda_rel = lbda/self.lbda_ref - 1
            alpha = 0
            for i,a in enumerate(alphaCoeffs):
                alpha += a * lbda_rel**i
            s1,s0,b1,b0,e1,e0 = self.corrCoeffs
            sigma = s0 + s1*alpha
            beta  = b0 + b1*alpha
            eta   = e0 + e1*alpha
            gaussian = S.exp(-r**2/2/sigma**2)
            moffat = (1 + r**2/alpha**2)**(-beta)
            # PSF=moffat + eta*gaussian, maximum is 1+eta
            return moffat + eta*gaussian - (eta + 1)/2

    def FWHM(self, param, lbda):
        """Estimate FWHM of PSF at wavelength lbda."""

        alphaCoeffs = param[6:7+self.alphaDeg]
        # Compute FWHM from radial profile
        fwhm = 2*S.optimize.fsolve(func=self._HWHM_fn, x0=1.,
                                   args=(alphaCoeffs,lbda))

        return fwhm                     # In spaxels

class long_exposure_psf(ExposurePSF): 

    name = 'long'
    corrCoeffs = [0.215,0.545,0.345,1.685,0.0,1.04] # long exposures

class short_exposure_psf(ExposurePSF):

    name = 'short'
    corrCoeffs = [0.2,0.56,0.415,1.395,0.16,0.6] # short exposures

# ########## MAIN ##############################

if __name__ == "__main__":

    # Options ================================================================

    usage = "usage: [%prog] [options] -i inE3D.fits " \
            "-o outSpec.fits -s outSky.fits -f file.log"

    parser = optparse.OptionParser(usage, version=__version__)
    parser.add_option("-i", "--in", type="string", dest="input",
                      help="Input datacube (euro3d format)")
    parser.add_option("-d", "--skyDeg", type="int", dest="skyDeg",
                      help="Sky polynomial background degree [%default]",
                      default=0 )
    parser.add_option("-a", "--alphaDeg", type="int",
                      help="Alpha polynomial degree [%default]",
                      default=1)
    parser.add_option("-o", "--out", type="string",
                      help="Output star spectrum")
    parser.add_option("-s", "--sky", type="string",
                      help="Output sky spectrum")
    parser.add_option("-p", "--plot", action='store_true',
                      help="Plot flag (syn. '--graph=png')")
    parser.add_option("-g", "--graph", type="string",
                      help="Graphic output format ('eps', 'png' or 'pylab')")
    parser.add_option("-v", "--verbosity", type="int",
                      help="Verbosity level (<0: quiet) [%default]",
                      default=0)
    parser.add_option("-f", "--file", type="string",
                      help="Save file with the different parameters fitted.")
    parser.add_option("-m", "--method", type="string",
                      help="Extraction method ['%default'].",
                      default="PSF")
    parser.add_option("-r", "--radius", type="float",
                      help="Aperture radius for non-PSF extraction " \
                      "[%default sigma]", default=5.)

    opts,pars = parser.parse_args()
    if not opts.input or not opts.out or not opts.sky:
        parser.error("At least one option is missing among " \
                     "'--in', '--out' and '--sky'.")

    if opts.graph:
        opts.plot = True
    elif opts.plot:
        opts.graph = 'png'

    if opts.method not in ('PSF','aperture','optimal'):
        parser.error("Extraction method '%s' unrecognized "
                     "(PSF|aperture|optimal)." % method)

    # Nb of param. in polynomial background
    npar_poly = int((opts.skyDeg+1)*(opts.skyDeg+2)/2) 

    # Input datacube =========================================================

    print_msg("Opening datacube %s" % opts.input, opts.verbosity, 0)
    inhdr = pyfits.getheader(opts.input, 1) # 1st extension
    obj = inhdr.get('OBJECT', 'Unknown')
    efftime = inhdr.get('EFFTIME')
    airmass = inhdr.get('AIRMASS', 0.0)
    channel = inhdr.get('CHANNEL', 'Unknown').upper()
    alphaDeg = opts.alphaDeg
    npar_psf  = 7 + alphaDeg

    if channel.startswith('B'):
        slices=[10, 900, 65]
    elif channel.startswith('R'):
        slices=[10, 1500, 130]
    else:
        parser.error("Input datacube %s has no valid CHANNEL keyword (%s)" % \
                     (opts.input, channel))

    # Select the PSF (short or long)
    if efftime > 5.:                    # Long exposure
        psfFn = long_exposure_psf
    else:                               # Short exposure
        psfFn = short_exposure_psf

    print_msg("  Object: %s, Airmass: %.2f; Efftime: %.1fs [%s]" % \
              (obj, airmass, efftime, psfFn.name), opts.verbosity, 0)
    print_msg("  Channel: '%s', extracting slices: %s" % (channel,slices),
              opts.verbosity, 0)

    cube    = pySNIFS.SNIFS_cube(opts.input, slices=slices)
    cube.x /= SpaxelSize                # From arcsec to spx
    cube.y /= SpaxelSize

    print_msg("  Meta-slices before selection: %d " \
              "from %.2f to %.2f by %.2f A" % \
              (len(cube.lbda), cube.lbda[0], cube.lbda[-1], cube.lstep),
              opts.verbosity, 1)

    # Normalisation of the signal and variance in order to avoid numerical
    # problems with too small numbers
    norm = cube.data.mean()
    cube.data /= norm
    cube.var /= norm**2

    # Rejection of bad points ================================================

    # YC: not clear this selection is needed anymore...
    print_msg("Rejection of slices with bad values...", opts.verbosity, 0)
    max_spec = cube.data.max(axis=1)    # Max per slice
    med = S.median(F.median_filter(max_spec, 5) - max_spec)
    tmp2 = (max_spec - med)**2
    indice = tmp2 < 25*S.median(tmp2)
    if (-indice).any():                 # Some discarded slices
        print_msg("   %d slices discarded: %s" % \
                  (len(cube.lbda[-indice]), cube.lbda[-indice]),
                  opts.verbosity, 0)
        cube.data = cube.data[indice]
        cube.lbda = cube.lbda[indice]
        cube.var = cube.var[indice]
        cube.nslice = len(cube.lbda)

    # Computing guess parameters from slice by slice 2D fit ==================

    print_msg("Slice-by-slice 2D-fitting...", opts.verbosity, 0)

    tmp = get_start(cube, psfFn, skyDeg=opts.skyDeg)
    (delta_vec,theta_vec,xc_vec,yc_vec,ell_vec,PA_vec, \
     alpha_vec,int_vec,sky_vec,khi2_vec,error_mat) = tmp
    print_msg("", opts.verbosity, 1)

    # 3D model fitting =======================================================
    
    print_msg("Datacube 3D-fitting...", opts.verbosity, 0)

    # Computing the initial guess for the 3D fitting from the results of the
    # slice by slice 2D fit
    lbda_ref = cube.lbda.mean()
    nslice = cube.nslice
    lbda_rel = cube.lbda / lbda_ref - 1
    
    # the xc, yc vectors obtained from 2D fit are smoothed, then the
    # position corresponding to the reference wavelength is read in the
    # filtered vectors. Finally, the parameters theta and delta are
    # determined from the xc, yc vectors.
    ind = ( S.absolute(xc_vec)<7 ) & ( S.absolute(yc_vec)<7 )
    if not ind.all():                   # Some centroids outside FoV
        print "%d/%d centroid positions discarded from ADR initial guess" % \
              (len(xc_vec[-ind]),nslice)
        if (len(xc_vec[ind])<=1):
            raise ValueError('Not enough points to determine ADR initial guess')

    xc_vec2 = xc_vec[ind]
    yc_vec2 = yc_vec[ind]
    ADR_coef = 206265*(atmosphericIndex(cube.lbda) -
                       atmosphericIndex(lbda_ref)) / SpaxelSize # In spaxels

    polADR = pySNIFS.fit_poly(yc_vec2, 3, 1, xc_vec2)
    xc = xc_vec2[S.argmin(S.absolute(cube.lbda[ind] / lbda_ref - 1))]
    yc = polADR(xc)
    
    delta = S.tan(S.arccos(1./airmass)) # ADR power
    theta = S.arctan(polADR(1))         # ADR angle
    if (xc_vec2[-1]-xc_vec2[0]) > 0:
        theta += S.pi

    # 2) Other parameters:
    polAlpha = pySNIFS.fit_poly(alpha_vec,3,alphaDeg,lbda_rel)
    alpha = polAlpha.coeffs 
    ell = S.median(ell_vec)
    PA = S.median(PA_vec)

    # Filling in the guess parameter arrays (px) and bounds arrays (bx)
    p1 = [None]*(npar_psf+nslice)
    p1[:6] = [delta, theta, xc, yc, ell, PA]
    p1[6:npar_psf] = alpha[::-1]
    p1[npar_psf:npar_psf+nslice] = int_vec.tolist()

    b1 = [[None, None],                 # delta 
          [None, None],                 # theta 
          [None, None],                 # x0 
          [None, None],                 # y0 
          [0., None],                   # ellipticity 
          [None, None]]                 # PA
    b1 += [[0,None]] + [[None, None]]*alphaDeg # a0 > 0
    b1 += [[0, None]]*nslice            # Intensities

    p2 = S.ravel(sky_vec.T)
    b2 = ([[0.,None]]+[[None,None]]*(npar_poly-1))*nslice 

    print_msg("  Initial guess: %s" % p1[:12], opts.verbosity, 2)

    # Instanciating the model class
    data_model = pySNIFS_fit.model(data=cube,
                                   func=['%s;%f,%f,%f' % \
                                         (psfFn.__name__,SpaxelSize,lbda_ref,alphaDeg),
                                         'poly2D;%d' % opts.skyDeg],
                                   param=[p1,p2],
                                   bounds=[b1,b2],
                                   myfunc={psfFn.__name__:psfFn})

    guesspar = data_model.flatparam

    if opts.verbosity >= 3:
        data_model.fit(maxfun=2000, save=True, msge=1)
    else:
        data_model.fit(maxfun=2000, save=True)

    # Storing result and guess parameters
    fitpar   = data_model.fitpar
    khi2     = data_model.khi2
    cov      = data_model.param_error(fitpar)
    cov      = S.where(cov<0,0.,cov)    # YC: ???
    errorpar = S.sqrt(cov.diagonal())

    print_msg("  Fit result: %s" % fitpar[:npar_psf], opts.verbosity, 2)

    # Compute seeing (FWHM in arcsec)
    seeing = data_model.func[0].FWHM(fitpar[:npar_psf], lbda_ref) * SpaxelSize
    print_msg("  Seeing estimate: %.2f'' FWHM" % seeing, opts.verbosity, 0)

    # Computing final spectra for object and background ======================

    # Compute aperture radius
    if opts.method == 'PSF':
        radius = None
        method = 'PSF'
    else:
        radius = opts.radius * seeing/2.355 # Aperture radius [arcsec]
        method = "%s r = %.1f sigma = %.2f''" % \
                 (opts.method, opts.radius, radius)
    print_msg("Extracting the spectrum [method=%s]..." % method,
              opts.verbosity, 0)

    full_cube = pySNIFS.SNIFS_cube(opts.input)
    print_msg("Cube %s: %d slices, %d spaxels" % \
              (os.path.split(opts.input)[1], full_cube.nslice, full_cube.nlens),
              opts.verbosity, 0)

    lbda,spec,var = comp_spec(full_cube, psfFn, [SpaxelSize,lbda_ref,alphaDeg],
                              fitpar[:npar_psf], skyDeg=opts.skyDeg,
                              method=opts.method, radius=radius)

    # Save star spectrum, update headers ==============================

    fit_param_hdr(inhdr,fitpar[:npar_psf],lbda_ref,opts.input,opts.skyDeg,khi2,
                  alphaDeg, opts.method, opts.radius, seeing)
    step = inhdr.get('CDELTS')
    star_spec = pySNIFS.spectrum(data=spec[:,0],start=lbda[0],step=step)
    star_spec.WR_fits_file(opts.out,header_list=inhdr.items())
    star_var = pySNIFS.spectrum(data=var[:,0],start=lbda[0],step=step)
    star_var.WR_fits_file('var_'+opts.out,header_list=inhdr.items())

    # Save sky spectrum/spectra ==============================================

    spec[:,1:] /= SpaxelSize**2         # Per arcsec^2
    var[:,1:]  /= SpaxelSize**4

    sky_spec_list = pySNIFS.spec_list([ pySNIFS.spectrum(data=s,start=lbda[0],
                                                         step=step)
                                        for s in spec[:,1:] ])
    sky_var_list = pySNIFS.spec_list([ pySNIFS.spectrum(data=v,start=lbda[0],
                                                        step=step)
                                       for v in var[:,1:] ])
    bkg_cube,bkg_spec = build_sky_cube(full_cube,
                                       sky_spec_list.list,sky_var_list.list,
                                       opts.skyDeg)
    bkg_spec.data /= cube.nlens
    bkg_spec.var  /= cube.nlens
    
    sky_spec = pySNIFS.spectrum(data=bkg_spec.data,start=lbda[0],step=step)
    sky_spec.WR_fits_file(opts.sky,header_list=inhdr.items())
    sky_var = pySNIFS.spectrum(data=bkg_spec.var,start=lbda[0],step=step)
    sky_var.WR_fits_file('var_'+opts.sky,header_list=inhdr.items())

    # Save adjusted parameter file ===========================================
    
    if opts.file:
        create_log_file(opts.file, delta_vec,theta_vec,xc_vec,yc_vec,\
                        ell_vec,PA_vec,alpha_vec,khi2_vec,khi2,\
                        data_model, nslice)

    # Create output graphics =================================================

    if opts.plot:
        print_msg("Producing output figures [%s]..." % \
                  opts.graph, opts.verbosity, 0)

        import matplotlib
        if opts.graph=='png':
            matplotlib.use('Agg')
        elif opts.graph=='eps':
            matplotlib.use('PS')
        else:
            opts.graph = 'pylab'
        import pylab

        if opts.graph=='pylab':
            plot1 = plot2 = plot3 = plot4 = plot6 = plot7 = plot8 = plot5 = ''
        else:
            basename = os.path.splitext(opts.out)[0]
            plot1 = os.path.extsep.join((basename+"_plt" , opts.graph))
            plot2 = os.path.extsep.join((basename+"_fit1", opts.graph))
            plot3 = os.path.extsep.join((basename+"_fit2", opts.graph))
            plot4 = os.path.extsep.join((basename+"_fit3", opts.graph))
            plot6 = os.path.extsep.join((basename+"_fit4", opts.graph))
            plot7 = os.path.extsep.join((basename+"_fit5", opts.graph))
            plot8 = os.path.extsep.join((basename+"_fit6", opts.graph))
            plot5 = os.path.extsep.join((basename+"_fit7", opts.graph))

        # Plot of the star and sky spectra -----------------------------------

        print_msg("Producing spectra plot %s..." % plot1, opts.verbosity, 1)

        fig1 = pylab.figure()

        axS = fig1.add_subplot(3, 1, 1)
        axB = fig1.add_subplot(3, 1, 2)
        axN = fig1.add_subplot(3, 1, 3)

        axS.plot(star_spec.x, star_spec.data, 'b')
        axS.set_title("Star spectrum [%s, %s]" % (obj,method))
        axS.set_xlim(star_spec.x[0],star_spec.x[-1])
        axS.set_xticklabels([])

        axB.plot(bkg_spec.x, bkg_spec.data, 'g')
        axB.set_xlim(bkg_spec.x[0],bkg_spec.x[-1])
        axB.set_title("Background spectrum (per sq. arcsec)")
        axB.set_xticklabels([])

        axN.plot(star_spec.x, S.sqrt(star_var.data), 'b')
        axN.plot(bkg_spec.x, S.sqrt(bkg_spec.var), 'g')
        axN.set_title("Error spectra")
        axN.semilogy()
        axN.set_xlim(star_spec.x[0],star_spec.x[-1])
        axN.set_xlabel("Wavelength [A]")

        # Plot of the fit on each slice --------------------------------------

        print_msg("Producing slice fit plot %s..." % plot2, opts.verbosity, 1)

        ncol = S.floor(S.sqrt(cube.nslice))
        nrow = S.ceil(cube.nslice/float(ncol))

        fig2 = pylab.figure()

        fig2.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97, )
        for i in xrange(cube.nslice):   # Loop over meta-slices
            ax = fig2.add_subplot(nrow, ncol, i+1)
            data = cube.data[i,:]
            fit = data_model.evalfit()[i,:]
            fmin = min(data.min(), fit.min()) - 2e-2
            ax.plot(data-fmin)          # Signal
            ax.plot(fit-fmin)           # Fit
            ax.semilogy()
            ax.set_xlim(0,len(data))
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.1,0.8, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='left', transform=ax.transAxes)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Spaxel ID", fontsize=8)
                ax.set_ylabel("Flux + cte", fontsize=8)

        # Plot of the fit on rows and columns sum ----------------------------

        print_msg("Producing profile plot %s..." % plot3, opts.verbosity, 1)

        # Creating a standard SNIFS cube with the adjusted data
        cube_fit = pySNIFS.SNIFS_cube(lbda=cube.lbda)
        cube_fit.x /= SpaxelSize        # x in spaxel 
        cube_fit.y /= SpaxelSize        # y in spaxel

        psf_model = psfFn([SpaxelSize, lbda_ref, alphaDeg], cube=cube_fit)
        bkg_model = pySNIFS_fit.poly2D(0, cube_fit)

        psf = psf_model.comp(fitpar[:psf_model.npar])
        bkg = bkg_model.comp(fitpar[psf_model.npar: \
                                    psf_model.npar+bkg_model.npar])
        cube_fit.data =  psf + bkg

        fig3 = pylab.figure()

        fig3.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97, )
        for i in xrange(cube.nslice):   # Loop over slices
            ax = fig3.add_subplot(nrow, ncol, i+1)
            sigSlice = S.nan_to_num(cube.slice2d(i, coord='p'))
            varSlice = S.nan_to_num(cube.slice2d(i, coord='p', var=True))
            modSlice = cube_fit.slice2d(i, coord='p')
            prof_I = sigSlice.sum(axis=0) # Sum along rows
            prof_J = sigSlice.sum(axis=1) # Sum along columns
            err_I = S.sqrt(varSlice.sum(axis=0))
            err_J = S.sqrt(varSlice.sum(axis=1))
            mod_I = modSlice.sum(axis=0)
            mod_J = modSlice.sum(axis=1)
            ax.errorbar(range(len(prof_I)),prof_I,err_I, fmt='bo', ms=3)
            ax.plot(mod_I, 'b-')
            ax.errorbar(range(len(prof_J)),prof_J,err_J, fmt='r^', ms=3)
            ax.plot(mod_J, 'r-')
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.1,0.8, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='left', transform=ax.transAxes)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I (blue) or J (red)", fontsize=8)
                ax.set_ylabel("Flux", fontsize=8)

        # Plot of the star center of gravity and adjusted center -------------

        print_msg("Producing ADR plot %s..." % plot4, opts.verbosity, 1)

        xfit = fitpar[2] + \
               fitpar[0] * psf_model.ADR_coef[:,0] * S.cos(fitpar[1])
        yfit = fitpar[3] + \
               fitpar[0] * psf_model.ADR_coef[:,0] * S.sin(fitpar[1])
        xguess = guesspar[2] + \
                 guesspar[0]*psf_model.ADR_coef[:,0]*S.cos(guesspar[1])
        yguess = guesspar[3] + \
                 guesspar[0]*psf_model.ADR_coef[:,0]*S.sin(guesspar[1])

        fig4 = pylab.figure()

        ax4a = fig4.add_subplot(2, 2, 1)
        ax4a.errorbar(cube.lbda, xc_vec,yerr=error_mat[:,2],
                      fmt='b.',ecolor='b',label="Fit 2D")
        ax4a.plot(cube.lbda, xguess, 'k--', label="Guess 3D")
        ax4a.plot(cube.lbda, xfit, 'g', label="Fit 3D")
        ax4a.set_xlabel("Wavelength [A]")
        ax4a.set_ylabel("X center [spaxels]")
        pylab.setp(ax4a.get_xticklabels()+ax4a.get_yticklabels(), fontsize=8)
        leg = ax4a.legend(loc='best')
        pylab.setp(leg.get_texts(), fontsize='smaller')

        ax4b = fig4.add_subplot(2, 2, 2)
        ax4b.errorbar(cube.lbda, yc_vec,yerr=error_mat[:,3],fmt='b.',ecolor='b')
        ax4b.plot(cube.lbda, yfit, 'g')
        ax4b.plot(cube.lbda, yguess, 'k--')
        ax4b.set_xlabel("Wavelength [A]")
        ax4b.set_ylabel("Y center [spaxels]")
        pylab.setp(ax4b.get_xticklabels()+ax4b.get_yticklabels(), fontsize=8)

        ax4c = fig4.add_subplot(2, 1, 2, aspect='equal', adjustable='datalim')
        ax4c.errorbar(xc_vec, yc_vec,xerr=error_mat[:,2],yerr=error_mat[:,3],
                      fmt=None, ecolor='g')
        ax4c.scatter(xc_vec, yc_vec,c=cube.lbda[::-1],
                     cmap=matplotlib.cm.Spectral, zorder=3)
        ax4c.plot(xguess, yguess, 'k--')
        ax4c.plot(xfit, yfit, 'g')
        ax4c.text(0.03, 0.85,
                  r'$\rm{Guess:}\hspace{0.5} x_{0}=%4.2f,\hspace{0.5} ' \
                  r'y_{0}=%4.2f,\hspace{0.5} \delta=%5.2f,\hspace{0.5} ' \
                  r'\theta=%6.2f^\circ$' % \
                  (xc, yc, delta, theta/S.pi*180),
                  transform=ax4c.transAxes)
        ax4c.text(0.03, 0.75,
                  r'$\rm{Fit:}\hspace{0.5} x_{0}=%4.2f,\hspace{0.5} ' \
                  r'y_{0}=%4.2f,\hspace{0.5} \delta=%5.2f,\hspace{0.5} ' \
                  r'\theta=%6.2f^\circ$' % \
                  (fitpar[2], fitpar[3], fitpar[0], fitpar[1]/S.pi*180),
                  transform=ax4c.transAxes)
        ax4c.set_xlabel("X center [spaxels]")
        ax4c.set_ylabel("Y center [spaxels]")
        fig4.text(0.5, 0.93, "ADR plot [%s, airmass=%.2f]" % (obj, airmass),
                  horizontalalignment='center', size='large')

        # Plot of the other model parameters ---------------------------------

        print_msg("Producing model parameter plot %s..." % plot6,
                  opts.verbosity, 1)

        guess_disp = 0
        fit_disp = 0
        for i in xrange(len(alpha)):
            guess_disp += alpha[::-1][i] * lbda_rel**i
            fit_disp   += fitpar[6+i] * lbda_rel**i

        def confidence_interval(lbda_rel, cov, index):
            ci = 0
            for i,j in enumerate(index):
                ci += cov[j][j]*lbda_rel**i
            return S.sqrt(ci)

        confidence_alpha = confidence_interval(lbda_rel,cov,range(6,npar_psf))
        confidence_ell   = confidence_interval(lbda_rel,cov,[4])
        confidence_PA    = confidence_interval(lbda_rel,cov,[5])

        fig6 = pylab.figure()

        ax6a = fig6.add_subplot(2, 1, 1)
        ax6a.errorbar(cube.lbda, alpha_vec,error_mat[:,6],
                      fmt='b.', ecolor='blue', label="Fit 2D")
        ax6a.plot(cube.lbda, guess_disp, 'k--', label="Guess 3D")
        ax6a.plot(cube.lbda, fit_disp, 'g', label="Fit 3D")
        ax6a.plot(cube.lbda, fit_disp + confidence_alpha, 'g:',
                  label='_nolegend_')
        ax6a.plot(cube.lbda, fit_disp - confidence_alpha, 'g:',
                  label='_nolegend_')

        ax6a.text(0.03, 0.8,
                  r'$\rm{Guess coeffs:}\hspace{0.5} %s$' % \
                  (','.join([ 'a_%d=%.2f' % (i,a)
                              for i,a in enumerate(alpha[::-1]) ]) ),
                  transform=ax6a.transAxes, fontsize=11)
        ax6a.text(0.03, 0.7,
                  r'$\rm{Fit coeffs:}\hspace{0.5} %s$' % \
                  (','.join([ 'a_%d=%.2f' % (i,a)
                              for i,a in enumerate(fitpar[6:6+alphaDeg+1]) ])),
                  transform=ax6a.transAxes, fontsize=11)

        leg = ax6a.legend(loc='best')
        pylab.setp(leg.get_texts(), fontsize='smaller')
        ax6a.set_ylabel(r'$\alpha$')
        ax6a.set_xticklabels([])
        ax6a.set_title("Model parameters [%s, seeing %.2f'' FWHM]" % \
                       (obj,seeing))

        def plot_non_chromatic_param(ax, par_vec, lbda, guess_par, fitpar,
                                     str_par, error_vec=None, confidence=None):
            if error_vec is not None:
                ax.errorbar(lbda, par_vec, error_vec, fmt='b.', ecolor='blue')
            else:
                ax.plot(lbda, par_vec, 'bo')
            ax.axhline(guess_par, color='k', linestyle='--')
            ax.axhline(fitpar, color='g', linestyle='-')
            if confidence is not None:
                ax.plot(lbda, fitpar+confidence, 'g:')
                ax.plot(lbda, fitpar-confidence, 'g:')
            ax.set_ylabel(r'$%s$' % (str_par))
            ax.text(0.03, 0.7,
                    r'$\rm{Guess:}\hspace{0.5} %s = %4.2f,\hspace{0.5} ' \
                    r'\rm{Fit:}\hspace{0.5} %s = %4.2f$' % \
                    (str_par, guess_par, str_par, fitpar),
                    transform=ax.transAxes, fontsize=11)
            pylab.setp(ax.get_yticklabels(), fontsize=8)

        ax6c = fig6.add_subplot(4, 1, 3)
        plot_non_chromatic_param(ax6c, ell_vec,cube.lbda,ell,fitpar[4],'1/q',
                                 error_vec=error_mat[:,4],
                                 confidence=confidence_ell)
        ax6c.set_xticklabels([])
        
        ax6d = fig6.add_subplot(4, 1, 4)
        plot_non_chromatic_param(ax6d, PA_vec/S.pi*180, cube.lbda,
                                 PA/S.pi*180, fitpar[5]/S.pi*180, 'PA',
                                 error_vec=error_mat[:,5]/S.pi*180,
                                 confidence=confidence_PA)
        ax6d.set_xlabel("Wavelength [A]")

        # Plot of the radial profile -----------------------------------------

        print_msg("Producing radial profile plot %s..." % plot7,
                  opts.verbosity, 1)

        fig7 = pylab.figure()

        fig7.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97, )
        for i in xrange(cube.nslice):   # Loop over slices
            ax = fig7.add_subplot(nrow, ncol, i+1)
            # Use Elliptical radius instead of plain radius
            #r = S.hypot(cube.x-xfit[i],cube.y-yfit[i])
            dx,dy = cube.x-xfit[i],cube.y-yfit[i]
            r = S.sqrt(dx**2 + fitpar[4]*dy**2 + 2*fitpar[5]*dx*dy)
            ax.plot(r, cube.data[i], 'b.')
            #rfit = S.hypot(cube_fit.x-xfit[i],cube_fit.y-yfit[i])
            dx,dy = cube_fit.x-xfit[i],cube_fit.y-yfit[i]
            rfit = S.sqrt(dx**2 + fitpar[4]*dy**2 + 2*fitpar[5]*dx*dy)
            ax.plot(rfit, cube_fit.data[i], 'r,')
            ax.plot(rfit, psf[i], 'g,')
            ax.plot(rfit, bkg[i],'c,')
            ax.semilogy()
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.9,0.8, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='right', transform=ax.transAxes)
            if method!='PSF':
                ax.axvline(radius, color='y')
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Elliptical radius [spaxels]", fontsize=8)
                ax.set_ylabel("Flux", fontsize=8)
            ax.axis([0, r.max()*1.1, 
                     cube.data[i][cube.data[i]>0].min()/1.2,
                     cube.data[i].max()*1.2])

        # Contour plot of each slice -----------------------------------------

        print_msg("Producing PSF contour plot %s..." % plot8,
                  opts.verbosity, 1)

        fig8 = pylab.figure()

        fig8.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97, )
        extent = (cube.x.min()-0.5,cube.x.max()+0.5,
                  cube.y.min()-0.5,cube.y.max()+0.5)
        for i in xrange(cube.nslice):   # Loop over meta-slices
            ax = fig8.add_subplot(ncol, nrow, i+1, aspect='equal')
            data = cube.slice2d(i, coord='p')
            fit = cube_fit.slice2d(i, coord='p')
            vmin,vmax = pylab.prctile(fit, (5.,95.)) # Percentiles
            lev = S.logspace(S.log10(vmin),S.log10(vmax),5)
            ax.contour(data, lev, origin='lower', extent=extent)
            cnt = ax.contour(fit, lev, ls='--', origin='lower', extent=extent)
            pylab.setp(cnt.collections, linestyle='dotted')
            ax.errorbar((xc_vec[i],),(yc_vec[i],),
                        xerr=(error_mat[i,2],),yerr=(error_mat[i,3],),
                        fmt=None, ecolor='k')
            ax.plot((xfit[i],),(yfit[i],), 'g+')
            if opts.method != 'PSF':
                ax.add_patch(matplotlib.patches.Circle((xfit[i],yfit[i]),
                                                       radius, fc=None, ec='g'))
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.1,0.1, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='left', transform=ax.transAxes)
            ax.axis(extent)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I", fontsize=8)
                ax.set_ylabel("J", fontsize=8)

        # Residuals of each slice --------------------------------------------

        print_msg("Producing residuals plot %s..." % plot5,
                  opts.verbosity, 1)

        fig5 = pylab.figure()

        fig5.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97, )
        for i in xrange(cube.nslice):   # Loop over meta-slices
            ax   = fig5.add_subplot(ncol, nrow, i+1, aspect='equal')
            data = cube.slice2d(i, coord='p')
            var  = cube.slice2d(i, coord='p', var=True)
            fit  = cube_fit.slice2d(i, coord='p')
            res  = S.nan_to_num((data - fit)/S.sqrt(var))
            vmin,vmax = pylab.prctile(res, (3.,97.)) # Percentiles
            ax.imshow(res, origin='lower', extent=extent, vmin=vmin, vmax=vmax)
            ax.plot((xfit[i],),(yfit[i],), 'k+')
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.1,0.1, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='left', transform=ax.transAxes)
            ax.axis(extent)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I", fontsize=8)
                ax.set_ylabel("J", fontsize=8)

        if opts.graph=='pylab':
            pylab.show()
        else:
            fig1.savefig(plot1)
            fig2.savefig(plot2)
            fig3.savefig(plot3)
            fig4.savefig(plot4)
            fig6.savefig(plot6)
            fig7.savefig(plot7)
            fig8.savefig(plot8)
            fig5.savefig(plot5)

# End of extract_star.py =====================================================
