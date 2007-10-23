#!/usr/bin/env python
######################################################################
## Filename:      extract_star_2.py
## Version:       $Revision$
## Description:   Standard star spectrum extraction
## Author:        $Author$
## $Id$
######################################################################

"""
Primarily based on the point source extractor of Emmanuel Pecontal
(extract_star.py).  This version of extract_star replaces double
gaussian PSF profile by an empirical PSF profile (Gaussian + Moffat).
"""

__author__ = "Clement BUTON"
__version__ = '$Id$'

import os
import sys
import optparse
import copy

import pyfits
import pySNIFS
import pySNIFS_fit

import scipy as S
from scipy import linalg as L
from scipy import interpolate as I 
from scipy.ndimage import filters as F

# Definitions ========================================================

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
    P *= 0.75006168                     # Convert P to mmHg: *= 760./1013.25
    n = 1 + (n-1) * P * \
        ( 1 + (1.049 - 0.0157*T)*1e-6*P ) / \
        ( 720.883*(1 + 0.003661*T) )

    return n
    
def plot_non_chromatic_param(ax, par_vec, lbda, guess_par, fitpar, str_par):

    ax.plot(lbda, par_vec, 'bo')
    ax.axhline(guess_par, color='k', linestyle='--')
    ax.axhline(fitpar, color='b', linestyle='-')
    ax.set_ylabel(r'$%s$' % (str_par))
    ax.text(0.03, 0.7,
            r'$\rm{Guess:}\hspace{0.5} %s = %4.2f,\hspace{0.5} ' \
            r'\rm{Fit:}\hspace{0.5} %s = %4.2f$' % \
            (str_par, guess_par, str_par, fitpar),
            transform=ax.transAxes, fontsize=11)
    pylab.setp(ax.get_yticklabels(), fontsize=8)

def fit_param_hdr(hdr,param,lbda_ref,cube, sky_deg):
    
    hdr.update('ES_VERS' ,__version__)
    hdr.update('ES_CUBE' ,cube,       'extract_star input cube')
    hdr.update('ES_DELTA',param[0],   'extract_star ADR power')
    hdr.update('ES_THETA',param[1],   'extract_star ADR angle')
    hdr.update('ES_LREF' ,lbda_ref,   'extract_star ref. lambda')    
    hdr.update('ES_X0'   ,param[2],   'extract_star ref. x0')
    hdr.update('ES_Y0'   ,param[3],   'extract_star ref. y0')
    hdr.update('ES_A0'   ,param[4],   'extract_star ref. a0')    
    hdr.update('ES_A1'   ,param[5],   'extract_star ref. a1')
    hdr.update('ES_ELL'  ,param[6],   'extract_star ref. ellipticity')
    hdr.update('ES_ROT'  ,param[7],   'extract_star ref. rotation')
    hdr.update('ES_SDEG', sky_deg,    'extract_star polynomial bkgnd degree')

def comp_spec(cube, psf_param, efftime, intpar=[None, None],poly_deg=0):
    
    npar_poly = int((poly_deg+1)*(poly_deg+2)/2)                # Number of parameters of the polynomial background
    
    # DIRTY PATCH TO REMOVE BAD SPECTRA FROM THEIR VARIANCE
    cube.x /= intpar[0]                                         # x in spaxel
    cube.y /= intpar[0]                                         # y in spaxel
    cube.var[cube.var>1e20] = 0
    if efftime > 5:
        model = long_exposure_psf(intpar, cube)
        s1,s0,b1,b0,e1,e0 = [0.215,0.545,0.345,1.685,0.0,1.04]  # long exposures
    else:
        model = short_exposure_psf(intpar, cube)
        s1,s0,b1,b0,e1,e0 = [0.2,0.56,0.415,1.395,0.16,0.6]     # short exposures
    
    # The PSF parameters are only the shape parameters. We set the intensity
    # of each slice to 1.
    param = psf_param.tolist() + [1.]*cube.nslice    

    # Rejection of bad points
    lapl = F.laplace(cube.data/cube.data.mean())
    fdata = F.median_filter(cube.data, size=[1, 3])
    hist = pySNIFS.histogram(S.ravel(S.absolute(lapl)), nbin=100,
                             Max=100, cumul=True)
    threshold = hist.x[S.argmax(S.where(hist.data<0.9999, 0, 1))]
    cube.var *= (S.absolute(lapl) <= threshold)
    weight = S.sqrt(S.where(cube.var!=0, 1./cube.var, 0))

    # Fit on masked data*
    psf = S.array(model.comp(param), dtype='d')
    X = S.zeros((cube.nslice,cube.nlens,npar_poly+1),'d')
    X[:,:,0] = psf*weight
    X[:,:,1] = weight
    n = 2
    for d in S.arange(poly_deg)+1:
        for j in S.arange(d+1):
            X[:,:,n] = weight*cube.x**(d-j)*cube.y**j
            n=n+1

    Norm = S.mean(cube.data)
    cube.data = cube.data / Norm
    cube.var = cube.var / Norm**2
    A = S.array([S.dot(S.transpose(x),x) for x in X])
    b = weight*cube.data
    B = S.array([S.dot(S.transpose(X[i]),b[i]) for i in S.arange(cube.nslice)])
    C = S.array([L.inv(a) for a in A])
    D = S.array([L.solve(A[i],B[i]) for i in S.arange(cube.nslice)])
##     D = S.array([pySNIFS_fit.fnnls(A[i],B[i])[0] for i in S.arange(cube.nslice)])
    V = S.array([S.diag(c) for c in C])
    
##     obj = D[:,0]
##     sky0 = D[:,1]
##     sky_slope = S.sqrt(D[:,2]**2 + D[:,3]**2)*S.sign(D[:,2]/D[:,3])
##     sky_orient = S.arctan(D[:,2]/D[:,3])*180/S.pi
##     var_obj = V[:,0]
##     var_sky0 = V[:,1]
##     var_sky_slope = (D[:,2]*sqrt(V[:,2]) + D[:,3]*sqrt(V[:,3]))**2 / sky_slope**2
##     var_sky_orient = (180/S.pi)**2*(D[:,3]*sqrt(V[:,2]) + D[:,2]*sqrt(V[:,3]))**2 / sky_slope**4

    # The 3D psf model is not normalized to 1 in integral. The result must be
    # renormalized by (eps), eps = eta*2*S.pi*sigma**2 / (S.sqrt(ell)) + S.pi*alpha**2 / ((beta-1)*(S.sqrt(ell)))

    lbda_rel = model.l[:,0] / model.lbda_ref
    alpha    = psf_param[4]*lbda_rel**psf_param[5]
    beta     = b1*alpha+b0
    sigma    = s1*alpha+s0
    eta      = e1*alpha+e0
    ell      = psf_param[6]
    
    eps = S.pi*(2*eta*sigma**2 + alpha**2/(beta-1) )/S.sqrt(ell)    
    
    D[:,0] *= eps
    V[:,0] *= eps**2

    # Change sky normalization from 'per spaxel' to 'per arcsec**2'
    D[:,0] /= intpar[0]**2                 # intpar[0] is spaxel width
    V[:,0] /= intpar[0]**4
    
##     spec = S.zeros((2*npar_poly, cube.nslice), 'd')
##     spec[0,:] = cube.lbda
##     spec[1:npar_poly+2,:] = D
##     spec[npar_poly+2:,:] =  V 
  
    return cube.lbda,D,V

def get_start(cube,poly_deg,verbosity,efftime):
    npar_poly = int((poly_deg+1)*(poly_deg+2)/2)            # Number of parameters of the polynomial background
    n = 2
    if n>7:
        raise ValueError('The number of edge pixels should be less than 7')
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
    
    xc_vec  = S.zeros(cube.nslice, dtype='d')
    yc_vec  = S.zeros(cube.nslice, dtype='d')
    a0_vec  = S.zeros(cube.nslice, dtype='d')
    a1_vec  = S.zeros(cube.nslice, dtype='d')
    ell_vec = S.zeros(cube.nslice, dtype='d')
    rot_vec = S.zeros(cube.nslice, dtype='d')
    int_vec = S.zeros(cube.nslice, dtype='d')
    sky_vec = S.zeros((cube.nslice,npar_poly), dtype='d')    

    for i in xrange(cube.nslice):
        cube_sky.var   = copy.deepcopy(cube.var[i,  S.newaxis])
        cube_sky.data  = copy.deepcopy(cube.data[i, S.newaxis])
        cube_star.var  = copy.deepcopy(cube.var[i,  S.newaxis])
        cube_star.data = copy.deepcopy(cube.data[i, S.newaxis])
        cube_star.lbda = S.array([cube.lbda[i]])
        if opts.verbosity >= 1:
            sys.stdout.write('\rSlice %2d/%d' % (i+1, cube.nslice))
            sys.stdout.flush()
        print_msg("", opts.verbosity, 2)
        
        #Fit a 2D polynomial of degree poly_deg on the edge pixels of a given cube slice.  
        ind = S.where((cube_sky.i<n)|(cube_sky.i>=15-n)|(cube_sky.j<n)|(cube_sky.j>=15-n))[0]
        p0 = S.median(S.transpose(cube_sky.data)[ind])[0]
        ind = S.where((cube.i>=n)&(cube.i<15-n)&(cube.j>=n)&(cube.j<15-n))[0]
        S.transpose(cube_sky.var)[ind] = 0.
        model_sky = pySNIFS_fit.model(data=cube_sky,func=['poly2D;%d'%poly_deg],\
                                      param=[[p0]+[0.]*(npar_poly-1)],\
                                      bounds=[[[0,None]]+[[None,None]]*(npar_poly-1)])
        model_sky.fit()
        cube_sky.data = model_sky.evalfit()
        
        # Guess parameters for the current slice
        star_int = F.median_filter(cube_star.data[0], 3) 
        star_int = S.absolute(star_int - cube_sky.data[0])
        imax = star_int.max()                               # Intensity
        xc = S.average(cube_star.x, weights=star_int)       # Centroid
        yc = S.average(cube_star.y, weights=star_int)
        cube_star.data = cube_star.data - cube_sky.data

        # Filling in the guess parameter arrays (px) and bounds arrays (bx)
        p = [0., 0., xc, yc, 2., 0, 1., 0., imax]           # psf function;0.43
        b = [None]*(8+cube_star.nslice)                     # Empty list of length 8+cube2.nslice
        b[0:8] = [[None, None],                             # delta
                   [-S.pi, S.pi],                           # theta
                   [None, None],                            # x0
                   [None, None],                            # y0
                   [None, None],                            # a0 
                   [0, 0],                                  # a1
                   [.6, 2.5],                               # ellipticity 
                   [None, None]]                            # rotation   
        b[8:8+cube_star.nslice] = [[0, None]] * cube_star.nslice   

        print_msg("    Initial guess: %s" % [p], opts.verbosity, 2)        
        
        # Instanciating of a model class
        lbda_ref = cube_star.lbda[0]

        if efftime > 5:
            model_star = pySNIFS_fit.model(data=cube_star,
                                           func=['long_exposure_psf;0.43, %f' % lbda_ref],
                                           param=[p], bounds=[b],
                                           myfunc={'long_exposure_psf':long_exposure_psf})

        else:
            model_star = pySNIFS_fit.model(data=cube_star,
                                           func=['short_exposure_psf;0.43, %f' % lbda_ref],
                                           param=[p], bounds=[b],
                                           myfunc={'short_exposure_psf':short_exposure_psf})        

        # Fit of the current slice
        if opts.verbosity >= 3:
            model_star.fit(maxfun=400, msge=1)
        else:
            model_star.fit(maxfun=400 )

        # Storing the result of the current slice parameters
        xc_vec[i]  = model_star.fitpar[2]
        yc_vec[i]  = model_star.fitpar[3]
        a0_vec[i]  = model_star.fitpar[4]       
        a1_vec[i]  = model_star.fitpar[5]
        ell_vec[i] = model_star.fitpar[6]
        rot_vec[i] = model_star.fitpar[7]
        int_vec[i] = model_star.fitpar[8]
        sky_vec[i] = model_sky.fitpar

        print_msg("    Fit result: %s" % model_star.fitpar, opts.verbosity, 2)

    return xc_vec,yc_vec,a0_vec,a1_vec,ell_vec,rot_vec,int_vec,sky_vec

def build_sky_cube(cube,sky,sky_var,deg):

    npar_poly = len(sky)
    poly = pySNIFS_fit.poly2D(deg,cube)
    cube2 = pySNIFS.zerolike(cube)
    cube2.x = (cube2.x)**2
    cube2.y = (cube2.y)**2
    poly2 = pySNIFS_fit.poly2D(deg,cube2)
    param = S.zeros((npar_poly,cube.nslice),'d')
    vparam = S.zeros((npar_poly,cube.nslice),'d')
    for i in S.arange(npar_poly):
        param[i,:] = sky[i].data
        vparam[i,:] = sky_var[i].data
    data = poly.comp(S.ravel(param))
    var = poly2.comp(S.ravel(vparam))
    bkg_cube = pySNIFS.zerolike(cube)
    bkg_cube.data = data
    bkg_cube.var = var
    bkg_spec = bkg_cube.get_spec(no=bkg_cube.no)

    return bkg_cube,bkg_spec

# PSF classes ========================================================

class long_exposure_psf:
    """
    Empirical PSF 3D function used by the L{model} class.
    """
    def __init__(self,intpar=[None,None],cube=None):
        """
        Initiating the class.
        @param intpar: Internal parameters (pixel size in cube spatial unit and reference wavelength). A
            list of two numbers.
        @param cube: Input cube. This is a L{SNIFS_cube} object.
        """
        self.pix = intpar[0]
        self.lbda_ref = intpar[1]
        self.npar_ind = 1
        self.npar_cor = 8
        self.npar = self.npar_ind*cube.nslice + self.npar_cor
        self.name = 'long_exposure_psf'
        self.x = S.zeros((cube.nslice,cube.nlens),'d')
        self.y = S.zeros((cube.nslice,cube.nlens),'d')
        self.l = S.zeros(cube.data.T.shape,'d')
        self.x[:][:] = cube.x
        self.y[:][:] = cube.y
        self.l[:][:] = cube.lbda
        self.l = S.transpose(self.l)
        self.n_ref = atmosphericIndex(self.lbda_ref)
        self.ADR_coef = 206265*(atmosphericIndex(self.l) - self.n_ref) / 0.43 #ADR in spaxels

    def comp(self,param):
        """
        Compute the function.
        @param param: Input parameters of the polynomial. A list of numbers:
                - C{param[0:8]}: The 8 parameters of the PSF shape
                     - C{param[0]}: Atmospheric dispersion power
                     - C{param[1]}: Atmospheric dispersion position angle
                     - C{param[2]}: X center at the reference wavelength
                     - C{param[3]}: Y center at the reference wavelength
                     - C{param[4]}: Moffat radius norm a0 
                     - C{param[5]}: Moffat radius power a1
                     - C{param[6]}: Ellipticity
                     - C{param[7]}: Rotation
                - C{param[8:]} : The Intensity parameters (one for each slice in the cube).
        """

        self.param = S.asarray(param)

        # ADR params
        delta = self.param[0]
        theta = self.param[1]
        xref  = self.param[2]
        yref  = self.param[3]
        x0 = delta*self.ADR_coef*S.cos(theta) + xref
        y0 = delta*self.ADR_coef*S.sin(theta) + yref

        # other params
        a0  = self.param[4] 
        a1  = self.param[5]
        ell = self.param[6]
        rot = self.param[7]

        # aliases + correlations params (fixed)
        lbda_rel = self.l / self.lbda_ref
        alpha = a0*lbda_rel**a1

        s1,s0,b1,b0,e1,e0 = [0.215,0.545,0.345,1.685,0.0,1.04] # Long exposures
        beta  = b1*alpha+b0
        sigma = s1*alpha+s0
        eta   = e1*alpha+e0

        dx  = self.x - x0
        dy  = self.y - y0
        r2 = dx**2 + ell*dy**2 + 2*rot*dx*dy
        gaussian = S.exp(-r2/2/sigma**2)
        ea = 1 + r2/alpha**2
        moffat = ea**(-beta)

        # function
        return self.param[8:,S.newaxis] * ( eta*gaussian + moffat )
    
    def deriv(self,param):
        """
        Compute the derivative of the function with respect to its parameters.
        @param param: Input parameters of the polynomial. A list numbers (see L{SNIFS_psf_3D.comp}).
        @param correlation: Input parameters psf correlations. A list of 6 numbers.
        """
        self.param = S.asarray(param)
        grad = S.zeros((self.npar_cor+self.npar_ind,)+S.shape(self.x),'d')

        # ADR params
        delta = self.param[0]
        theta = self.param[1]
        xref  = self.param[2]
        yref  = self.param[3]
        costheta = S.cos(theta)
        sintheta = S.sin(theta)
        x0 = delta*self.ADR_coef*costheta + xref
        y0 = delta*self.ADR_coef*sintheta + yref

        # other params
        a0  = self.param[4] 
        a1  = self.param[5]
        ell = self.param[6]
        rot = self.param[7]

        # aliases + correlations params (fixed)
        lbda_rel = self.l / self.lbda_ref
        alpha = a0*lbda_rel**a1

        s1,s0,b1,b0,e1,e0 = [0.215,0.545,0.345,1.685,0.0,1.04] # Long exposures
        beta  = b1*alpha+b0
        sigma = s1*alpha+s0
        eta   = e1*alpha+e0
        
        dx = self.x - x0
        dy = self.y - y0
        r2 = dx**2 + ell*dy**2 + 2*rot*dx*dy
        gaussian = S.exp(-r2/2/sigma**2)
        ea = 1 + r2/alpha**2
        moffat = ea**(-beta)
        logea = S.log(ea)

        # derivatives
        tmp = eta*gaussian/sigma**2 + 2*beta*moffat/ea/alpha**2
        grad[2] = tmp*(    dx + rot*dy)
        grad[3] = tmp*(ell*dy + rot*dx)
        grad[0] =       self.ADR_coef*(costheta*grad[2] + sintheta*grad[3])
        grad[1] = delta*self.ADR_coef*(costheta*grad[3] - sintheta*grad[2])
        grad[4] = gaussian * lbda_rel**a1 * ( e1 + eta*r2*s1/sigma**3 ) + \
                  moffat*( -b1 * lbda_rel**a1 * logea + 2*beta*r2/(a0*ea*alpha**2) )
        grad[5] = grad[4] * a0 * S.log(lbda_rel)
        grad[6] = -tmp/2 * dy**2
        grad[7] = -tmp   * dx*dy
        grad[8] = eta*gaussian + moffat

        grad[0:8] *= self.param[S.newaxis,8:,S.newaxis]
        
        return grad

class short_exposure_psf:
    """
    Empirical PSF 3D function used by the L{model} class.
    """
    def __init__(self,intpar=[None,None],cube=None):
        """
        Initiating the class.
        @param intpar: Internal parameters (pixel size in cube spatial unit and reference wavelength). A
            list of two numbers.
        @param cube: Input cube. This is a L{SNIFS_cube} object.
        """
        self.pix = intpar[0]
        self.lbda_ref = intpar[1]
        self.npar_ind = 1
        self.npar_cor = 8
        self.npar = self.npar_ind*cube.nslice + self.npar_cor
        self.name = 'short_exposure_psf'
        self.x = S.zeros((cube.nslice,cube.nlens),'d')
        self.y = S.zeros((cube.nslice,cube.nlens),'d')
        self.l = S.zeros(cube.data.T.shape,'d')
        self.x[:][:] = cube.x
        self.y[:][:] = cube.y
        self.l[:][:] = cube.lbda
        self.l = S.transpose(self.l)
        self.n_ref = atmosphericIndex(self.lbda_ref)
        self.ADR_coef = 206265*(atmosphericIndex(self.l) - self.n_ref) / 0.43 #ADR in spaxels

    def comp(self,param):
        """
        Compute the function.
        @param param: Input parameters of the polynomial. A list of numbers:
                - C{param[0:8]}: The 8 parameters of the PSF shape
                     - C{param[0]}: Atmospheric dispersion power
                     - C{param[1]}: Atmospheric dispersion position angle
                     - C{param[2]}: X center at the reference wavelength
                     - C{param[3]}: Y center at the reference wavelength
                     - C{param[4]}: Moffat radius norm a0
                     - C{param[5]}: Moffat radius power a1
                     - C{param[6]}: Ellipticity
                     - C{param[7]}: Rotation
                - C{param[8:]} : The Intensity parameters (one for each slice in the cube).
        """

        self.param = S.asarray(param)

        # ADR params
        delta = self.param[0]
        theta = self.param[1]
        xref  = self.param[2]
        yref  = self.param[3]
        x0 = delta*self.ADR_coef*S.cos(theta) + xref
        y0 = delta*self.ADR_coef*S.sin(theta) + yref

        # other params
        a0  = self.param[4] 
        a1  = self.param[5]
        ell = self.param[6]
        rot = self.param[7]

        # aliases + correlations params (fixed)
        lbda_rel = self.l / self.lbda_ref
        alpha = a0*lbda_rel**a1

        s1,s0,b1,b0,e1,e0 = [0.2,0.56,0.415,1.395,0.16,0.6] # Short exposures
        beta  = b1*alpha+b0
        sigma = s1*alpha+s0
        eta   = e1*alpha+e0

        dx  = self.x - x0
        dy  = self.y - y0
        r2 = dx**2 + ell*dy**2 + 2*rot*dx*dy
        gaussian = S.exp(-r2/2/sigma**2)
        ea = 1 + r2/alpha**2
        moffat = ea**(-beta)

        # function
        return self.param[8:,S.newaxis] * ( eta*gaussian + moffat )
    
    def deriv(self,param):
        """
        Compute the derivative of the function with respect to its parameters.
        @param param: Input parameters of the polynomial. A list numbers (see L{SNIFS_psf_3D.comp}).
        @param correlation: Input parameters psf correlations. A list of 6 numbers.
        """
        self.param = S.asarray(param)
        grad = S.zeros((self.npar_cor+self.npar_ind,)+S.shape(self.x),'d')

        # ADR params
        delta = self.param[0]
        theta = self.param[1]
        xref  = self.param[2]
        yref  = self.param[3]
        costheta = S.cos(theta)
        sintheta = S.sin(theta)
        x0 = delta*self.ADR_coef*costheta + xref
        y0 = delta*self.ADR_coef*sintheta + yref

        # other params
        a0  = self.param[4] 
        a1  = self.param[5]
        ell = self.param[6]
        rot = self.param[7]

        # aliases + correlations params (fixed)
        lbda_rel = self.l / self.lbda_ref
        alpha = a0*lbda_rel**a1

        s1,s0,b1,b0,e1,e0 = [0.2,0.56,0.415,1.395,0.16,0.6] # Short exposures
        beta  = b1*alpha+b0
        sigma = s1*alpha+s0
        eta   = e1*alpha+e0
        
        dx = self.x - x0
        dy = self.y - y0
        r2 = dx**2 + ell*dy**2 + 2*rot*dx*dy
        gaussian = S.exp(-r2/2/sigma**2)
        ea = 1 + r2/alpha**2
        moffat = ea**(-beta)
        logea = S.log(ea)

        # derivatives
        tmp = eta*gaussian/sigma**2 + 2*beta*moffat/ea/alpha**2
        grad[2] = tmp*(    dx + rot*dy)
        grad[3] = tmp*(ell*dy + rot*dx)
        grad[0] =       self.ADR_coef*(costheta*grad[2] + sintheta*grad[3])
        grad[1] = delta*self.ADR_coef*(costheta*grad[3] - sintheta*grad[2])
        grad[4] = gaussian * lbda_rel**a1 * ( e1 + eta*r2*s1/sigma**3 ) + \
                  moffat*( -b1 * lbda_rel**a1 * logea + 2*beta*r2/(a0*ea*alpha**2) )
        grad[5] = grad[4] * a0 * S.log(lbda_rel)
        grad[6] = -tmp/2 * dy**2
        grad[7] = -tmp   * dx*dy
        grad[8] = eta*gaussian + moffat

        grad[0:8] *= self.param[S.newaxis,8:,S.newaxis]
        
        return grad

# ########## MAIN ##############################
    
if __name__ == "__main__":

    # Options ==============================

    usage = "usage: [%prog] [options] -i inE3D.fits " \
            "-o outSpec.fits -s outSky.fits"

    parser = optparse.OptionParser(usage, version=__version__)
    parser.add_option("-i", "--in", type="string", dest="input", 
                      help="Input datacube (euro3d format)")
    parser.add_option("-d", "--deg", type="int", dest="sky_deg", 
                      help="Sky polynomial background degree [%default]",
                      default=0 )
    parser.add_option("-o", "--out", type="string", 
                      help="Output star spectrum")
    parser.add_option("-s", "--sky", type="string",
                      help="Output sky spectrum")
    parser.add_option("-p", "--plot", action='store_true',
                      help="Plot flag (syn. '--graph=png')")
    parser.add_option("-g", "--graph", type="string",
                      help="Graphic output format (e.g. 'eps' or 'png') " \
                      "[%default]", default="png")
    parser.add_option("-v", "--verbosity", type="int",
                      help="Verbosity level (<0: quiet) [%default]",
                      default=0)
    
    opts,pars = parser.parse_args()
    if not opts.input or not opts.out or not opts.sky:
        parser.error("At least one option is missing among " \
                     "'--in', '--out' and '--sky'.")    
    if opts.plot:
        opts.graph = 'png'
        
    npar_poly = int((opts.sky_deg+1)*(opts.sky_deg+2)/2)  # Number of parameters of the polynomial background
    
    # Input datacube ==============================
    
    print_msg("Opening datacube %s" % opts.input, opts.verbosity, 0)
    inhdr = pyfits.getheader(opts.input, 1) # 1st extension
    obj = inhdr.get('OBJECT', 'Unknown')
    efftime = inhdr.get('EFFTIME')
    airmass = inhdr.get('AIRMASS', 0.0)    
    channel = inhdr.get('CHANNEL', 'Unknown').upper()

    if channel.startswith('B'):
        slices=[10, 900, 65]
    elif channel.startswith('R'):
        slices=[10, 1500, 130]
    else:
        parser.error("Input datacube %s has no valid CHANNEL keyword (%s)" % \
                     (opts.input, channel))
    if efftime > 5.:                    # Long exposure
        psfFn = long_exposure_psf
    else:                               # Short exposure
        psfFn = short_exposure_psf

    print_msg("  Object: %s, Airmass: %.2f; Efftime: %.1fs [%s]" % \
              (obj, airmass, efftime, efftime>5 and 'long' or 'short'),
              opts.verbosity, 0)
    print_msg("  Channel: %s, extracting slices: %s" % (channel,slices),
              opts.verbosity, 0)
    
    cube = pySNIFS.SNIFS_cube(opts.input, slices=slices)
    cube.data = S.array(cube.data,'d')
    cube.var = S.array(cube.var,'d')
    cube.x = S.array(cube.x,'d') / 0.43
    cube.y = S.array(cube.y,'d') / 0.43
    cube.lbda =S.array(cube.lbda,'d')

    print_msg("  Meta-slices before selection: %d " \
              "from %.2f to %.2f by %.2f A" % \
              (len(cube.lbda), cube.lbda[0], cube.lbda[-1],
               cube.lbda[1]-cube.lbda[0]), opts.verbosity, 1)

    # Normalisation of the signal and variance in order to avoid numerical
    # problems with too small numbers
    norm = cube.data.mean()
    cube.data /= norm
    cube.var /= norm**2
    
    # Rejection of bad points ==============================

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
        
    # Computing guess parameters from slice by slice 2D fit =================
    
    print_msg("Slice-by-slice 2D-fitting...", opts.verbosity, 0)
    
    xc_vec,yc_vec,a0_vec,a1_vec,ell_vec,rot_vec,int_vec,sky_vec = get_start(cube,opts.sky_deg,0,efftime)
        
    print_msg("", opts.verbosity, 1)

    # 3D model fitting ==============================
        
    print_msg("Datacube 3D-fitting...", opts.verbosity, 0)
    
    # Computing the initial guess for the 3D fitting from the results of the
    # slice by slice 2D fit
    lbda_ref = cube.lbda.mean()
    nslice = cube.nslice
    # 1) Position parameters:
    #    the xc, yc vectors obtained from 2D fit are smoothed, then the
    #    position corresponding to the reference wavelength is read in the
    #    filtered vectors. Finally, the parameters theta and delta are
    #    determined from the xc, yc vectors.
    ind = ( S.absolute(xc_vec)<7 ) & ( S.absolute(yc_vec)<7 )
    if not ind.all():                   # Some centroids outside FoV
        print "%d/%d centroid positions discarded from ADR initial guess" % \
              (len(xc_vec[-ind]),nslice)
        if (len(xc_vec[ind])<=1):
            raise ValueError('Not enough points to determine ADR initial guess')
    
    xc_vec2 = xc_vec[ind]
    yc_vec2 = yc_vec[ind]
    ADR_coef = 206265*(atmosphericIndex(cube.lbda) -
                       atmosphericIndex(lbda_ref)) / 0.43          # In spaxels

    polADR = pySNIFS.fit_poly(yc_vec2, 3, 1, xc_vec2)
    theta = S.arctan(polADR(1))                                    # YC: seems buggy
    #theta = S.arctan2(polADR(1),-1)                               # YC: not better...
    x0 = xc_vec2[S.argmin(S.absolute(lbda_ref - cube.lbda[ind]))]
    y0 = polADR(x0)
    
    delta = S.tan(S.arccos(1./airmass))

    # 2) Other parameters:
    a0  = S.median(a0_vec*(cube.lbda/lbda_ref)**(-1/3.))
    ell = S.median(ell_vec)
    rot = S.median(rot_vec)

    # Filling in the guess parameter arrays (px) and bounds arrays (bx)
    p1 = [None]*(8+nslice)
    b1 = [None]*(8+nslice)
    p1[0:8] = [delta, theta, x0, y0, a0, -0.33, ell, rot]
    p1[8:8+nslice] = int_vec.tolist()
    
    b1[0:8] = [[None, None],           # delta      
               [None, None],           # theta      
               [None, None],           # x0         
               [None, None],           # y0         
               [None, None],           # a0         
               [None, None],           # a1         
               [.6, 2.5],              # ellipticity               
               [None, None]]           # rotation   
    b1[8:8+nslice] = [[0, None]] * nslice

    p2 = S.ravel(S.transpose(sky_vec.tolist()))
    b2 = ([[0.,None]]+[[None,None]]*(npar_poly-1))*nslice    

    print_msg("  Initial guess: %s" % p1[:11], opts.verbosity, 2)
    
    # Instanciating the model class
    data_model = pySNIFS_fit.model(data=cube,
                                   func=['%s;0.43,%f' % \
                                         (psfFn.__name__,lbda_ref),
                                         'poly2D;%d' % opts.sky_deg],
                                   param=[p1,p2],
                                   bounds=[b1,b2],
                                   myfunc={psfFn.__name__:psfFn})
        
    guesspar = data_model.flatparam
    
    # The fit is launched twice. This is a dirty trick to avoid it to get
    # quickly stuck on a bad solution...
    if opts.verbosity >= 3:
        data_model.fit(maxfun=400, save=True, msge=1) 
        data_model.fit(msge=1)
    else:
        data_model.fit(maxfun=400, save=True) 
        data_model.fit()
        
    # Storing result and guess parameters
    fitpar = data_model.fitpar

    print_msg("  Fit result: %s" % fitpar[:8], opts.verbosity, 2)
    #print_msg("  Seeing estimate: %.2f arcsec FWHM" % (1.55 * fitpar[4]**0.6 * lbda_ref**(0.6*fitpar[5])), opts.verbosity, 0)

    # Computing final spectra for object and background =====================
    
    print_msg("Extracting the spectrum...", opts.verbosity, 0)

    full_cube = pySNIFS.SNIFS_cube(opts.input)
    lbda,spec,var = comp_spec(full_cube, fitpar[0:8], efftime, intpar=[0.43, lbda_ref], poly_deg=opts.sky_deg)
    npar_poly = int((opts.sky_deg+1)*(opts.sky_deg+2)/2) 

    # Save star spectrum ==============================
    
    step = inhdr.get('CDELTS')
    
    fit_param_hdr(inhdr,data_model.fitpar,lbda_ref,opts.input,opts.sky_deg)
    star_spec = pySNIFS.spectrum(data=spec[:,0],start=lbda[0],step=step)
    star_spec.WR_fits_file(opts.out,header_list=inhdr.items())
    star_var = pySNIFS.spectrum(data=var[:,0],start=lbda[0],step=step)
    star_var.WR_fits_file('var_'+opts.out,header_list=inhdr.items())
    
    # Save sky spectrum ==============================

    inhdr = pyfits.getheader(opts.input, 1) # 1st extension
    fit_param_hdr(inhdr,data_model.fitpar,lbda_ref,opts.input,opts.sky_deg)
    sky_spec = pySNIFS.spec_list([pySNIFS.spectrum(data=s,start=lbda[0],step=step) for s in S.transpose(spec)[1:]])
    sky_spec.WR_fits_file(opts.sky,header_list=inhdr.items())
    sky_var = pySNIFS.spec_list([pySNIFS.spectrum(data=v,start=lbda[0],step=step) for v in S.transpose(var)[1:]])
    sky_var.WR_fits_file('var_'+opts.sky,header_list=inhdr.items())

    bkg_cube,bkg_spec = build_sky_cube(full_cube,sky_spec.list,sky_var.list,opts.sky_deg)

    # Create output graphics ==============================
    
    if opts.plot:

        import matplotlib
        matplotlib.use('Agg')
        import pylab

        basename = os.path.splitext(opts.out)[0]
        plot1 = os.path.extsep.join((basename+"_plt" , opts.graph))
        plot2 = os.path.extsep.join((basename+"_fit1", opts.graph))
        plot3 = os.path.extsep.join((basename+"_fit2", opts.graph))
        plot4 = os.path.extsep.join((basename+"_fit3", opts.graph))
        plot6 = os.path.extsep.join((basename+"_fit4", opts.graph))
        plot7 = os.path.extsep.join((basename+"_fit5", opts.graph))
        plot8 = os.path.extsep.join((basename+"_fit6", opts.graph))
        plot5 = os.path.extsep.join((basename+"_fit7", opts.graph))
        
        # Plot of the star and sky spectra ------------------------------
        
        print_msg("Producing spectra plot %s..." % plot1, opts.verbosity, 1)
        
        fig1 = pylab.figure()
        axS = fig1.add_subplot(3, 1, 1)
        axB = fig1.add_subplot(3, 1, 2)
        axN = fig1.add_subplot(3, 1, 3)
        axS.plot(star_spec.x, star_spec.data, 'b')
        axS.set_title("Star spectrum [%s]" % obj)
        axS.set_xlim(star_spec.x[0],star_spec.x[-1])
        axS.set_xticklabels([])
        bkg_spec.data /= cube.nlens
        bkg_spec.var  /= cube.nlens**2
        axB.plot(bkg_spec.x, bkg_spec.data, 'g')
        axB.set_xlim(bkg_spec.x[0],bkg_spec.x[-1])
        axB.set_title("Background spectrum (per spx)")
        axB.set_xticklabels([])
        axN.plot(star_spec.x, S.sqrt(star_var.data), 'b')
        axN.plot(bkg_spec.x, S.sqrt(bkg_spec.var), 'g')
        axN.set_title("Error spectra")
        axN.semilogy()
        axN.set_xlim(star_spec.x[0],star_spec.x[-1])
        axN.set_xlabel("Wavelength [A]")
        fig1.savefig(plot1)
        
        # Plot of the fit on each slice ------------------------------
        
        print_msg("Producing slice fit plot %s..." % plot2, opts.verbosity, 1)

        ncol = S.floor(S.sqrt(cube.nslice))
        nrow = S.ceil(cube.nslice/float(ncol))
        
        fig2 = pylab.figure()
        fig2.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97, )
        for i in xrange(cube.nslice):   # Loop over meta-slices
            ax = fig2.add_subplot(nrow, ncol, i+1)
            data = data_model.data.data[i,:]
            fit = data_model.evalfit()[i,:]
            fmin = min(data.min(), fit.min()) - 2e-2
            ax.plot(data-fmin)          # Signal
            ax.plot(fit-fmin)           # Fit
            ax.semilogy()
            ax.set_xlim(0,len(data))
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.1,0.1, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='left', transform=ax.transAxes)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Spaxel ID", fontsize=8)
                ax.set_ylabel("Flux + cte", fontsize=8)
        fig2.savefig(plot2)

        # Plot of the fit on rows and columns sum ----------------------------
        
        print_msg("Producing profile plot %s..." % plot3, opts.verbosity, 1)

        # Creating a standard SNIFS cube with the adjusted data
        cube_fit = pySNIFS.SNIFS_cube(lbda=cube.lbda)
        cube_fit.x /= 0.43     # x in spaxel 
        cube_fit.y /= 0.43     # y in spaxel

        func1 = psfFn(intpar=[data_model.func[0].pix,
                              data_model.func[0].lbda_ref],
                      cube=cube_fit)            
        func2 = pySNIFS_fit.poly2D(0, cube_fit)
        
        cube_fit.data = func1.comp(fitpar[0:func1.npar]) + \
                        func2.comp(fitpar[func1.npar:func1.npar+func2.npar])

        fig3 = pylab.figure()
        fig3.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97, )
        for i in xrange(cube.nslice):   # Loop over slices
            ax = fig3.add_subplot(nrow, ncol, i+1)
            # YC - Why is there some NaN's in data slices?
            # (eg e3d_TC07_153_099_003_17_B.fits)
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
        fig3.savefig(plot3)
        
        # Plot of the star center of gravity and adjusted center --------------
        
        print_msg("Producing ADR plot %s..." % plot4, opts.verbosity, 1)

        xfit = fitpar[0] * data_model.func[0].ADR_coef[:,0] * \
               S.cos(fitpar[1]) + fitpar[2]
        yfit = fitpar[0] * data_model.func[0].ADR_coef[:,0] * \
               S.sin(fitpar[1]) + fitpar[3]
        xguess = guesspar[0] * data_model.func[0].ADR_coef[:,0] * \
                 S.cos(guesspar[1]) + guesspar[2]
        yguess = guesspar[0] * data_model.func[0].ADR_coef[:,0] * \
                 S.sin(guesspar[1]) + guesspar[3]

        fig4 = pylab.figure()
        ax4a = fig4.add_subplot(2, 2, 1)
        ax4a.plot(cube.lbda, xc_vec, 'b.', label="Fit 2D")
        ax4a.plot(cube.lbda, xguess, 'k--', label="Guess 3D")
        ax4a.plot(cube.lbda, xfit, 'b', label="Fit 3D")
        ax4a.set_xlabel("Wavelength [A]")
        ax4a.set_ylabel("X center [spaxels]")
        pylab.setp(ax4a.get_xticklabels()+ax4a.get_yticklabels(), fontsize=8)
        leg = ax4a.legend(loc='best')
        pylab.setp(leg.get_texts(), fontsize='smaller')

        ax4b = fig4.add_subplot(2, 2, 2)
        ax4b.plot(cube.lbda, yc_vec, 'b.')
        ax4b.plot(cube.lbda, yfit, 'b')
        ax4b.plot(cube.lbda, yguess, 'k--')
        ax4b.set_xlabel("Wavelength [A]")
        ax4b.set_ylabel("Y center [spaxels]")
        pylab.setp(ax4b.get_xticklabels()+ax4b.get_yticklabels(), fontsize=8)

        ax4c = fig4.add_subplot(2, 1, 2)
        ax4c.scatter(xc_vec, yc_vec,
                     c=cube.lbda[::-1], cmap=matplotlib.cm.Spectral)
        ax4c.plot(xguess, yguess, 'k--')
        ax4c.plot(xfit, yfit, 'b')
        ax4c.text(0.03, 0.85,
                  r'$\rm{Guess:}\hspace{0.5} x_{0}=%4.2f,\hspace{0.5} ' \
                  r'y_{0}=%4.2f,\hspace{0.5} \delta=%5.2f,\hspace{0.5} ' \
                  r'\theta=%6.2f^\circ$' % \
                  (x0, y0, delta, theta*180./S.pi),
                  transform=ax4c.transAxes)
        ax4c.text(0.03, 0.75,
                  r'$\rm{Fit:}\hspace{0.5} x_{0}=%4.2f,\hspace{0.5} ' \
                  r'y_{0}=%4.2f,\hspace{0.5} \delta=%5.2f,\hspace{0.5} ' \
                  r'\theta=%6.2f^\circ$' % \
                  (fitpar[2], fitpar[3], fitpar[0], fitpar[1]*180./S.pi),
                  transform=ax4c.transAxes)
        ax4c.set_xlabel("X center [spaxels]")
        ax4c.set_ylabel("Y center [spaxels]")
        fig4.text(0.5, 0.93, "ADR plot [%s, airmass=%.2f]" % (obj, airmass), 
                  horizontalalignment='center', size='large')
        fig4.savefig(plot4)

        # Plot of the other model parameters ------------------------------
        
        print_msg("Producing model parameter plot %s..." % plot6,
                  opts.verbosity, 1)
        
        guess_disp = a0*(cube.lbda/lbda_ref)**(-1/3.)
        fit_disp   = fitpar[4]*(cube.lbda/lbda_ref)**fitpar[5]
        th_disp    = fitpar[4]*(cube.lbda/lbda_ref)**(-1/3.)

        fig6 = pylab.figure()
        ax6a = fig6.add_subplot(2, 1, 1)
        ax6a.plot(cube.lbda, a0_vec, 'bo', label="Fit 2D")
        ax6a.plot(cube.lbda, guess_disp, 'k--', label="Guess 3D")
        ax6a.plot(cube.lbda, fit_disp, 'b', label="Fit 3D")
        ax6a.plot(cube.lbda, th_disp, 'g', label="Theoretical")
        ax6a.text(0.03, 0.8,
                  r'$\rm{Guess:}\hspace{0.5} a_0=%.2f, a_1=-1/3,\hspace{0.5} ' \
                  r'\rm{Fit:}\hspace{0.5} a_0=%.2f, a_1=%.2f$' % \
                  (a0,fitpar[4],fitpar[5]),
                  transform=ax6a.transAxes, fontsize=11)
        leg = ax6a.legend(loc='best')
        pylab.setp(leg.get_texts(), fontsize='smaller')
        ax6a.set_ylabel(r'$\alpha$')
        ax6a.set_xticklabels([])
        ax6a.set_title("Model parameters [%s]" % obj)
        
        ax6c = fig6.add_subplot(4, 1, 3)
        plot_non_chromatic_param(ax6c, ell_vec, cube.lbda, ell, fitpar[6],'1/q')
        ax6c.set_xticklabels([])
        ax6d = fig6.add_subplot(4, 1, 4)
        plot_non_chromatic_param(ax6d, rot_vec/S.pi*180, cube.lbda,
                                 rot/S.pi*180, fitpar[7]/S.pi*180, '\\theta')
        fig6.savefig(plot6)

        # Plot of the radial profile --------------

        print_msg("Producing radial profile plot %s..." % plot7,
                  opts.verbosity, 1)
        fig7 = pylab.figure()
        fig7.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97, )
        for i in xrange(cube.nslice):   # Loop over slices
            ax = fig7.add_subplot(nrow, ncol, i+1)
            ax.plot(S.hypot(cube.x-xfit[i],cube.y-yfit[i]),
                    cube.data[i], 'b.')
            ax.plot(S.hypot(cube_fit.x-xfit[i],cube_fit.y-yfit[i]),
                    cube_fit.data[i], 'r,')
            ax.plot(S.hypot(cube_fit.x-xfit[i],cube_fit.y-yfit[i]),
                    func1.comp(fitpar[0:func1.npar])[i], 'g,')
            ax.plot(S.hypot(cube_fit.x-xfit[i],cube_fit.y-yfit[i]),
                    func2.comp(fitpar[func1.npar:func1.npar+func2.npar])[i], 'c,')
            ax.semilogy()
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.1,0.1, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='left', transform=ax.transAxes)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Radius [spaxels]", fontsize=8)
                ax.set_ylabel("Flux", fontsize=8)
        fig7.savefig(plot7)

        # Contour plot of each slice ------------------------------
        
        print_msg("Producing PSF contour plot %s..." % plot8,
                  opts.verbosity, 1)

        fig8 = pylab.figure()
        fig8.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97, )
        extent = (cube.x.min()-0.5,cube.x.max()+0.5,
                  cube.y.min()-0.5,cube.y.max()+0.5)
        for i in xrange(cube.nslice):                              # Loop over meta-slices
            ax = fig8.add_subplot(ncol, nrow, i+1, aspect='equal')
            data = cube.slice2d(i, coord='p')
            fit = cube_fit.slice2d(i, coord='p')
            vmin,vmax = pylab.prctile(fit, (5.,95.))               # Percentiles
            lev = S.logspace(S.log10(vmin),S.log10(vmax),5)
            ax.contour(data, lev, origin='lower', extent=extent)
            cnt = ax.contour(fit, lev, ls='--', origin='lower', extent=extent)
            pylab.setp(cnt.collections, linestyle='dotted')
            ax.plot((xfit[i],),(yfit[i],), 'k+')
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.1,0.1, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='left', transform=ax.transAxes)
            ax.axis(extent)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I", fontsize=8)
                ax.set_ylabel("J", fontsize=8)
        fig8.savefig(plot8)

        # Residuals of each slice ------------------------------
        
        print_msg("Producing residuals plot %s..." % plot5,
                  opts.verbosity, 1)

        fig5 = pylab.figure()
        fig5.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97, )
        for i in xrange(cube.nslice):                # Loop over meta-slices
            ax = fig5.add_subplot(ncol, nrow, i+1, aspect='equal')
            data = cube.slice2d(i, coord='p')
            var = cube.slice2d(i, coord='p', var=True)
            fit = cube_fit.slice2d(i, coord='p')
            res = S.nan_to_num((data - fit)/S.sqrt(var)) 
            vmin,vmax = pylab.prctile(res, (3.,97.))     # Percentiles
            ax.imshow(res, origin='lower', extent=extent, vmin=vmin, vmax=vmax)
            ax.plot((xfit[i],),(yfit[i],), 'k+')
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.1,0.1, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='left', transform=ax.transAxes)
            ax.axis(extent)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I", fontsize=8)
                ax.set_ylabel("J", fontsize=8)
        fig5.savefig(plot5)

