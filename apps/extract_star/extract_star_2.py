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
(extract_star.py).  This version of extract_star only replaces double
gaussian PSF profile by an empirical PSF profile (Gaussain + Moffat).
"""

__author__ = "Clement BUTON"
__version__ = '$Id$'

import os
import sys
import optparse

import pySNIFS
import pySNIFS_fit

import pyfits

import scipy as S
from scipy.ndimage import filters as F

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
    P *= 760./1013.25                   # Convert P to mmHg
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

def fit_param_hdr(hdr,param,lbda_ref,cube):
    
    hdr.update('ES_VERS' ,__version__)
    hdr.update('ES_CUBE' ,cube,       'extract_star input cube')
    hdr.update('ES_DELTA',param[0],   'extract_star ADR power')
    hdr.update('ES_THETA',param[1],   'extract_star ADR angle')
    hdr.update('ES_LREF' ,lbda_ref,   'extract_star ref. lambda')    
    hdr.update('ES_X0'   ,param[2],   'extract_star ref. x0')
    hdr.update('ES_Y0'   ,param[3],   'extract_star ref. y0')
    hdr.update('ES_A1'   ,param[4],   'extract_star ref. a1')
    hdr.update('ES_A0'   ,param[5],   'extract_star ref. a0')    
    hdr.update('ES_ELL'  ,param[6],   'extract_star ref. ellipticity')
    hdr.update('ES_ROT'  ,param[7],   'extract_star ref. rotation')

def linear_fit_PSFnCte(param, weight, cube, model):

    # Fit on masked data
    psf = S.array(model.comp(param), dtype='d')
    alpha = S.sum(weight*psf**2, axis=1)
    beta = S.sum(weight*psf, axis=1)
    gamma = S.sum(weight*psf*cube.data, axis=1)
    delta = S.sum(weight, axis=1)
    epsilon = S.sum(weight*cube.data, axis=1)
    det = (beta**2 - alpha*delta)

    if (det==0).any():                  # Some pb in the fit, return 0 instead of NaN
        idx = (det==0)
        frac = 1.*len(alpha[idx])/len(alpha)
        print "%d/%d px [%.0f%%] cannot be extracted" % \
              (len(alpha[idx]), len(alpha), frac*100)
        # You can try to recover on non-null values (e.g. det[idx]=1;
        # alpha[idx]=0) but that's pretty useless because the PSF model is
        # wrong anyway. You can return S.zeros((5, cube.nslice), 'd'), but
        # that will only produce blank spectra while the extraction is known
        # to have failed. Or you can raise an error so that nothing is
        # produced.

        if frac<0.01:                   # Try to recover if less than 1%
            alpha[idx] = 0
            beta[idx] = 0
            gamma[idx] = 0
            delta[idx] = 0
            epsilon[idx] = 0
            det[idx] = 1
        else:
            raise ValueError("Cannot extract valid spectrum from %s" % \
                             cube_file)

    obj = (beta*epsilon - delta*gamma)/det
    sky = (beta*gamma - alpha*epsilon)/det
    hess = S.zeros((len(obj),2,2),'d')
    hess[:,0,0] = delta
    hess[:,0,1] = beta
    hess[:,1,0] = beta
    hess[:,1,1] = alpha
    cofact_hess = S.zeros((len(obj),2,2),'d')
    cofact_hess[:,0,0] = -delta
    cofact_hess[:,0,1] = beta
    cofact_hess[:,1,0] = beta
    cofact_hess[:,1,1] = -alpha
    
    cov = S.transpose(S.transpose(cofact_hess,(2,1,0))/det,(2,0,1))/2
    var_obj = cov[:,0,0]
    var_sky = cov[:,1,1]
    # This is were one could implement optimal extraction. - YC

    return obj,sky,var_obj,var_sky
    
def comp_spec(cube_file, psf_param, intpar=[None, None]):

    cube = pySNIFS.SNIFS_cube(cube_file)
    cube.x /= intpar[0]  # x in spaxel
    cube.y /= intpar[0]  # y in spaxel
    inhdr = pyfits.getheader(cube_file, 1)
    # DIRTY PATCH TO REMOVE BAD SPECTRA FROM THEIR VARIANCE
    cube.var[cube.var>1e20] = 0
    model = my_psf_function(intpar, cube)
    
    # The PSF parameters are only the shape parameters. We set the intensity
    # of each slice to 1.
    param = psf_param.tolist() + [1.]*cube.nslice    

    # Rejection of bad points
    lapl = F.laplace(cube.data/cube.data.mean())
    fdata = F.median_filter(cube.data, size=[1, 3])  # i don't know for what this line is used (CB)
    hist = pySNIFS.histogram(S.ravel(S.absolute(lapl)), nbin=100,
                             Max=100, cumul=True)
    threshold = hist.x[S.argmax(S.where(hist.data<0.9999, 0, 1))]
    cube.var *= (S.absolute(lapl) <= threshold)
    weight = S.where(cube.var!=0, 1./cube.var, 0)

    obj,sky,var_obj,var_sky = linear_fit_PSFnCte(param, weight, cube, model)

    # The 3D psf model is not normalized to 1 in integral. The result must be
    # renormalized by (eps), eps = eta*2*S.pi*sigma**2 / (S.sqrt(ell)) + S.pi*alpha**2 / ((beta-1)*(S.sqrt(ell)))
    
    s1,s0,b1,b0,e1,e0 = [0.215,0.545,0.345,1.685,0.0,1.04]             #only long exposure
    
    lbda_rel = model.l[:,0]/model.lbda_ref

    alpha    = psf_param[4] * lbda_rel + psf_param[5]
    beta     = b1*alpha+b0
    sigma    = s1*alpha+s0
    eta      = e1*alpha+e0
    ell      = psf_param[6]
    
    eps = eta*2*S.pi*sigma**2 / (S.sqrt(ell)) + S.pi*alpha**2 / ((beta-1)*(S.sqrt(ell)))

    obj *= eps
    var_obj *= eps**2

    # Change sky normalization from 'per spaxel' to 'per arcsec**2'
    sky /= intpar[0]**2                 # intpar[0] is spaxel width
    var_sky /= intpar[0]**4
    
    spec = S.zeros((5, cube.nslice), 'd')
    spec[0,:] = cube.lbda
    spec[1,:] = obj
    spec[2,:] = sky    
    spec[3,:] = var_obj
    spec[4,:] = var_sky
  
    return spec

class my_psf_function:
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
        self.name = 'my_PSF_function'
        self.x = S.zeros((cube.nslice,cube.nlens),'d')
        self.y = S.zeros((cube.nslice,cube.nlens),'d')
        self.l = S.zeros(S.shape(S.transpose(cube.data)),'d')
        self.x[:][:] = cube.x
        self.y[:][:] = cube.y
        self.l[:][:] = cube.lbda
        self.l = S.transpose(self.l)
        self.n_ref = 1e-6*(64.328 + 29498.1/(146.-1./(self.lbda_ref*1e-4)**2) + 255.4/(41.-1./(self.lbda_ref*1e-4)**2)) + 1.
        self.ADR_coef = 206265*(1e-6*(64.328 + 29498.1/(146.-1./(self.l*1e-4)**2) + 255.4/(41.-1./(self.l*1e-4)**2)) + 1. - self.n_ref) / 0.43 #ADR in spaxels
        
    def comp(self,param):
        """
        Compute the function.
        @param param: Input parameters of the polynomial. A list of numbers:
                - C{param[0:8]}: The 8 parameters of the PSF shape
                     - C{param[0]}: Atmospheric dispersion power
                     - C{param[1]}: Atmospheric dispersion position angle
                     - C{param[2]}: X center at the reference wavelength
                     - C{param[3]}: Y center at the reference wavelength
                     - C{param[4]}: Moffat radius coefficient
                     - C{param[5]}: Moffat radius origin 
                     - C{param[6]}: Ellipticity
                     - C{param[7]}: Rotation
                - C{param[8:]} : The Intensity parameters (one for each slice in the cube).
        """
        self.param = param

        # ADR params
        delta = self.param[0]
        theta = self.param[1]
        costheta = S.cos(theta)
        sintheta = S.sin(theta)
        xref = self.param[2]
        yref = self.param[3]        
        x0 = delta*self.ADR_coef*costheta + xref
        y0 = delta*self.ADR_coef*sintheta + yref

        # other params + correlations params (fixed)
        a1 = self.param[4]
        a0 = self.param[5] 
        ellipticity = self.param[6]
        rotation = self.param[7]
        s1,s0,b1,b0,e1,e0 = [0.215,0.545,0.345,1.685,0.0,1.04]                 #only long exposure

        # aliases
        lbda_rel = self.l / self.lbda_ref
        x = self.x
        y = self.y
        xrel  = x - x0
        yrel  = y - y0
        alpha = a1 * lbda_rel + a0
        beta  = b1*alpha+b0
        sigma = s1*alpha+s0
        eta   = e1*alpha+e0
        ellipse = xrel**2 + ellipticity*yrel**2 + 2*rotation*xrel*yrel
        
        es = S.exp(-ellipse/(2*sigma**2))
        ea = (1+ellipse/alpha**2)

        # function
        return S.reshape(param[8:],(len(param[8:]),1))*( eta*es + ea**(-beta) )
    
    def deriv(self,param):
        """
        Compute the derivative of the function with respect to its parameters.
        @param param: Input parameters of the polynomial. A list numbers (see L{SNIFS_psf_3D.comp}).
        @param correlation: Input parameters psf correlations. A list of 6 numbers.
        """
        self.param = param
        grad = S.zeros((self.npar_cor+self.npar_ind,)+S.shape(self.x),'d')

        # ADR params
        delta = self.param[0]
        theta = self.param[1]
        costheta = S.cos(theta)
        sintheta = S.sin(theta)
        xref = self.param[2]
        yref = self.param[3]
        x0 = delta*self.ADR_coef*costheta + xref
        y0 = delta*self.ADR_coef*sintheta + yref

        # other params + correlations params (fixed)
        a1 = self.param[4]
        a0 = self.param[5] 
        ell = self.param[6]
        rot = self.param[7]
        s1,s0,b1,b0,e1,e0 = [0.215,0.545,0.345,1.685,0.0,1.04]  #only long exposure

        # aliases
        lbda_rel = self.l / self.lbda_ref
        x = self.x
        y = self.y
        xrel = x - x0
        yrel = y - y0
        alpha = a1 * lbda_rel + a0
        beta  = b1*alpha+b0
        sigma = s1*alpha+s0
        eta   = e1*alpha+e0
        ellipse = xrel**2 + ell*yrel**2 + 2*rot*xrel*yrel

        es = S.exp(-ellipse/(2*sigma**2))
        ea = (1+ellipse/alpha**2)
        lnea = S.log(ea)

        # derivatives
        grad[2] = eta*es*(xrel + rot*yrel)/sigma**2 + \
                  ea**(-beta)*(-beta)/(ea*alpha**2)*(-2*xrel-2*rot*yrel)
        grad[3] = eta*es*(ell*yrel + rot*xrel)/sigma**2 + \
                  ea**(-beta)*(-beta)/(ea*alpha**2)*(-2*ell*yrel-2*rot*xrel)
        grad[0] =       self.ADR_coef*(costheta*grad[2] + sintheta*grad[3])
        grad[1] = delta*self.ADR_coef*(costheta*grad[3] - sintheta*grad[2])
        grad[4] = e1*es*lbda_rel + eta*ellipse*s1*es*lbda_rel/sigma**3 + ea**(-beta)*(-b1*lbda_rel*lnea-(-2*beta*ellipse*lbda_rel/(ea*alpha**3)))
        grad[5] = e1*es + eta*ellipse*s1*es/sigma**3 + ea**(-beta)*(-b1*lnea-(-2*beta*ellipse/(ea*alpha**3)))        
        grad[6] = -0.5*eta*es*yrel**2/sigma**2 + ea**(-beta)*(-beta)/(ea*alpha**2)*yrel**2
        grad[7] = -eta*es*yrel*xrel/sigma**2 + 2*ea**(-beta)*(-beta)/(ea*alpha**2)*yrel*xrel
        grad[8] = eta*es + ea**(-beta) 

        grad[0:8] = grad[0:8] * S.reshape(param[8:],(1,len(param[8:]),1))
        
        return grad

# ########## MAIN ##############################
    
if __name__ == "__main__":

    # Options ==============================

    usage = "usage: [%prog] [options] -i inE3D.fits " \
            "-o outSpec.fits -s outSky.fits"

    parser = optparse.OptionParser(usage, version=__version__)
    parser.add_option("-i", "--in", type="string", dest="input", 
                      help="Input datacube (euro3d format)")
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

    print_msg("  Object: %s, Airmass: %.2f" % (obj,airmass),
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
    
    nslice  = cube.nslice
    xc_vec  = S.zeros(cube.nslice, dtype='d')
    yc_vec  = S.zeros(cube.nslice, dtype='d')
    a0_vec  = S.zeros(cube.nslice, dtype='d')
    a1_vec  = S.zeros(cube.nslice, dtype='d')
    ell_vec = S.zeros(cube.nslice, dtype='d')
    rot_vec = S.zeros(cube.nslice, dtype='d')
    int_vec = S.zeros(cube.nslice, dtype='d')
    sky_vec = S.zeros(cube.nslice, dtype='d')

    sky_sup = []
    n = 0     
    
    for i in xrange(cube.nslice):
        if opts.verbosity >= 1:
            sys.stdout.write('\rSlice %2d/%d' % (i+1, cube.nslice))
            sys.stdout.flush()
        print_msg("", opts.verbosity, 2)
        cube2 = pySNIFS.SNIFS_cube()
        cube2.nslice = 1
        cube2.nlens = cube.nlens
        cube2.data = cube.data[i, S.newaxis]
        cube2.x = cube.x
        cube2.y = cube.y
        cube2.i = cube.i
        cube2.j = cube.j
        cube2.lbda = S.array([cube.lbda[i]])
        cube2.var = cube.var[i, S.newaxis]

        # Guess parameters for the current slice
        sky = (cube2.data[0]+3*S.sqrt(cube2.var[0])).min()  # Background
        sl_int = F.median_filter(cube2.data[0], 3)          # Centroid
        imax = sl_int.max()                                 # Intensity
        sl_int -= sky
        xc = S.average(cube2.x, weights=sl_int)
        yc = S.average(cube2.y, weights=sl_int)

        # Filling in the guess parameter arrays (px) and bounds arrays (bx)
        p1 = [0., 0., xc, yc, 0., 2.4, 1., 0., imax]        # my_psf_function;0.43
        b1 = [None]*(8+cube2.nslice)                        # Empty list of length 8+cube2.nslice
        b1[0:8] = [[None, None],                            # delta
                   [-S.pi, S.pi],                           # theta
                   [None, None],                            # x0
                   [None, None],                            # y0
                   [0.,0.],                                 # a1
                   [0.1, None],                             # a0 
                   [.6, 2.5],                               # ellipticity 
                   [None, None]]                            # rotation   
        b1[8:8+cube2.nslice] = [[0, None]] * cube2.nslice   
                                                            
        p2 = [sky]                                          # poly2D;0
        #b2 = [[0.005, sky_sup[i]]]
        b2 = [[0.005, None]]
        
        print_msg("    Initial guess: %s" % [p1,p2], opts.verbosity, 2)        
        
        # Instanciating of a model class
        lbda_ref = cube2.lbda[0]
        sl_model = pySNIFS_fit.model(data=cube2,
                                     func=['my_psf_function;0.43, %f' % lbda_ref,
                                           'poly2D;0'],
                                     param=[p1,p2], bounds=[b1,b2],
                                     myfunc={'my_psf_function':my_psf_function})

        # Fit of the current slice
        if opts.verbosity >= 3:
            sl_model.fit(maxfun=400, msge=1)
        else:
            sl_model.fit(maxfun=400 )

        # Storing the result of the current slice parameters
        xc_vec[i]  = sl_model.fitpar[2]
        yc_vec[i]  = sl_model.fitpar[3]
        a1_vec[i]  = sl_model.fitpar[4]
        a0_vec[i]  = sl_model.fitpar[5]       
        ell_vec[i] = sl_model.fitpar[6]
        rot_vec[i] = sl_model.fitpar[7]
        int_vec[i] = sl_model.fitpar[8]
        sky_vec[i] = sl_model.fitpar[9]

        print_msg("    Fit result: %s" %sl_model.fitpar, opts.verbosity, 2)
        
    print_msg("", opts.verbosity, 1)

    # 3D model fitting ==============================
        
    print_msg("Datacube 3D-fitting...", opts.verbosity, 0)
    
    # Computing the initial guess for the 3D fitting from the results of the
    # slice by slice 2D fit
    lbda_ref = cube.lbda.mean()
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
                       atmosphericIndex(lbda_ref)) / 0.43 # In spaxels

    polADR = pySNIFS.fit_poly(yc_vec2, 3, 1, xc_vec2)
    theta = S.arctan(polADR(1))
    x0 = xc_vec2[S.argmin(S.absolute(lbda_ref - cube.lbda[ind]))]
    y0 = polADR(x0)
    
    delta_x_vec = ((xc_vec2-x0)/(S.cos(theta)*ADR_coef[ind]))[ADR_coef[ind]!=0]
    delta_y_vec = ((yc_vec2-y0)/(S.sin(theta)*ADR_coef[ind]))[ADR_coef[ind]!=0]
    if theta == 0:
        delta = S.median(delta_x_vec)
    elif theta == S.pi/2.:
        delta = S.median(delta_y_vec)
    else:
        delta_x = S.median(delta_x_vec)
        delta_y = S.median(delta_y_vec)
        delta = S.mean([delta_x, delta_y])

    # 2) Other parameters:
    polAlpha = pySNIFS.fit_poly(a0_vec,3,1,cube.lbda/lbda_ref)
    a0 = polAlpha.coeffs[1]
    a1 = polAlpha.coeffs[0]
    ell = S.median(ell_vec)
    rot = S.median(rot_vec)

    # Filling in the guess parameter arrays (px) and bounds arrays (bx)
    p1 = [None]*(8+cube.nslice)
    b1 = [None]*(8+cube.nslice)
    p1[0:8] = [delta, theta, x0, y0, a1, a0, ell, rot]
    p1[8:8+cube.nslice] = int_vec.tolist()
    
    b1[0:8] = [[None, None],           # delta      
               [None, None],           # theta      
               [None, None],           # x0         
               [None, None],           # y0         
               #[-2., 0.],              # a1         
               #[0.1, None],            # a0         
               [None, None],           # a1         
               [None, None],           # a0         
               [.6, 2.5],              # ellipticity               
               [None, None]]           # rotation   
    b1[8:8+cube.nslice] = [[0, None]] * cube.nslice

    p2 = sky_vec.tolist()
    b2 = [[0.005, None]] * cube.nslice

    print_msg("  Initial guess: %s" % p1[:11], opts.verbosity, 2)
    
    # Instanciating the model class
    data_model = pySNIFS_fit.model(data=cube,
                                   func=['my_psf_function;0.43, %f' % lbda_ref,
                                         'poly2D;0'],
                                   param=[p1,p2], bounds=[b1,b2],
                                   myfunc={'my_psf_function':my_psf_function})
    
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
    #print_msg("  Seeing estimate: %.2f arcsec FWHM" % (fitpar[4]*2.355), opts.verbosity, 0)

    # Computing final spectra for object and background =====================
    
    print_msg("Extracting the spectrum...", opts.verbosity, 0)
    spec = comp_spec(opts.input, fitpar[0:8], intpar=[0.43, lbda_ref])

    # Save star spectrum ==============================
    
    step = inhdr.get('CDELTS')
    
    fit_param_hdr(inhdr,data_model.fitpar,lbda_ref,opts.input)
    star_spec = pySNIFS.spectrum(data=spec[1],
                                 start=spec[0][0],step=step)
    star_spec.WR_fits_file(opts.out,header_list=inhdr.items())
    star_var = pySNIFS.spectrum(data=spec[3],
                                 start=spec[0][0],step=step)
    star_var.WR_fits_file('var_'+opts.out,header_list=inhdr.items())
    
    # Save sky spectrum ==============================

    sky_spec = pySNIFS.spectrum(data=spec[2],
                                start=spec[0][0],step=step)
    sky_spec.WR_fits_file(opts.sky,header_list=inhdr.items())
    sky_var = pySNIFS.spectrum(data=spec[4],
                                start=spec[0][0],step=step)
    sky_var.WR_fits_file('var_'+opts.sky,header_list=inhdr.items())

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
        plot5 = os.path.extsep.join((basename+"_fit4", opts.graph))
        plot6 = os.path.extsep.join((basename+"_fit5", opts.graph))
        plot7 = os.path.extsep.join((basename+"_fit6", opts.graph))
        plot8 = os.path.extsep.join((basename+"_fit7", opts.graph))
        
        # Plot of the star and sky spectra ------------------------------
        
        print_msg("Producing spectra plot %s..." % plot1, opts.verbosity, 1)
        
        fig1 = pylab.figure()
        axS = fig1.add_subplot(3, 1, 1)
        axB = fig1.add_subplot(3, 1, 2)
        axN = fig1.add_subplot(3, 1, 3)
        axS.plot(spec[0], spec[1], 'b')
        axS.set_title("Star spectrum [%s]" % obj)
        axS.set_xlim(spec[0][0],spec[0][-1])
        axS.set_xticklabels([])
        axB.plot(spec[0], spec[2], 'g')
        axB.set_xlim(spec[0][0],spec[0][-1])
        axB.set_title("Background spectrum (per spx)")
        axB.set_xticklabels([])
        axN.plot(spec[0], S.sqrt(spec[3]), 'b')
        axN.plot(spec[0], S.sqrt(spec[4]), 'g')
        axN.set_title("Error spectra")
        axN.semilogy()
        axN.set_xlim(spec[0][0],spec[0][-1])
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

        func1 = my_psf_function(intpar=[data_model.func[0].pix,
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
        ax4a.legend(loc='best')

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
        
        guess_disp = a0+a1*cube.lbda/lbda_ref
        fit_disp   = fitpar[5]+fitpar[4]*cube.lbda/lbda_ref
        th_disp    = (fitpar[4]+fitpar[5])*(cube.lbda/lbda_ref)**(-0.2)

        fig6 = pylab.figure()
        ax6a = fig6.add_subplot(2, 1, 1)
        ax6a.plot(cube.lbda, a0_vec, 'bo', label="Fit 2D")
        ax6a.plot(cube.lbda, guess_disp, 'k--', label="Guess 3D")
        ax6a.plot(cube.lbda, fit_disp, 'b', label="Fit 3D")
        ax6a.plot(cube.lbda, th_disp, 'g', label="Theoretical")
        ax6a.legend(loc='best')
        ax6a.set_ylabel(r'$\alpha$')
        ax6a.text(0.03, 0.8,
                  r'$\rm{Guess:}\hspace{0.5} a_0,a_1 = %4.2f,%4.2f,\hspace{0.5} ' \
                  r'\rm{Fit:}\hspace{0.5} a_0,a_1 = %4.2f,%4.2f$' % \
                  (a0,a1, fitpar[5],fitpar[4]),
                  transform=ax6a.transAxes, fontsize=11)
        ax6a.set_xticklabels([])
        ax6a.set_title("Model parameters [%s]" % obj)
        
        ax6c = fig6.add_subplot(4, 1, 3)
        plot_non_chromatic_param(ax6c, ell_vec, cube.lbda, ell, fitpar[6],'1/q')
        ax6c.set_xticklabels([])
        ax6d = fig6.add_subplot(4, 1, 4)
        plot_non_chromatic_param(ax6d, rot_vec, cube.lbda, rot, fitpar[7], '\\theta')
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
            #ax.axhline(sky_sup[i],ls='--',c='k')
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
        extent = (cube.x.min(),cube.x.max(),cube.y.min(),cube.y.max())
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
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I", fontsize=8)
                ax.set_ylabel("J", fontsize=8)
        fig8.savefig(plot8)

