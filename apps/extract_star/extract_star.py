#!/usr/bin/env python
######################################################################
## Filename:      extract_star.py
## Version:       $Revision$
## Description:   Standard star spectrum extraction
## Author:        $Author$
## $Id$
######################################################################

__author__ = "Emmanuel Pecontal"
__version__ = '$Id$'

import os
import sys
import optparse
import copy

import pySNIFS
import pySNIFS_fit

import pyfits

import numpy as N
import scipy as S
from scipy import linalg as L
from scipy import interpolate as I 
from scipy.ndimage import filters as F

def print_msg(string, verbosity, limit=0):
    """Print message 'str' if verbosity level >= verbosity limit."""

    if verbosity >= limit:
        print string


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
    #ax.legend(('2D PSF', '3D PSF Guess', '3D PSF Fit'))
    pylab.setp(ax.get_yticklabels(), fontsize=8)

def fit_param_hdr(hdr, param, lbda_ref, cube, sky_deg):

    hdr.update('ES_VERS', __version__)
    hdr.update('ES_CUBE', cube,       'extract_star input cube')
    hdr.update('ES_ALPHA',param[0],   'extract_star ADR power')
    hdr.update('ES_THETA',param[1],   'extract_star ADR angle')
    hdr.update('ES_LREF', lbda_ref,   'extract_star ref. lambda')
    hdr.update('ES_X0',   param[2],   'extract_star ref. x0')
    hdr.update('ES_Y0',   param[3],   'extract_star ref. y0')
    hdr.update('ES_SIGC', param[4],   'extract_star ref. sigma')
    hdr.update('ES_GAMMA',param[5],   'extract_star sigma index')
    hdr.update('ES_Q',    param[6],   'extract_star Q-param.')
    hdr.update('ES_EPS',  param[7],   'extract_star EPS-param.')
    hdr.update('ES_SIGK', param[8],   'extract_star sigma_k')
    hdr.update('ES_QK',   param[9],   'extract_star q_k')
    hdr.update('ES_THETK',param[10],  'extract_star theta_k')
    hdr.update('SEEING',  param[4]*2.355, # Seeing estimate (FWHM in arcsec)
               'extract_star seeing')
    hdr.update('ES_SDEG', sky_deg,       'extract_star sky polynomial background degree')

def comp_spec(cube, psf_param, intpar=[None, None],poly_deg=0):
    npar_poly = int((poly_deg+1)*(poly_deg+2)/2)  # Number of parameters of the polynomial background
    
    # DIRTY PATCH TO REMOVE BAD SPECTRA FROM THEIR VARIANCE
    cube.var[cube.var>1e20] = 0
    model = pySNIFS_fit.SNIFS_psf_3D(intpar, cube)
    # The PSF parameters are only the shape parameters. We set the intensity
    # of each slice to 1.
    param = psf_param.tolist() + [1.]*cube.nslice    

    # Rejection of bad points
    lapl = F.laplace(cube.data/cube.data.mean())
    fdata = F.median_filter(cube.data, size=[1, 3])
    hist = pySNIFS.histogram(N.ravel(N.abs(lapl)), nbin=100,
                             Max=100, cumul=True)
    threshold = hist.x[N.argmax(N.where(hist.data<0.9999, 0, 1))]
    cube.var *= (N.abs(lapl) <= threshold)
    weight = N.sqrt(N.where(cube.var!=0, 1./cube.var, 0))

    # Fit on masked data*
    psf = N.array(model.comp(param), dtype='d')
    X = N.zeros((cube.nslice,cube.nlens,npar_poly+1),'d')
    X[:,:,0] = psf*weight
    X[:,:,1] = weight
    n = 2
    for d in N.arange(poly_deg)+1:
        for j in N.arange(d+1):
            X[:,:,n] = weight*cube.x**(d-j)*cube.y**j
            n=n+1

    A = N.array([N.dot(N.transpose(x),x) for x in X])
    b = weight*cube.data
    B = N.array([N.dot(N.transpose(X[i]),b[i]) for i in N.arange(cube.nslice)])
    C = N.array([L.inv(a) for a in A])
    S = N.array([L.solve(A[i],B[i]) for i in N.arange(cube.nslice)])
    #S = N.array([pySNIFS_fit.fnnls(A[i],B[i])[0] for i in N.arange(cube.nslice)])
    V = N.array([N.diag(c) for c in C])
    
##     obj = S[:,0]
##     sky0 = S[:,1]
##     sky_slope = N.sqrt(S[:,2]**2 + S[:,3]**2)*N.sign(S[:,2]/S[:,3])
##     sky_orient = N.arctan(S[:,2]/S[:,3])*180/N.pi
##     var_obj = V[:,0]
##     var_sky0 = V[:,1]
##     var_sky_slope = (S[:,2]*sqrt(V[:,2]) + S[:,3]*sqrt(V[:,3]))**2 / sky_slope**2
##     var_sky_orient = (180/N.pi)**2*(S[:,3]*sqrt(V[:,2]) + S[:,2]*sqrt(V[:,3]))**2 / sky_slope**4

    # The 3D psf model is not normalized to 1 in integral. The result must be
    # renormalized by (1+eps)
    
    S[:,0] *= 1 + psf_param[7]
    V[:,0] *= (1 + psf_param[7])**2

    # Change sky normalization from 'per spaxel' to 'per arcsec**2'
##     S[:,0] /= intpar[0]**2                 # intpar[0] is spaxel width
##     V[:,0] /= intpar[0]**4
    
##     spec = N.zeros((2*npar_poly, cube.nslice), 'd')
##     spec[0,:] = cube.lbda
##     spec[1:npar_poly+2,:] = S
##     spec[npar_poly+2:,:] =  V 
  
    return cube.lbda,S,V


def get_start(cube,poly_deg,verbosity):
    npar_poly = int((poly_deg+1)*(poly_deg+2)/2)  # Number of parameters of the polynomial background
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
    
    xc_vec   = N.zeros(cube.nslice, dtype='d')
    yc_vec   = N.zeros(cube.nslice, dtype='d')
    sigc_vec = N.zeros(cube.nslice, dtype='d')
    q_vec    = N.zeros(cube.nslice, dtype='d')
    eps_vec  = N.zeros(cube.nslice, dtype='d')
    sigk_vec = N.zeros(cube.nslice, dtype='d')
    qk_vec   = N.zeros(cube.nslice, dtype='d')
    theta_vec= N.zeros(cube.nslice, dtype='d')
    int_vec  = N.zeros(cube.nslice, dtype='d')
    sky_vec  = N.zeros((cube.nslice,npar_poly), dtype='d')
    
    for i in xrange(cube.nslice):
        cube_sky.var = copy.deepcopy(cube.var[i, N.newaxis])
        cube_sky.data = copy.deepcopy(cube.data[i, N.newaxis])
        cube_star.var = copy.deepcopy(cube.var[i, N.newaxis])
        cube_star.data = copy.deepcopy(cube.data[i, N.newaxis])
        cube_star.lbda = N.array([cube.lbda[i]])
        if verbosity >= 1:
            sys.stdout.write('\rSlice %2d/%d' % (i+1, cube.nslice))
            sys.stdout.flush()
        print_msg("", verbosity, 2)
        #print_msg("  Slice %2d/%d" % (i+1, cube.nslice), verbosity, 1)
    
        #Fit a 2D polynomial of degree poly_deg on the edge pixels of a given cube slice.  
        ind = N.where((cube_sky.i<n)|(cube_sky.i>=15-n)|(cube_sky.j<n)|(cube_sky.j>=15-n))[0]
        p0 = N.median(N.transpose(cube_sky.data)[ind])[0]
        ind = N.where((cube.i>=n)&(cube.i<15-n)&(cube.j>=n)&(cube.j<15-n))[0]
        N.transpose(cube_sky.var)[ind] = 0.
        model_sky = pySNIFS_fit.model(data=cube_sky,func=['poly2D;%d'%poly_deg],\
                                      param=[[p0]+[0.]*(npar_poly-1)],\
                                      bounds=[[[0,None]]+[[None,None]]*(npar_poly-1)])
        model_sky.fit()
        cube_sky.data = model_sky.evalfit()
    
        star_int = F.median_filter(cube_star.data[0], 3) 
        star_int = N.abs(star_int - cube_sky.data[0])
        imax = star_int.max()                      # Intensity
        xc = N.average(cube_star.x, weights=star_int)  # Centroid
        yc = N.average(cube_star.y, weights=star_int)
        cube_star.data = cube_star.data - cube_sky.data
        
        # Filling in the guess parameter arrays (px) and bounds arrays (bx)
        p = [0, 0, xc, yc, 0.3, -0.2, 1.84, 0.42, 0.2, 1., 0., imax] # SNIFS_psf_3D;0.43
        b = [None]*(11+cube_star.nslice) # Empty list of length 11+cube_star.nslice
        b[0:11] = [[0, 0],       # alpha
                    [0, 0],      # theta
                    [None, None],       # x0
                    [None, None],       # y0
                    [0.01, None],       # sigc
                    [-0.3, 0],          # alpha (chrom. dependance of sigc)
                    [1.84, 1.84],       # Fixed q
                    [0.42, 0.42],       # Fixed epsilon
                    [0.01, None],       # sigk
                    [1., None],         # qk
                    [0., N.pi]]         # theta_k
        b[11:11+cube_star.nslice] = [[0, None]] * cube_star.nslice

        
        print_msg("    Initial guess: %s" % [p], verbosity, 2)

        # Instanciating of a model class
        lbda_ref = cube_star.lbda[0]
        model_star = pySNIFS_fit.model(data=cube_star,
                                     func=['SNIFS_psf_3D;0.43, %f' % lbda_ref],
                                     param=[p], bounds=[b])

        # Fit of the current slice
        if verbosity >= 3:
            model_star.fit(maxfun=200, msge=1)
        else:
            model_star.fit(maxfun=200)

        # Storing the result of the current slice parameters
        xc_vec[i]   = model_star.fitpar[2]
        yc_vec[i]   = model_star.fitpar[3]
        sigc_vec[i] = model_star.fitpar[4]
        q_vec[i]    = model_star.fitpar[6]
        eps_vec[i]  = model_star.fitpar[7]
        sigk_vec[i] = model_star.fitpar[8]
        qk_vec[i]   = model_star.fitpar[9]
        theta_vec[i]= model_star.fitpar[10]
        int_vec[i]  = model_star.fitpar[11]
        sky_vec[i]  = model_sky.fitpar
        
        print_msg("    Fit result: %s" % \
                  model_star.fitpar, verbosity, 2)
    return xc_vec,yc_vec,sigc_vec,q_vec,eps_vec,sigk_vec,qk_vec,theta_vec,int_vec,sky_vec

def build_sky_cube(cube,sky,sky_var,deg):
    npar_poly = len(sky)
    poly = pySNIFS_fit.poly2D(deg,cube)
    cube2 = pySNIFS.zerolike(cube)
    cube2.x = (cube2.x)**2
    cube2.y = (cube2.y)**2
    poly2 = pySNIFS_fit.poly2D(deg,cube2)
    param = N.zeros((npar_poly,cube.nslice),'d')
    vparam = N.zeros((npar_poly,cube.nslice),'d')
    for i in N.arange(npar_poly):
        param[i,:] = sky[i].data
        vparam[i,:] = sky_var[i].data
    data = poly.comp(N.ravel(param))
    var = poly2.comp(N.ravel(vparam))
    bkg_cube = pySNIFS.zerolike(cube)
    bkg_cube.data = data
    bkg_cube.var = var
    bkg_spec = bkg_cube.get_spec(no=bkg_cube.no)
    return bkg_cube,bkg_spec

# ########## MAIN ##############################

if __name__ == "__main__":

    # Options ==============================

    usage = "usage: [%prog] [options] -i inE3D.fits -d sky_deg" \
            "-o outSpec.fits -s outSky.fits"

    parser = optparse.OptionParser(usage, version=__version__)
    parser.add_option("-i", "--in", type="string", dest="input", 
                      help="Input datacube (euro3d format)")
    parser.add_option("-d", "--deg", type="int", dest="sky_deg", 
                      help="Degree of the sky background polynomial")
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
    airmass = inhdr.get('AIRMASS', 0.0)
    channel = inhdr.get('CHANNEL', 'Unknown').upper()
    if channel.startswith('B'):
        # slices=[10, 900, 20]
        slices=[10, 900, 65]
    elif channel.startswith('R'):
        # slices=[10, 1500, 30]
        slices=[10, 1500, 130]
    else:
        parser.error("Input datacube %s has no valid CHANNEL keyword (%s)" % \
                     (opts.input, channel))

    print_msg("  Object: %s, Airmass: %.2f" % (obj,airmass),
              opts.verbosity, 0)
    print_msg("  Channel: %s, extracting slices: %s" % (channel,slices),
              opts.verbosity, 0)
    
    cube = pySNIFS.SNIFS_cube(opts.input, slices=slices)

    cube.data = N.array(cube.data,'d')
    cube.var = N.array(cube.var,'d')
    cube.x = N.array(cube.x,'d')
    cube.y = N.array(cube.y,'d')
    cube.lbda = N.array(cube.lbda,'d')
    
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
    
    ## print_msg("Rejection of slices with bad values...", opts.verbosity, 0)
    ## max_spec = cube.data.max(axis=1)    # Max per slice
    ## med = N.median(F.median_filter(max_spec, 5) - max_spec)
    ## tmp2 = (max_spec - med)**2
    ## indice = tmp2 < 25*N.median(tmp2)
    ## if (-indice).any():                 # Some discarded slices
    ##     print_msg("   %d slices discarded: %s" % \
    ##               (len(cube.lbda[-indice]), cube.lbda[-indice]),
    ##               opts.verbosity, 0)
    ##     cube.data = cube.data[indice]
    ##     cube.lbda = cube.lbda[indice]
    ##     cube.var = cube.var[indice]
    ##     cube.nslice = len(cube.lbda)

    # Computing guess parameters from slice by slice 2D fit =================
    
    print_msg("Slice-by-slice 2D-fitting...", opts.verbosity, 0)
    
    xc_vec,yc_vec,sigc_vec,q_vec,eps_vec,sigk_vec,qk_vec,theta_vec,int_vec,sky_vec = get_start(cube,opts.sky_deg,0)

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
    #    filtered vectors. Finally, the parameters theta and alpha are
    #    determined from the xc, yc vectors.
    ind = ( N.absolute(xc_vec)<7 ) & ( N.absolute(yc_vec)<7 )
    if not ind.all():                   # Some centroids outside FoV
        print "%d/%d centroid positions discarded from ADR initial guess" % \
              (len(xc_vec[-ind]),nslice)
        if (len(xc_vec[ind])<=1):
            raise ValueError('Not enough points to determine ADR initial guess')
    
    xc_vec2 = xc_vec[ind]
    yc_vec2 = yc_vec[ind]
    ADR_coef = 206265*(atmosphericIndex(cube.lbda) -
                       atmosphericIndex(lbda_ref))

    P = pySNIFS.fit_poly(yc_vec2, 3, 1, xc_vec2)
    theta = N.arctan(P[1])
    x0 = xc_vec2[N.argmin(N.abs(lbda_ref - cube.lbda[ind]))]
    y0 = S.poly1d(P)(x0)
    
##     alpha_x_vec = ((xc_vec2-x0)/(N.cos(theta)*ADR_coef[ind]))[ADR_coef[ind]!=0]
##     alpha_y_vec = ((yc_vec2-y0)/(N.sin(theta)*ADR_coef[ind]))[ADR_coef[ind]!=0]
    
##     if theta == 0:
##         alpha = N.median(alpha_x_vec)
##     elif theta == N.pi/2.:
##         alpha = N.median(alpha_y_vec)
##     else:
##         alpha_x = N.median(alpha_x_vec)
##         alpha_y = N.median(alpha_y_vec)
##         alpha = N.mean([alpha_x, alpha_y])

    airmass = dict(cube.e3d_data_header)['AIRMASS']
    
    alpha = N.tan(N.arccos(1./airmass))
    
    # 2) Other parameters:
    sigc   = N.median(sigc_vec*(cube.lbda/lbda_ref)**0.2) 
    q      = N.median(q_vec)
    qk     = N.median(qk_vec)
    eps    = N.median(eps_vec)
    sigk   = N.median(sigk_vec)
    theta_k= N.median(theta_vec)

    # Filling in the guess parameter arrays (px) and bounds arrays (bx)
    p1 = [None]*(11+cube.nslice)
    b1 = [None]*(11+cube.nslice)
    p1[0:11] = [alpha, theta, x0, y0, sigc, -0.2, q, eps, sigk, qk, theta_k]
    b1[0:11] = [[0, 2*alpha],           # alpha
                [-N.pi, N.pi],          # theta
                [None, None],           # x0
                [None, None],           # y0
                [0.01, None],           # sigc 
                [-0.3, 0],              # alpha (chrom. dependance of sigc)
                [1.84, 1.84],           # Fixed q
                [0.42, 0.42],           # Fixed epsilon
                [0.01, None],           # sigk 
                [1., None],             # qk
                [0., N.pi]]             # theta_k
    p1[11:11+cube.nslice] = int_vec.tolist()
    b1[11:11+cube.nslice] = [[0, None]] * cube.nslice

    p2 = N.ravel(N.transpose(sky_vec.tolist()))
    b2 = ([[0.,None]]+[[None,None]]*(npar_poly-1))*cube.nslice
    
    print_msg("  Initial guess: %s" % p1[:11], opts.verbosity, 2)

    # Instanciating the model class
    data_model = pySNIFS_fit.model(data=cube,
                                   func=['SNIFS_psf_3D;0.43, %f' % lbda_ref,
                                         'poly2D;%d'%opts.sky_deg],
                                   param=[p1,p2], bounds=[b1,b2])
    guesspar = data_model.flatparam
    
    # The fit is launched twice. This is a dirty trick to avoid it to get
    # quickly stuck on a bad solution...
    if opts.verbosity >= 3:
        data_model.fit(maxfun=200, save=True, msge=1) 
        data_model.fit(msge=1)
    else:
        data_model.fit(maxfun=200, save=True) 
        data_model.fit()
        
    # Storing result and guess parameters
    fitpar = data_model.fitpar

    print_msg("  Fit result: %s" % fitpar[:11], opts.verbosity, 2)
    print_msg("  Seeing estimate: %.2f arcsec FWHM" % (fitpar[4]*2.355),
              opts.verbosity, 0)

    # Computing final spectra for object and background =====================
    
    print_msg("Extracting the spectrum...", opts.verbosity, 0)
    
    full_cube = pySNIFS.SNIFS_cube(opts.input)
    lbda,spec,var = comp_spec(full_cube, fitpar[0:11], intpar=[0.43, lbda_ref], poly_deg=opts.sky_deg)
    npar_poly = int((opts.sky_deg+1)*(opts.sky_deg+2)/2) 

    # Save star spectrum ==============================

    step = inhdr.get('CDELTS')
    
    fit_param_hdr(inhdr,data_model.fitpar,lbda_ref,opts.input,opts.sky_deg)
    star_spec = pySNIFS.spectrum(data=spec[:,0],start=lbda[0],step=step)
    star_spec.WR_fits_file(opts.out,header_list=inhdr.items())
    star_var = pySNIFS.spectrum(data=var[:,0],start=lbda[0],step=step)
    star_var.WR_fits_file('var_'+opts.out,header_list=inhdr.items())
    
    # Save sky parameters spectra ==============================

    inhdr = pyfits.getheader(opts.input, 1) # 1st extension
    fit_param_hdr(inhdr,data_model.fitpar,lbda_ref,opts.input,opts.sky_deg)
    sky_spec = pySNIFS.spec_list([pySNIFS.spectrum(data=s,start=lbda[0],step=step) for s in N.transpose(spec)[1:]])
    sky_spec.WR_fits_file(opts.sky,header_list=inhdr.items())
    sky_var = pySNIFS.spec_list([pySNIFS.spectrum(data=v,start=lbda[0],step=step) for v in N.transpose(var)[1:]])
    sky_var.WR_fits_file('var_'+opts.sky,header_list=inhdr.items())

    print npar_poly
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
        plot5 = os.path.extsep.join((basename+"_fit4", opts.graph))
        plot6 = os.path.extsep.join((basename+"_fit5", opts.graph))
        plot7 = os.path.extsep.join((basename+"_fit6", opts.graph))
        plot8 = os.path.extsep.join((basename+"_fit7", opts.graph))
        
        # Plot of the star and sky spectra ------------------------------

        print_msg("Producing spectra plot %s..." % plot1, opts.verbosity, 1)
        
        fig1 = pylab.figure()
        axS = fig1.add_subplot(2, 1, 1)
        axB = fig1.add_subplot(2, 1, 2)
        #axN = fig1.add_subplot(3, 1, 3)
        
        axS.set_title("Star spectrum [%s]" % obj)
        axS.set_xticklabels([])
        spl = I.UnivariateSpline(star_spec.x,star_spec.data,w=1/N.sqrt(star_var.data),s=star_spec.len/1.5)
        x = star_spec.x.tolist()+star_spec.x[-1::-1].tolist()
        y = (spl(star_spec.x)-2*N.sqrt(star_var.data)).tolist()+(spl(star_spec.x)+2*N.sqrt(star_var.data))[-1::-1].tolist()
        axS.fill(x,y,facecolor='k',alpha=0.2)
        star_spec.overplot(ax=axS,color='b')
        axS.legend(['signal','+- 2 sigma'],loc='best')
        
        axB.set_title("Mean background spectrum (per spx)")
##         axB.set_xticklabels([])
        bkg_spec.data = bkg_spec.data / cube.nlens
        bkg_spec.var = bkg_spec.var / (cube.nlens)**2
        spl = I.UnivariateSpline(bkg_spec.x,bkg_spec.data,w=1/N.sqrt(bkg_spec.var),s=bkg_spec.len/1.5)
        x = bkg_spec.x.tolist()+bkg_spec.x[-1::-1].tolist()
        y = (spl(bkg_spec.x)-2*N.sqrt(bkg_spec.var)).tolist()+(spl(bkg_spec.x)+2*N.sqrt(bkg_spec.var))[-1::-1].tolist()
        axB.fill(x,y,facecolor='k',alpha=0.2)
        bkg_spec.overplot(ax=axB,color='b')
        axB.legend(['signal','+- 2 sigma'],loc='best')
        axB.set_xlabel("Wavelength [A]")
##         axN.set_title("Error spectra")
##         axN.semilogy()
##         axN.set_xlabel("Wavelength [A]")
##         star_var.data = N.sqrt(star_var.data)
##         bkg_spec.var = N.sqrt(bkg_spec.var) / cube.nlens
##         star_var.overplot(ax=axN,color='b')
##         bkg_spec.overplot(ax=axN,var=True,color='g')
        fig1.savefig(plot1)
        
        # Plot of the fit on each slice ------------------------------
        
        print_msg("Producing slice fit plot %s..." % plot2, opts.verbosity, 1)

        ncol = N.floor(N.sqrt(cube.nslice))
        nrow = N.ceil(cube.nslice/float(ncol))

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
        func1 = pySNIFS_fit.SNIFS_psf_3D(intpar=[data_model.func[0].pix,
                                                 data_model.func[0].lbda_ref],
                                         cube=cube_fit)
        func2 = pySNIFS_fit.poly2D(deg=data_model.func[1].deg, cube=cube_fit)
        cube_fit.data = func1.comp(fitpar[0:func1.npar]) + \
                        func2.comp(fitpar[func1.npar:func1.npar+func2.npar])

        fig3 = pylab.figure()
        fig3.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97, )
        for i in xrange(cube.nslice):   # Loop over slices
            ax = fig3.add_subplot(nrow, ncol, i+1)
            # YC - Why is there some NaN's in data slices?
            # (eg e3d_TC07_153_099_003_17_B.fits)
            sigSlice = N.nan_to_num(cube.slice2d(i, coord='p'))
            varSlice = N.nan_to_num(cube.slice2d(i, coord='p', var=True))
            modSlice = cube_fit.slice2d(i, coord='p')
            prof_I = sigSlice.sum(axis=0) # Sum along rows
            prof_J = sigSlice.sum(axis=1) # Sum along columns
            err_I = N.sqrt(varSlice.sum(axis=0))
            err_J = N.sqrt(varSlice.sum(axis=1))
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
               N.cos(fitpar[1]) + fitpar[2]
        yfit = fitpar[0] * data_model.func[0].ADR_coef[:,0] * \
               N.sin(fitpar[1]) + fitpar[3]
        xguess = guesspar[0] * data_model.func[0].ADR_coef[:,0] * \
                 N.cos(guesspar[1]) + guesspar[2]
        yguess = guesspar[0] * data_model.func[0].ADR_coef[:,0] * \
                 N.sin(guesspar[1]) + guesspar[3]

        fig4 = pylab.figure()
        ax4a = fig4.add_subplot(2, 2, 1)
        ax4a.plot(cube.lbda, xc_vec, 'b.', label="Fit 2D")
        ax4a.plot(cube.lbda, xguess, 'k--', label="Guess 3D")
        ax4a.plot(cube.lbda, xfit, 'b', label="Fit 3D")
        ax4a.set_xlabel("Wavelength [A]")
        ax4a.set_ylabel("X center [arcsec]")
        pylab.setp(ax4a.get_xticklabels()+ax4a.get_yticklabels(), fontsize=8)
        ax4a.legend(loc='best')

        ax4b = fig4.add_subplot(2, 2, 2)
        ax4b.plot(cube.lbda, yc_vec, 'b.')
        ax4b.plot(cube.lbda, yfit, 'b')
        ax4b.plot(cube.lbda, yguess, 'k--')
        ax4b.set_xlabel("Wavelength [A]")
        ax4b.set_ylabel("Y center [arcsec]")
        pylab.setp(ax4b.get_xticklabels()+ax4b.get_yticklabels(), fontsize=8)

        ax4c = fig4.add_subplot(2, 1, 2)
        ax4c.scatter(xc_vec, yc_vec,
                     c=cube.lbda[::-1], cmap=matplotlib.cm.Spectral)
        ax4c.plot(xguess, yguess, 'k--')
        ax4c.plot(xfit, yfit, 'b')
        ax4c.text(0.03, 0.85,
                  r'$\rm{Guess:}\hspace{0.5} x_{0}=%4.2f,\hspace{0.5} ' \
                  r'y_{0}=%4.2f,\hspace{0.5} \alpha=%5.2f,\hspace{0.5} ' \
                  r'\theta=%6.2f^\circ$' % \
                  (x0, y0, alpha, theta*180./N.pi),
                  transform=ax4c.transAxes)
        ax4c.text(0.03, 0.75,
                  r'$\rm{Fit:}\hspace{0.5} x_{0}=%4.2f,\hspace{0.5} ' \
                  r'y_{0}=%4.2f,\hspace{0.5} \alpha=%5.2f,\hspace{0.5} ' \
                  r'\theta=%6.2f^\circ$' % \
                  (fitpar[2], fitpar[3], fitpar[0], fitpar[1]*180./N.pi),
                  transform=ax4c.transAxes)
        ax4c.set_xlabel("X center [arcsec]")
        ax4c.set_ylabel("Y center [arcsec]")
        fig4.text(0.5, 0.93, "ADR plot [%s, airmass=%.2f]" % (obj, airmass), 
                  horizontalalignment='center', size='large')
        fig4.savefig(plot4)

        # Plot dispersion, adjusted core and theoretical dispersion ---------
        
        print_msg("Producing intrinsic dispersion plot %s..." % plot5,
                  opts.verbosity, 1)

        guess_disp = guesspar[4]*(cube.lbda/lbda_ref)**guesspar[5]
        core_disp = fitpar[4]*(cube.lbda/lbda_ref)**fitpar[5]
        th_disp = fitpar[4]*(cube.lbda/lbda_ref)**(-0.2)

        fig5 = pylab.figure()
        ax5b = fig5.add_subplot(2, 1, 1)
        ax5b.plot(cube.lbda, sigc_vec, 'bo', label="Fit 2D")
        ax5b.plot(cube.lbda, guess_disp, 'k--', label="Guess 3D")
        ax5b.plot(cube.lbda, core_disp, 'b', label="Fit 3D")
        ax5b.legend(loc='best')
        ax5b.set_ylabel(r'$\sigma_c$')
        ax5b.set_title("Intrinsic dispersion [%s, seeing=%.2f arcsec]" % \
                       (obj, fitpar[4]*2.355))

        ax5a = fig5.add_subplot(2, 1, 2) 
        ax5a.plot(cube.lbda, th_disp, 'k--', label="Sigma core (guess 3D)")
        ax5a.plot(cube.lbda, core_disp, 'b', label="Sigma core (fit 3D)")
        ax5a.plot((lbda_ref,),(fitpar[4],), 'ro', label='_nolegend_')
        ax5a.legend(loc='best')
        ax5a.set_ylabel(r'$\sigma_c$')
        ax5a.set_xlabel("Wavelength [A]")
        ax5a.text(0.03, 0.2,
                  r'$\rm{Guess:}\hspace{0.5} \sigma_{c}=%4.2f,' \
                  r'\hspace{0.5} \gamma=-0.2$' % sigc,
                  transform=ax5a.transAxes)
        ax5a.text(0.03, 0.1,
                  r'$\rm{Fit:}\hspace{0.5} \sigma_{c}=%4.2f,' \
                  r'\hspace{0.5} \gamma=%4.2f$' % (fitpar[4], fitpar[5]),
                  transform=ax5a.transAxes)
        fig5.savefig(plot5)

        # Plot of the other model parameters ------------------------------
        
        print_msg("Producing model parameter plot %s..." % plot6,
                  opts.verbosity, 1)

        fig6 = pylab.figure()
        ax6a = fig6.add_subplot(5, 1, 1)
        plot_non_chromatic_param(ax6a, q_vec, cube.lbda, q, fitpar[6], 'q')
        ax6a.set_xticklabels([])
        ax6a.set_title("Model parameters [%s]" % obj)
        ax6b = fig6.add_subplot(5, 1, 2)
        plot_non_chromatic_param(ax6b, eps_vec, cube.lbda, eps, fitpar[7],
                                 '\\epsilon')
        ax6b.set_xticklabels([])
        ax6c = fig6.add_subplot(5, 1, 3)
        plot_non_chromatic_param(ax6c, sigk_vec, cube.lbda, sigk, fitpar[8],
                                 '\\sigma_k')
        ax6c.set_xticklabels([])
        ax6d = fig6.add_subplot(5, 1, 4)
        plot_non_chromatic_param(ax6d, 1/qk_vec, cube.lbda, 1/qk, 1/fitpar[9], '1/q_k')
        ax6d.set_xticklabels([])
        ax6e = fig6.add_subplot(5, 1, 5)
        plot_non_chromatic_param(ax6e,theta_vec,cube.lbda,theta_k,fitpar[10],
                                 '\\theta_k')
        ax6e.set_xlabel('Wavelength [A]')
        fig6.savefig(plot6)

        # Plot of the radial profile --------------

        print_msg("Producing radial profile plot %s..." % plot7,
                  opts.verbosity, 1)

        fig7 = pylab.figure()
        fig7.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97, )
        for i in xrange(cube.nslice):   # Loop over slices
            ax = fig7.add_subplot(nrow, ncol, i+1)
            ax.plot(N.hypot(cube.x-xfit[i],cube.y-yfit[i]),
                    cube.data[i], 'b.')
            ax.plot(N.hypot(cube_fit.x-xfit[i],cube_fit.y-yfit[i]),
                    cube_fit.data[i], 'r,')
            ax.semilogy()
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.1,0.1, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='left', transform=ax.transAxes)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Radius [arcsec]", fontsize=8)
                ax.set_ylabel("Flux", fontsize=8)
        fig7.savefig(plot7)

        # Contour plot of each slice ------------------------------
        
        print_msg("Producing PSF contour plot %s..." % plot8,
                  opts.verbosity, 1)

        fig8 = pylab.figure()
        fig8.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97, )
        extent = (cube.x.min(),cube.x.max(),cube.y.min(),cube.y.max())
        for i in xrange(cube.nslice):   # Loop over meta-slices
            ax = fig8.add_subplot(ncol, nrow, i+1, aspect='equal')
            data = cube.slice2d(i, coord='p')
            fit = cube_fit.slice2d(i, coord='p')
            vmin,vmax = pylab.prctile(fit, (5.,95.)) # Percentiles
            lev = N.logspace(N.log10(vmin),N.log10(vmax),5)
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

