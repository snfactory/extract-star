#!/usr/bin/env python
######################################################################
## Filename:      extract_star.py
## Version:       $Revision$
## Description:   Standard star spectrum extraction
## Author:        $Author$
## $Id$
######################################################################

__author__ = "Emmanuel Pecontal"
__version__ = '$Revision$'

import os
import sys
import optparse

import pySNIFS
import pySNIFS_fit

import pyfits

import numpy as N
import scipy as S
from scipy.ndimage import filters as F

def print_msg(str, verbosity, limit=0):
    """Print message 'str' if verbosity level >= verbosity limit."""

    if verbosity >= limit:
        print str


def atmosphericIndex(lbda, P=608, T=2):

    """Compute atmospheric refractive index: lbda in angstrom, P
    in mbar, T in C, RH in %.

    NOTE: Cohen & Cromer 1988 (PASP, 100, 1582) give P = 456 mmHg = 608 mbar
    and T = 2C for Mauna Kea.  Further note that typical water abundances on
    Mauna Kea are close enough to zero not to significantly impact these
    calculations."""

    # Sea-level (P=760mm Hg, T=15C)
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
    ax.set_ylabel(r'$%s$' % (str_par), fontsize=15)
    ax.text(0.05, 0.7,
            r'$\rm{Guess:}\hspace{0.5} %s = %4.2f,\hspace{0.5} ' \
            r'$\rm{Fit:}\hspace{0.5} %s = %4.2f$' % \
            (str_par, guess_par, str_par, fitpar),
            transform=ax.transAxes, fontsize=10)
    #ax.legend(('2D PSF', '3D PSF Guess', '3D PSF Fit'))
    pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=8)

def comp_spec(cube_file, psf_param, intpar=[None, None]):

    cube = pySNIFS.SNIFS_cube(cube_file)
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
    weight = N.where(cube.var!=0, 1./cube.var, 0)

    # Fit on masked data
    psf = N.array(model.comp(param), dtype='d')
    alpha = N.sum(weight*psf**2, axis=1)
    beta = N.sum(weight*psf, axis=1)
    gamma = N.sum(psf*cube.data*weight, axis=1)
    delta = N.sum(weight, axis=1)
    epsilon = N.sum(cube.data*weight, axis=1)
    det = (beta**2 - alpha*delta)
    obj = (beta*epsilon - delta*gamma)/det
    sky = (beta*gamma - alpha*epsilon)/det
    hess = N.zeros((len(obj),2,2),'d')
    hess[:,0,0] = delta
    hess[:,0,1] = beta
    hess[:,1,0] = beta
    hess[:,1,1] = alpha
    cofact_hess = N.zeros((len(obj),2,2),'d')
    cofact_hess[:,0,0] = -delta
    cofact_hess[:,0,1] = beta
    cofact_hess[:,1,0] = beta
    cofact_hess[:,1,1] = -alpha
    
    cov = N.transpose(N.transpose(cofact_hess,(2,1,0))/det,(2,0,1))/2
    var_obj = cov[:,0,0]
    var_sky = cov[:,1,1]
    # This is were one could implement optimal extraction. - YC

    spec = N.zeros((5, cube.nslice), 'd')
    spec[0,:] = cube.lbda
    spec[1,:] = obj
    spec[2,:] = sky
    spec[3,:] = var_obj
    spec[4,:] = var_sky
  
    return spec

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
    if opts.plot:
        opts.graph = 'png'

    # Input datacube ==============================
    
    print_msg("Opening datacube %s" % opts.input, opts.verbosity, 0)
    inhdr = pyfits.getheader(opts.input, 1) # 1st extension
    channel = inhdr.get('CHANNEL', 'Unknown').upper()
    if channel.startswith('B'):
        slices=[10, 900, 20]
    elif channel.startswith('R'):
        slices=[10, 1500, 30]
    else:
        parser.error("Input datacube %s has no valid CHANNEL keyword (%s)" % \
                     (opts.input, channel))

    print_msg("  Channel: %s" % channel, opts.verbosity, 0)
    print_msg("  Extracting slices: %s" % slices, opts.verbosity, 0)

    cube = pySNIFS.SNIFS_cube(opts.input, slices=slices)

    # Normalisation of the signal and variance in order to avoid numerical
    # problems with too small numbers
    norm = cube.data.mean()
    cube.data /= norm
    cube.var /= norm**2
    
    # Rejection of bad points ==============================
    
    print_msg("Rejection of slices with bad values...", opts.verbosity, 0)
    max_spec = cube.data.max(axis=1)
    med = N.median(F.median_filter(max_spec, 5) - max_spec)
    tmp2 = (max_spec - med)**2
    indice = tmp2 < 25*N.median(tmp2)
    cube.data = cube.data[indice]
    cube.lbda = cube.lbda[indice]
    cube.var = cube.var[indice]
    cube.nslice = len(cube.lbda)

    # Computing guess parameters from slice by slice 2D fit =================
    
    print_msg("Slice-by-slice 2D-fitting...", opts.verbosity, 0)
    
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
    sky_vec  = N.zeros(cube.nslice, dtype='d')
    for i in xrange(cube.nslice):
        if opts.verbosity >= 1:
            sys.stdout.write('\rSlice %2d/%d' % (i+1, cube.nslice))
            sys.stdout.flush()
        #print_msg("  Slice %2d/%d" % (i+1, cube.nslice), opts.verbosity, 1)
        cube2 = pySNIFS.SNIFS_cube()
        cube2.nslice = 1
        cube2.nlens = cube.nlens
        cube2.data = cube.data[i, N.newaxis]
        cube2.x = cube.x
        cube2.y = cube.y
        cube2.lbda = N.array([cube.lbda[i]])
        cube2.var = cube.var[i, N.newaxis]

        # Guess parameters for the current slice
        sky = (cube2.data[0]+5*cube2.var[0]).min() # Background
        sl_int = F.median_filter(cube2.data[0], 3) # Centroid
        imax = sl_int.max()              # Intensity
        sl_int -= sky
        xc = N.average(cube2.x, weights=sl_int)
        yc = N.average(cube2.y, weights=sl_int)

        # Filling in the guess parameter arrays (px) and bounds arrays (bx)
        p1 = [0, 0, xc, yc, 0.3, -0.2, 2.2, 0.1, 0.2, 1., 0., imax] # SNIFS_psf_3D;0.42
        b1 = [None]*(11+cube2.nslice) # Empty list of length 11+cube2.nslice
        b1[0:11] = [[None, None],
                    [-N.pi, N.pi],
                    [None, None],
                    [None, None],
                    [0.01, None],
                    [-5., 0],
                    [1., None], 
                    [0, None],
                    [0.01, None],
                    [1., None],
                    [0., N.pi]]
        b1[11:11+cube2.nslice] = [[0, None]] * cube2.nslice

        p2 = [sky]                      # poly2D;0
        b2 = [[0., None]]
        
        print_msg("    Initial guess: %s" % [p1,p2], opts.verbosity, 2)

        # Instanciating of a model class
        lbda_ref = cube2.lbda[0]
        sl_model = pySNIFS_fit.model(data=cube2,
                                     func=['SNIFS_psf_3D;0.42, %f' % lbda_ref,
                                           'poly2D;0'],
                                     param=[p1,p2], bounds=[b1,b2])

        # Fit of the current slice
        if opts.verbosity >= 3:
            sl_model.fit(maxfun=200, msge=1)
        else:
            sl_model.fit(maxfun=200)

        # Storing the result of the current slice parameters
        xc_vec[i]   = sl_model.fitpar[2]
        yc_vec[i]   = sl_model.fitpar[3]
        sigc_vec[i] = sl_model.fitpar[4]
        q_vec[i]    = sl_model.fitpar[6]
        eps_vec[i]  = sl_model.fitpar[7]
        sigk_vec[i] = sl_model.fitpar[8]
        qk_vec[i]   = sl_model.fitpar[9]
        theta_vec[i]= sl_model.fitpar[10]
        int_vec[i]  = sl_model.fitpar[11]
        sky_vec[i]  = sl_model.fitpar[12]

        print_msg("\n    Fit result: %s" % \
                  sl_model.fitpar, opts.verbosity, 2)

    # 3D model fitting ==============================
        
    print_msg("\nDatacube 3D-fitting...", opts.verbosity, 0)
    
    # Computing the initial guess for the 3D fitting from the results of the
    # slice by slice 2D fit
    lbda_ref = cube.lbda.mean()
    # 1) Position parameters:
    #    the xc, yc vectors obtained from 2D fit are smoothed, then the
    #    position corresponding to the reference wavelength is read in the
    #    filtered vectors. Finally, the parameters theta and alpha are
    #    determined from the xc, yc vectors.
    xc_vec = F.median_filter(xc_vec, 5)
    yc_vec = F.median_filter(yc_vec, 5)
    ADR_coef = 206265*(atmosphericIndex(cube.lbda) -
                       atmosphericIndex(lbda_ref))

    P = pySNIFS.fit_poly(yc_vec, 3, 1, xc_vec)
    theta = N.arctan(P[1])
    x0 = xc_vec[N.argmin(N.abs(lbda_ref - cube.lbda))]
    y0 = S.poly1d(P)(x0)
    
    alpha_x_vec = ((xc_vec-x0)/(N.cos(theta)*ADR_coef))[ADR_coef!=0]
    alpha_y_vec = ((yc_vec-y0)/(N.sin(theta)*ADR_coef))[ADR_coef!=0]
    if theta == 0:
        alpha = N.median(alpha_x_vec)
    elif theta == N.pi/2.:
        alpha = N.median(alpha_y_vec)
    else:
        alpha_x = N.median(alpha_x_vec)
        alpha_y = N.median(alpha_y_vec)
        alpha = N.mean([alpha_x, alpha_y])

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
    b1[0:11] = [[None, None],
                [-N.pi, N.pi],
                [None, None],
                [None, None],
                [0.01, None],
                [-5., 0],
                [1., None], 
                [0, None],
                [0.01, None],
                [1., None],
                [0., N.pi]]
    p1[11:11+cube.nslice] = int_vec.tolist()
    b1[11:11+cube.nslice] = [[0, None]] * cube.nslice

    p2 = sky_vec.tolist()
    b2 = [[0., None]] * cube.nslice

    print_msg("  Initial guess: %s" % p1[:11], opts.verbosity, 2)

    # Instanciating the model class
    data_model = pySNIFS_fit.model(data=cube,
                                   func=['SNIFS_psf_3D;0.42, %f' % lbda_ref,
                                         'poly2D;0'],
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

    # Computing final spectra for object and background =====================
    
    print_msg("Extracting the spectrum...", opts.verbosity, 0)
    spec = comp_spec(opts.input, fitpar[0:11], intpar=[0.42, lbda_ref])

    # The 3D psf model is not normalized to 1 in integral. The result must be
    # renormalized by (1+eps)
    spec[1] *= 1 + data_model.fitpar[7]
    spec[3] *= (1 + data_model.fitpar[7])**2
    
    # Save star spectrum ==============================

##     hdu = pyfits.PrimaryHDU(spec[1], inhdr)
##     # Delete/add some keywords
##     for key in ('EXTNAME','CTYPES','CRVALS','CDELTS','CRPIXS'):
##         del hdu.header[key]
##     hdu.header.update('CRVAL1', spec[0][0], after='NAXIS1')
##     hdu.header.update('CDELT1', inhdr.get('CDELTS'), after='CRVAL1')
##     for i,par in enumerate(fitpar[0:11]):
##         hdu.header.update('ESPAR%02d' % i, par)
##     hdu.writeto(opts.out, clobber=True) # Overwrite without asking
    star_spec = pySNIFS.spectrum(data=spec[1],var=spec[3],start=spec[0][0],step=inhdr.get('CDELTS'))
    star_spec.WR_fits_file(opts.out,header_list=inhdr.items())
    
    # Save sky spectrum ==============================

##     hdu = pyfits.PrimaryHDU(spec[2], inhdr)
##     # Delete/add some keywords
##     for key in ('EXTNAME','CTYPES','CRVALS','CDELTS','CRPIXS'):
##         del hdu.header[key]
##     hdu.header.update('CRVAL1', spec[0][0], after='NAXIS1')
##     hdu.header.update('CDELT1', inhdr.get('CDELTS'), after='CRVAL1')
##     for i,par in enumerate(fitpar[0:11]):
##         hdu.header.update('ESPAR%02d' % i, par)
##     hdu.writeto(opts.sky, clobber=True) # Overwrite without asking
    sky_spec = pySNIFS.spectrum(data=spec[2],var=spec[4],start=spec[0][0],step=inhdr.get('CDELTS'))
    sky_spec.WR_fits_file(opts.sky,header_list=inhdr.items())

    # Create output graphics ==============================
    
    if opts.plot:

        import matplotlib
        matplotlib.use('Agg')
        import pylab

        basename = os.path.splitext(opts.out)[0]
        plot1 = os.path.extsep.join((basename+"_plt" , opts.graph))
        plot1bis = os.path.extsep.join((basename+"_var" , opts.graph))
        plot2 = os.path.extsep.join((basename+"_fit1", opts.graph))
        plot3 = os.path.extsep.join((basename+"_fit2", opts.graph))
        plot4 = os.path.extsep.join((basename+"_fit3", opts.graph))
        plot5 = os.path.extsep.join((basename+"_fit4", opts.graph))
        plot6 = os.path.extsep.join((basename+"_fit5", opts.graph))
        
        # Plot of the star and sky spectra ------------------------------
        
        print_msg("Producing plot %s..." % plot1, opts.verbosity, 1)
        
        pylab.figure()
        pylab.subplot(2, 1, 1)
        pylab.plot(spec[0], spec[1])
        pylab.title("Star spectrum")
        pylab.subplot(2, 1, 2)
        pylab.plot(spec[0], spec[2])
        pylab.title("Background spectrum")
        pylab.savefig(plot1)
        
        # Plot of the star and sky variance spectra ------------------------------
        
        print_msg("Producing plot %s..." % plot1bis, opts.verbosity, 1)
        
        pylab.figure()
        pylab.subplot(2, 1, 1)
        pylab.plot(spec[0], spec[3])
        pylab.title("Star variance spectrum")
        pylab.subplot(2, 1, 2)
        pylab.plot(spec[0], spec[4])
        pylab.title("Background variance spectrum")
        pylab.savefig(plot1bis)
        # Plot of the fit on each slice ------------------------------
        
        print_msg("Producing plot %s..." % plot2, opts.verbosity, 1)

        pylab.figure()
        ncol = N.floor(N.sqrt(cube.nslice))
        nrow = N.ceil(cube.nslice/float(ncol))
        for i in xrange(cube.nslice):                 
            pylab.subplot(nrow, ncol, i+1)
            data = data_model.data.data[i,:]
            fit = data_model.evalfit()[i,:]
            imin = min(data.min(), fit.min())
            pylab.plot(data-imin+1e-2)
            pylab.plot(fit-imin+1e-2)
            pylab.semilogy()
            pylab.xticks(fontsize=4)
            pylab.yticks(fontsize=4)    
        pylab.savefig(plot2)

        # Plot of the fit on rows and columns sum ----------------------------
        
        print_msg("Producing plot %s..." % plot3, opts.verbosity, 1)

        pylab.figure()
        # Creating a standard SNIFS cube with the fitted data
        cube_fit = pySNIFS.SNIFS_cube(lbda=cube.lbda)
        func1 = pySNIFS_fit.SNIFS_psf_3D(intpar=[data_model.func[0].pix,
                                                 data_model.func[0].lbda_ref],
                                         cube=cube_fit)
        func2 = pySNIFS_fit.poly2D(0, cube_fit)
        cube_fit.data = func1.comp(fitpar[0:func1.npar]) + \
                        func2.comp(fitpar[func1.npar:func1.npar+func2.npar])
        for i in xrange(cube.nslice):                 
            pylab.subplot(nrow, ncol, i+1)
            pylab.plot(N.sum(cube.slice2d(i, coord='p'), axis=0), 'bo',
                       markersize=3)
            pylab.plot(N.sum(cube_fit.slice2d(i, coord='p'), axis=0), 'b-')
            pylab.plot(N.sum(cube.slice2d(i, coord='p'), axis=1), 'r^',
                       markersize=3)
            pylab.plot(N.sum(cube_fit.slice2d(i, coord='p'), axis=1), 'r-')
            pylab.xticks(fontsize=4)
            pylab.yticks(fontsize=4)    
        pylab.savefig(plot3)
        
        # Plot of the star center of gravity and fitted center ---------------
        
        print_msg("Producing plot %s..." % plot4, opts.verbosity, 1)

        sky = N.array(fitpar[11+cube.nslice:])
        xfit = fitpar[0] * data_model.func[0].ADR_coef[:,0] * \
               N.cos(fitpar[1]) + fitpar[2]
        yfit = fitpar[0] * data_model.func[0].ADR_coef[:,0] * \
               N.sin(fitpar[1]) + fitpar[3]
        xguess = guesspar[0] * data_model.func[0].ADR_coef[:,0] * \
                 N.cos(guesspar[1]) + guesspar[2]
        yguess = guesspar[0] * data_model.func[0].ADR_coef[:,0] * \
                 N.sin(guesspar[1]) + guesspar[3]

        pylab.figure()
        pylab.subplot(2, 2, 1)
        pylab.plot(cube.lbda, xc_vec, 'b.', label="Fitted 2D")
        pylab.plot(cube.lbda, xguess, 'k--', label="Guess 3D")
        pylab.plot(cube.lbda, xfit, 'b', label="Fitted 3D")
        pylab.xlabel("Wavelength [A]", fontsize=8)
        pylab.ylabel("X center [arcsec]", fontsize=8)
        pylab.xticks(fontsize=8)
        pylab.yticks(fontsize=8)
        pylab.legend(loc='best')
        pylab.subplot(2, 2, 2)
        pylab.plot(cube.lbda, yc_vec, 'b.')
        pylab.plot(cube.lbda, yfit, 'b')
        pylab.plot(cube.lbda, yguess, 'k--')
        pylab.xlabel("Wavelength [A]", fontsize=8)
        pylab.ylabel("Y center [arcsec]", fontsize=8)
        pylab.xticks(fontsize=8)
        pylab.yticks(fontsize=8)
        ax = pylab.subplot(2, 1, 2)
        #pylab.plot(xc_vec, yc_vec, 'bo')
        pylab.scatter(xc_vec, yc_vec,
                      c=cube.lbda[::-1], cmap=matplotlib.cm.Spectral)
        pylab.plot(xguess, yguess, 'k--')
        pylab.plot(xfit, yfit, 'b')
        pylab.text(0.03, 0.89,
                   r'$\rm{Guess:}\hspace{0.5} x_{0}=%4.2f,\hspace{0.5} ' \
                   r'y_{0}=%4.2f,\hspace{0.5} \alpha=%5.2f,\hspace{0.5} ' \
                   r'\theta=%6.2f^\circ$' % \
                   (x0, y0, alpha, theta*180./N.pi),
                   transform=ax.transAxes)
        pylab.text(0.03, 0.79,
                   r'$\rm{Fit:}\hspace{0.5} x_{0}=%4.2f,\hspace{0.5} ' \
                   r'y_{0}=%4.2f,\hspace{0.5} \alpha=%5.2f,\hspace{0.5} ' \
                   r'\theta=%6.2f^\circ$' % \
                   (fitpar[2], fitpar[3], fitpar[0], fitpar[1]*180./N.pi),
                   transform=ax.transAxes)
        pylab.xlabel("X center [arcsec]", fontsize=8)
        pylab.ylabel("Y center [arcsec]", fontsize=8)
        pylab.xticks(fontsize=8)
        pylab.yticks(fontsize=8)
        pylab.savefig(plot4)

        # Plot dispersion, fitted core dispersion and theoretical dispersion --
        
        print_msg("Producing plot %s..." % plot5, opts.verbosity, 1)

        sky = N.array(fitpar[11+cube.nslice:])[:, N.newaxis]
        core_disp = fitpar[4]*(cube.lbda/lbda_ref)**fitpar[5]
        guess_disp = guesspar[4]*(cube.lbda/lbda_ref)**guesspar[5]
        th_disp = fitpar[4]*(cube.lbda/lbda_ref)**(-0.2)

        pylab.figure()
        ax = pylab.subplot(2, 1, 2) 
        pylab.plot(cube.lbda, core_disp, 'b', label="Sigma core (Model)")
        pylab.plot(cube.lbda, th_disp, 'b--', label="Sigma core (Theoretical)")
        pylab.legend(loc='best')
        pylab.xlabel("Wavelength [A]", fontsize=8)
        pylab.ylabel(r'$\sigma_c$', fontsize=15)
        pylab.xticks(fontsize=8)
        pylab.yticks(fontsize=8)
        pylab.subplot(2, 1, 1)
        pylab.plot(cube.lbda, sigc_vec, 'bo', label="2D model")
        pylab.plot(cube.lbda, guess_disp, 'k--', label="3D Model guess")
        pylab.plot(cube.lbda, core_disp, 'b', label="3D Model fit")
        pylab.legend(loc='best')
        pylab.xlabel("Wavelength [A]", fontsize=8)
        pylab.ylabel(r'$\sigma_c$', fontsize=15)
        pylab.xticks(fontsize=8)
        pylab.yticks(fontsize=8)
        pylab.text(0.05, 0.9,
                   r'$\rm{Guess:}\hspace{0.5} \sigma_{c}=%4.2f,' \
                   r'\hspace{0.5} \gamma=-0.2$' % sigc,
                   transform=ax.transAxes)
        pylab.text(0.05, 0.8,
                   r'$\rm{Fit:}\hspace{0.5} \sigma_{c}=%4.2f,' \
                   r'\hspace{0.5} \gamma=%4.2f$' % (fitpar[4], fitpar[5]),
                   transform=ax.transAxes)
        pylab.savefig(plot5)

        # Plot of the other model parameters ------------------------------
        
        print_msg("Producing plot %s..." % plot6, opts.verbosity, 1)

        pylab.figure()
        ax = pylab.subplot(5, 1, 1)
        plot_non_chromatic_param(ax, q_vec, cube.lbda, q, fitpar[6], 'q')
        ax.set_xticklabels([])
        ax = pylab.subplot(5, 1, 2)
        plot_non_chromatic_param(ax, eps_vec, cube.lbda, eps, fitpar[7],
                                 '\\epsilon')
        ax.set_xticklabels([])
        ax = pylab.subplot(5, 1, 3)
        plot_non_chromatic_param(ax, sigk_vec, cube.lbda, sigk, fitpar[8],
                                 '\\sigma_k')
        ax.set_xticklabels([])
        ax = pylab.subplot(5, 1, 4)
        plot_non_chromatic_param(ax, qk_vec, cube.lbda, qk, fitpar[9], 'q_k')
        ax.set_xticklabels([])
        ax = pylab.subplot(5, 1, 5)
        plot_non_chromatic_param(ax, theta_vec, cube.lbda, theta_k, fitpar[10],
                                 '\\theta_k')
        pylab.xlabel('Wavelength [A]', fontsize=15)
        pylab.savefig(plot6)
