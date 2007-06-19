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
    pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)

def fit_param_hdr(hdr,param,lbda_ref,cube):

    hdr.update('ES_VERS',__version__)
    hdr.update('ES_CUBE',cube)
    hdr.update('ES_ALPHA',param[0])
    hdr.update('ES_THETA',param[1])
    hdr.update('ES_X0',param[2])
    hdr.update('ES_Y0',param[3])
    hdr.update('ES_SIGC',param[4])
    hdr.update('ES_GAMMA',param[5])
    hdr.update('ES_Q',param[6])
    hdr.update('ES_EPS',param[7])
    hdr.update('ES_SIGK',param[8])
    hdr.update('ES_QK',param[9])
    hdr.update('ES_THETK',param[10])
    hdr.update('ES_LREF',lbda_ref)
    hdr.update('SEEING',param[4]*2.355) # Seeing estimate (FWHM in arcsec)
    
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

    if (det==0).any(): # Some pb in the fit, return 0 instead of NaN
        idx = (det==0)
        frac = 1.*len(alpha[idx])/len(alpha)
        print "%d/%d px [%.0f%%] cannot be extracted" % \
              (len(alpha[idx]), len(alpha), frac*100)
        # You can try to recover on non-null values (e.g. det[idx]=1;
        # alpha[idx]=0) but that's pretty useless because the PSF model is
        # wrong anyway. You can return N.zeros((5, cube.nslice), 'd'), but
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
    if not opts.input or not opts.out or not opts.sky:
        parser.error("At least one option is missing among " \
                     "'--in', '--out' and '--sky'.")
    if opts.plot:
        opts.graph = 'png'

    # Input datacube ==============================
    
    print_msg("Opening datacube %s" % opts.input, opts.verbosity, 0)
    inhdr = pyfits.getheader(opts.input, 1) # 1st extension
    obj = inhdr.get('OBJECT', 'Unknown')
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

    print_msg("  Object: %s" % obj, opts.verbosity, 0)
    print_msg("  Channel: %s" % channel, opts.verbosity, 0)
    print_msg("  Extracting slices: %s" % slices, opts.verbosity, 0)
    
    cube = pySNIFS.SNIFS_cube(opts.input, slices=slices)

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
        print_msg("", opts.verbosity, 2)
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
        sky = (cube2.data[0]+3*N.sqrt(cube2.var[0])).min() # Background
        sl_int = F.median_filter(cube2.data[0], 3) # Centroid
        imax = sl_int.max()              # Intensity
        sl_int -= sky
        xc = N.average(cube2.x, weights=sl_int)
        yc = N.average(cube2.y, weights=sl_int)

        # Filling in the guess parameter arrays (px) and bounds arrays (bx)
        p1 = [0, 0, xc, yc, 0.3, -0.2, 1.84, 0.42, 0.2, 1., 0., imax] # SNIFS_psf_3D;0.43
        b1 = [None]*(11+cube2.nslice) # Empty list of length 11+cube2.nslice
        b1[0:11] = [[None, None],
                    [-N.pi, N.pi],
                    [None, None],
                    [None, None],
                    [0.01, None],
                    [-0.3, 0],
                    [1.84, 1.84],       # Fixed q
                    [0.42, 0.42],       # Fixed epsilon
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
                                     func=['SNIFS_psf_3D;0.43, %f' % lbda_ref,
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

        print_msg("    Fit result: %s" % \
                  sl_model.fitpar, opts.verbosity, 2)
    print_msg("", opts.verbosity, 1)

    # 3D model fitting ==============================
        
    print_msg("Datacube 3D-fitting...", opts.verbosity, 0)
    
    # Computing the initial guess for the 3D fitting from the results of the
    # slice by slice 2D fit
    lbda_ref = cube.lbda.mean()
    # 1) Position parameters:
    #    the xc, yc vectors obtained from 2D fit are smoothed, then the
    #    position corresponding to the reference wavelength is read in the
    #    filtered vectors. Finally, the parameters theta and alpha are
    #    determined from the xc, yc vectors.
    indx = set(N.where(abs(xc_vec)<3)[0])
    indy = set(N.where(abs(yc_vec)<3)[0])
    ind = list(indx.intersection(indy))
    if (len(ind)<=1):
        raise ValueError('Not enough points to fit the ADR initial guess')
    
    xc_vec2 = xc_vec[ind]
    yc_vec2 = yc_vec[ind]
    ADR_coef = 206265*(atmosphericIndex(cube.lbda) -
                       atmosphericIndex(lbda_ref))

    P = pySNIFS.fit_poly(yc_vec2, 3, 1, xc_vec2)
    theta = N.arctan(P[1])
    x0 = xc_vec2[N.argmin(N.abs(lbda_ref - cube.lbda))]
    y0 = S.poly1d(P)(x0)
    
    alpha_x_vec = ((xc_vec2-x0)/(N.cos(theta)*ADR_coef[ind]))[ADR_coef[ind]!=0]
    alpha_y_vec = ((yc_vec2-y0)/(N.sin(theta)*ADR_coef[ind]))[ADR_coef[ind]!=0]
    
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
                [-0.3, 0],
                [1.84, 1.84],           # Fixed q
                [0.42, 0.42],           # Fixed epsilon
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
                                   func=['SNIFS_psf_3D;0.43, %f' % lbda_ref,
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
    spec = comp_spec(opts.input, fitpar[0:11], intpar=[0.43, lbda_ref])

    # The 3D psf model is not normalized to 1 in integral. The result must be
    # renormalized by (1+eps)
    spec[1] *= 1 + data_model.fitpar[7]
    spec[3] *= (1 + data_model.fitpar[7])**2
    
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
        plot1bis = os.path.extsep.join((basename+"_var" , opts.graph))
        plot2 = os.path.extsep.join((basename+"_fit1", opts.graph))
        plot3 = os.path.extsep.join((basename+"_fit2", opts.graph))
        plot4 = os.path.extsep.join((basename+"_fit3", opts.graph))
        plot5 = os.path.extsep.join((basename+"_fit4", opts.graph))
        plot6 = os.path.extsep.join((basename+"_fit5", opts.graph))
        plot7 = os.path.extsep.join((basename+"_fit6", opts.graph))
        plot8 = os.path.extsep.join((basename+"_fit7", opts.graph))
        
        # Plot of the star and sky spectra ------------------------------

        print_msg("Producing plot %s..." % plot1, opts.verbosity, 1)
        
        fig1 = pylab.figure()
        axS = fig1.add_subplot(2, 1, 1)
        axB = fig1.add_subplot(2, 1, 2)
        axS.plot(spec[0], spec[1])
        axS.set_title("Star spectrum [%s]" % obj)
        axS.set_xlim(spec[0][0],spec[0][-1])
        axB.plot(spec[0], spec[2])
        axB.set_xlim(spec[0][0],spec[0][-1])
        axB.set_xlabel("Wavelength [AA]")
        axB.set_title("Background spectrum (per spx)")
        fig1.savefig(plot1)
        
        # Plot of the star and sky error spectra ------------------------------
        
        print_msg("Producing plot %s..." % plot1bis, opts.verbosity, 1)
        
        fig1B = pylab.figure()
        axS = fig1B.add_subplot(2, 1, 1)
        axB = fig1B.add_subplot(2, 1, 2)

        axS.plot(spec[0], N.sqrt(spec[3]))
        axS.set_title("Star error spectrum")
        axS.semilogy()
        axS.set_xlim(spec[0][0],spec[0][-1])
        axB.plot(spec[0], N.sqrt(spec[4]))
        axB.set_xlabel("Wavelength [AA]")
        axB.set_title("Background error spectrum")
        axB.semilogy()
        axB.set_xlim(spec[0][0],spec[0][-1])

        fig1B.savefig(plot1bis)

        # Plot of the fit on each slice ------------------------------
        
        print_msg("Producing plot %s..." % plot2, opts.verbosity, 1)

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
        
        print_msg("Producing plot %s..." % plot3, opts.verbosity, 1)

        # Creating a standard SNIFS cube with the adjusted data
        cube_fit = pySNIFS.SNIFS_cube(lbda=cube.lbda)
        func1 = pySNIFS_fit.SNIFS_psf_3D(intpar=[data_model.func[0].pix,
                                                 data_model.func[0].lbda_ref],
                                         cube=cube_fit)
        func2 = pySNIFS_fit.poly2D(0, cube_fit)
        cube_fit.data = func1.comp(fitpar[0:func1.npar]) + \
                        func2.comp(fitpar[func1.npar:func1.npar+func2.npar])

        fig3 = pylab.figure()
        fig3.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97, )
        for i in xrange(cube.nslice):   # Loop over slices
            ax = fig3.add_subplot(nrow, ncol, i+1)
            prof_I = cube.slice2d(i, coord='p').sum(axis=0)
            prof_J = cube.slice2d(i, coord='p').sum(axis=1)
            err_I = N.sqrt(cube.slice2d(i, coord='p', var=True).sum(axis=0))
            err_J = N.sqrt(cube.slice2d(i, coord='p', var=True).sum(axis=1))
            mod_I = cube_fit.slice2d(i, coord='p').sum(axis=0)
            mod_J = cube_fit.slice2d(i, coord='p').sum(axis=1)
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
        
        print_msg("Producing plot %s..." % plot4, opts.verbosity, 1)

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

        ax4c = pylab.subplot(2, 1, 2)
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
        fig4.savefig(plot4)

        # Plot dispersion, adjusted core and theoretical dispersion ---------
        
        print_msg("Producing plot %s..." % plot5, opts.verbosity, 1)

        core_disp = fitpar[4]*(cube.lbda/lbda_ref)**fitpar[5]
        guess_disp = guesspar[4]*(cube.lbda/lbda_ref)**guesspar[5]
        th_disp = fitpar[4]*(cube.lbda/lbda_ref)**(-0.2)

        pylab.figure()
        ax = pylab.subplot(2, 1, 2) 
        pylab.plot(cube.lbda, core_disp, 'b', label="Sigma core (Model)")
        pylab.plot(cube.lbda, th_disp, 'b--', label="Sigma core (Theoretical)")
        pylab.legend(loc='best')
        pylab.xlabel("Wavelength [A]")
        pylab.ylabel(r'$\sigma_c$')
        pylab.subplot(2, 1, 1)
        pylab.plot(cube.lbda, sigc_vec, 'bo', label="Fit 2D")
        pylab.plot(cube.lbda, guess_disp, 'k--', label="Guess 3D")
        pylab.plot(cube.lbda, core_disp, 'b', label="Fit 3D")
        pylab.legend(loc='best')
        pylab.xlabel("Wavelength [A]")
        pylab.ylabel(r'$\sigma_c$')
        pylab.text(0.03, 0.2,
                   r'$\rm{Guess:}\hspace{0.5} \sigma_{c}=%4.2f,' \
                   r'\hspace{0.5} \gamma=-0.2$' % sigc,
                   transform=ax.transAxes)
        pylab.text(0.03, 0.1,
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
        pylab.xlabel('Wavelength [A]')
        pylab.savefig(plot6)

        # Plot of the radial profile --------------

        print_msg("Producing plot %s..." % plot7, opts.verbosity, 1)

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
        
        print_msg("Producing plot %s..." % plot8, opts.verbosity, 1)

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

