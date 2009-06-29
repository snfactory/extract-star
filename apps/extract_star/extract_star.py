#!/usr/bin/env python
# -*- coding: utf-8 -*-
##############################################################################
## Filename:      extract_star.py
## Version:       $Revision$
## Description:   Standard star spectrum extraction
## Author:        $Author$
## $Id$
##############################################################################

"""Primarily based on E. Pecontal's point source extractor (extract_star.py).
This version replaces the double gaussian PSF profile by an ad-hoc PSF profile
(correlated Gaussian + Moffat).

Todo:

* replace meta-slice centroid-based initial guess by gaussian-fit.
* one could use Aitchison (or "additive log-ratio") transform to
  enforce the normalization constraint on alphas (see
  http://thread.gmane.org/gmane.comp.python.scientific.user/16180/focus=16187
  or ) or Multinomial logit (see
  http://en.wikipedia.org/wiki/Multinomial_logit and
  http://thread.gmane.org/gmane.comp.python.scientific.user/20318/focus=20320)

Polynomial approximation
========================

The polynomial approximations for alpha and ellipticity are expressed
internally as function of lr := (2*lambda - (lmin+lmax))/(lmax-lmin), to
minimize correlations. But the coeffs stored in header keywords are for
polynoms of lr := lambda/lref - 1.
"""

__author__ = "C. Buton, Y. Copin, E. Pecontal"
__version__ = '$Id$'

import os
import optparse

import pyfits                           # getheader

import pySNIFS
import pySNIFS_fit
import libExtractStar as libES

import scipy as S
from scipy import linalg as L
from scipy.ndimage import filters as F

SpaxelSize = 0.43                       # Spaxel size in arcsec
LbdaRef = 5000.                         # Use constant ref. for easy comparison

# Definitions ================================================================

def print_msg(str, limit):
    """Print message 'str' if verbosity level (opts.verbosity) >= limit."""

    if opts.verbosity >= limit:
        print str


def extract_spec(cube, psf_fn, psf_ctes, psf_param, skyDeg=0,
                 method='psf', radius=5.):
    """Extract object and sky spectra from cube according to PSF (described by
    psf_fn, psf_ctes and psf_params) in presence of sky (polynomial degree
    skyDeg) using method ('psf' or 'aperture' or 'optimal'). For aperture
    related methods, radius gives aperture radius in arcsec.

    Returns Spec,Var where Spec and Var are (nslice,npar+1)."""

    if method not in ('psf','aperture','optimal'):
        raise ValueError("Extraction method '%s' unrecognized" % method)
    if skyDeg < -1: 
        raise ValueError("skyDeg=%d is invalid (should be >=-1)" % skyDeg)

    if (cube.var>1e20).any(): 
        print "WARNING: discarding infinite variances in extract_spec"
        cube.var[cube.var>1e20] = 0
    if (cube.var<0).any():              # There should be none anymore
        print "WARNING: discarding negative variances in extract_spec"
        cube.var[cube.var<0] = 0

    # The PSF parameters are only the shape parameters. We set the intensity
    # of each slice to 1.
    param = S.concatenate((psf_param,[1.]*cube.nslice))

    # Linear least-squares fit: I*PSF + sky [ + a*x + b*y + ...]

    spxSize = psf_ctes[0]               # Spaxel size [arcsec]
    cube.x  = cube.i - 7                # x in spaxel
    cube.y  = cube.j - 7                # y in spaxel
    model = psf_fn(psf_ctes, cube)
    psf = model.comp(param, normed=True) # nslice,nlens

    npar_sky = int((skyDeg+1)*(skyDeg+2)/2) # Nb param. in polynomial bkgnd
    Z = S.zeros((cube.nslice,cube.nlens,npar_sky+1),'d')
    Z[:,:,0] = psf                      # Intensity
    if npar_sky:                        # =0 when no background (skyDeg<=-1)
        Z[:,:,1] = 1                    # Constant background
        n = 2
        for d in xrange(1,skyDeg+1):
            for j in xrange(d+1):
                Z[:,:,n] = cube.x**(d-j) * cube.y**j # Bkgnd polynomials
                n += 1                  # Finally: n = npar_sky + 1

    # Weighting
    weight = S.where(cube.var>0, 1/S.sqrt(cube.var), 0) # nslice,nlens
    X = (Z.T * weight.T).T              # nslice,nlens,npar+1
    b = weight*cube.data                # nslice,nlens

    # The linear least-squares fit could be done directly using
    # Spec = S.array([ L.lstsq(xx,bb)[0] for xx,bb in zip(X,b) ])
    # but A is needed anyway to compute covariance matrix C=1/A.
    # Furthermore, linear resolution 
    # [ L.solve(aa,bb) for aa,bb in zip(A,B) ]
    # can be replace by faster (~x10) matrix product
    # [ S.dot(cc,bb) for cc,bb in zip(C,B) ]
    # since C=1/A is already available.
    # See Numerical Recipes (2nd ed.), sect.15.4
    
    A = S.array([S.dot(xx.T, xx) for xx in X]) # nslice,npar+1,npar+1
    B = S.array([S.dot(xx.T, bb) for xx,bb in zip(X,b)]) # nslice,npar+1
    try:
        C = S.array([L.inv(aa) for aa in A])  # nslice,npar+1,npar+1
    except L.LinAlgError:
        raise L.LinAlgError("Singular matrix during spectrum extraction")
    # Spec & Var = nslice x Star,Sky,[slope_x...]
    Spec = S.array([S.dot(cc,bb) for cc,bb in zip(C,B)]) # nslice,npar+1
    Var = S.array([S.diag(cc) for cc in C]) # nslice,npar+1

    # Now, what about negative sky?
    # One could also use an NNLS fit to force parameter non-negativity:
    # [ pySNIFS_fit.fnnls(aa,bb)[0] for aa,bb in zip(A,B) ]
    # *BUT* 1. it is incompatible w/ non-constant sky (since it will force all
    # sky coeffs to >0). This can therefore be done only if skyDeg=0 (it would
    # otherwise involve optimization with constraints on sky positivity).
    # 2. There is no easy way to estimate covariance matrix from NNLS
    # fit. Since an NNLS fit on a negative sky slice would probably always
    # lead to a null sky, an NNLS fit is then equivalent to a standard 'PSF'
    # fit without sky.

    if skyDeg==0:
        negSky = Spec[:,1]<0            # Test for presence of negative sky
        if negSky.any():
            print "WARNING: %d slices w/ sky<0 in extract_spec" % \
                  (len(negSky.nonzero()[0]))
            print_msg(str(cube.lbda[negSky]), 2)
            # For slices w/ sky<0, fit only PSF without background
            A = S.array([ S.dot(xx,xx) for xx in X[negSky,:,0] ])
            B = S.array([ S.dot(xx,bb)
                          for xx,bb in zip(X[negSky,:,0],b[negSky]) ])
            C = 1/A
            Spec[negSky,0] = C*B        # Linear fit without sky
            Spec[negSky,1] = 0          # Set sky to null
            Var[negSky,0] = C
            Var[negSky,1] = 0

    if method=='psf':
        return cube.lbda,Spec,Var       # Nothing else to be done

    # Reconstruct background and subtract it from cube
    bkgnd = S.zeros_like(cube.data)
    var_bkgnd = S.zeros_like(cube.var)
    if npar_sky:
        for d in xrange(1,npar_sky+1):      # Loop over sky components
            bkgnd += (Z[:,:,d].T * Spec[:,d]).T
            var_bkgnd += (Z[:,:,d].T**2 * Var[:,d]).T
    subData = cube.data - bkgnd         # Bkgnd subtraction (nslice,nlens)
    subVar = cube.var.copy()
    good = cube.var>0
    subVar[good] += var_bkgnd[good]     # Variance of bkgnd-sub. signal

    # Replace invalid data (var=0) by model PSF = Intensity*PSF
    if not good.all():
        print_msg("Replacing %d vx with modeled signal" % \
                  len((~good).nonzero()[0]), 1)
        subData[~good] = (Spec[:,0]*psf.T).T[~good]

    # Plain summation over aperture
    
    # For the moment, a spaxel is either 100% or 0% within the aperture (same
    # limitation as quick_extract). Potential development:
    # 1. compute spaxel fraction inside the aperture
    # 2. extrapolate missing flux if aperture is partially outside FoV
    #    from PSF fit

    # Aperture center [spx] (nslice)
    xc = psf_param[2] + \
         psf_param[0]*model.ADR_coeff[:,0]*S.cos(psf_param[1])
    yc = psf_param[3] + \
         psf_param[0]*model.ADR_coeff[:,0]*S.sin(psf_param[1])
    # Aperture radius in spaxels
    aperRad = radius / spxSize
    print_msg("Aperture radius: %.2f arcsec = %.2f spx" % (radius,aperRad), 1)

    # Radius [spx] (nslice,nlens)
    r = S.hypot((model.x.T - xc).T, (model.y.T - yc).T)
    # Circular aperture (nslice,nlens)
    # Use r<aperRad[:,S.newaxis] if radius is a (nslice,) vec.
    frac = (r < aperRad).astype('float')

    # Check if aperture hits the FoV edges
    hit = ((xc - aperRad) < -7.5) | ((xc + aperRad) > 7.5) | \
          ((yc - aperRad) < -7.5) | ((yc + aperRad) > 7.5)
    if hit.any():
        # Find the closest edge
        ld = (xc - aperRad + 7.5).min() # Dist. to left edge (<0 if outside)
        rd =-(xc + aperRad - 7.5).max() # Dist. to right edge
        bd = (yc - aperRad + 7.5).min() # Dist. to bottom edge
        td =-(yc + aperRad - 7.5).max() # Dist. to top edge
        cd = -min(ld,rd,bd,td)          # Should be positive
        ns = int(cd) + 1                # Additional spaxels
        print "WARNING: Aperture (r=%.2f spx) hits FoV edges by %.2f spx" % \
              (aperRad, cd)

        if method=='optimal':
            print "WARNING: Model extrapolation outside FoV " \
                  "not implemented for optimal summation."

    if hit.any() and method=='aperture':

        # Extrapolate signal from PSF model
        print_msg("Signal extrapolation outside FoV...", 1)

        # Extend usual range by ns spx on each side
        nw = 15 + 2*ns                  # New FoV size in spaxels
        mid = (7 + ns)                  # FoV center
        extRange = S.arange(nw) - mid
        extx,exty = S.meshgrid(extRange[::-1],extRange) # nw,nw
        extnlens = extx.size                 # = nlens' = nw**2
        print_msg("  Extend FoV by %d spx: nlens=%d -> %d" % \
                  (ns, model.nlens, extnlens), 1)

        # Compute PSF on extended range (nslice,extnlens)
        extModel = psf_fn(psf_ctes, cube, coords=(extx,exty)) # Extended model
        extPsf = extModel.comp(param, normed=True) # nslice,extnlens

        # Embed background-subtracted data in extended model PSF
        origData = subData.copy()
        origVar = subVar.copy()
        subData = (Spec[:,0]*extPsf.T).T   # Extended model, nslice,extnlens
        subVar = S.zeros((extModel.nslice,extModel.nlens))
        for i in xrange(model.nlens):
            # Embeb original spx i in extended model array by finding
            # corresponding index j in new array
            j, = ((extModel.x[0]==model.x[0,i]) & \
                  (extModel.y[0]==model.y[0,i])).nonzero()
            subData[:,j[0]] = origData[:,i]
            subVar[:,j[0]] = origVar[:,i]

        r = S.hypot((extModel.x.T - xc).T, (extModel.y.T - yc).T)
        frac = (r < aperRad).astype('float')

    if method == 'aperture':
        # Replace signal and variance estimates from plain summation
        Spec[:,0] = (frac * subData).sum(axis=1)
        Var[:,0] = (frac**2 * subVar).sum(axis=1)
        return cube.lbda,Spec,Var

    if method=='optimal':
        # Model signal = Intensity*PSF + bkgnd
        modsig = (Spec[:,0]*psf.T).T + bkgnd # nslice,nlens

        # One has to have a model of the variance. This can be estimated from
        # a simple 'photon noise + RoN' model on each slice: signal ~ alpha*N
        # (alpha = 1/flat-field coeff and N = photon counts) and variance ~ (N
        # + RoN**2) * alpha**2 = (signal/alpha + RoN**2) * alpha**2 =
        # alpha*signal + beta. This model disregards spatial component of
        # flat-field, which is supposed to be constant of FoV.

        # Model variance = alpha*Signal + beta
        coeffs = S.array([ libES.polyfit_clip(modsig[s], cube.var[s], 1, clip=5)
                           for s in xrange(cube.nslice) ])
        coeffs = F.median_filter(coeffs, (5,1)) # A bit of smoothing...
        modvar = S.array([ S.polyval(coeffs[s], modsig[s])
                           for s in xrange(cube.nslice) ]) # nslice,nlens

        # Optimal weighting
        norm = (frac * psf).sum(axis=1) # PSF norm, nslice
        npsf = (psf.T / norm).T         # nslice,nlens
        weight = frac * npsf / modvar   # Unormalized weights, nslice,nlens
        norm = (weight * npsf).sum(axis=1) # Weight norm, nslice
        weight = (weight.T / norm).T    # Normalized weights, nslice,nlens
        
        # Replace signal and variance estimates from optimal summation
        Spec[:,0] = (weight * subData).sum(axis=1)
        Var[:,0] = (weight**2 * subVar).sum(axis=1)
        return cube.lbda,Spec,Var


def fit_slices(cube, psf_fn, skyDeg=0, nsky=2):
    """Fit (meta)slices of (meta)cube using PSF psf_fn and a background of
    polynomial degree skyDeg."""

    if skyDeg < -1: 
        raise ValueError("skyDeg=%d is invalid (should be >=-1)" % skyDeg)
    
    npar_psf = 7                        # Number of parameters of the psf
    npar_sky = int((skyDeg+1)*(skyDeg+2)/2) # Nb. param. in polynomial bkgnd

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

    # PSF + Intensity + Bkgnd coeffs
    param_arr = S.zeros((cube.nslice,npar_psf+1+npar_sky), dtype='d')
    khi2_vec  = S.zeros(cube.nslice, dtype='d')
    error_mat = S.zeros((cube.nslice,npar_psf+1+npar_sky), dtype='d')
    
    if nsky>7:                          # Nb of edge spx used for sky estimate
        raise ValueError('The number of edge pixels should be less than 7')
    skySpx = (cube_sky.i < nsky) | (cube_sky.i >= 15-nsky) | \
             (cube_sky.j < nsky) | (cube_sky.j >= 15-nsky)
    
    print_msg("  Adjusted parameters: [delta=0],[theta=0],xc,yc,PA,ell,alpha,I,"
              "%d bkgndCoeffs" % (skyDeg and npar_sky or 0), 2)

    for i in xrange(cube.nslice):
        cube_star.lbda = S.array([cube.lbda[i]])
        cube_star.data = cube.data[i, S.newaxis]
        cube_star.var  = cube.var[i,  S.newaxis]
        cube_sky.data  = cube.data[i, S.newaxis].copy() # To be modified
        cube_sky.var   = cube.var[i,  S.newaxis].copy()

        # Sky estimate (from FoV edge spx)
        medstar = F.median_filter(cube_star.data[0], 3)
        skyLev = S.median(cube_sky.data.T[skySpx].squeeze())
        if skyDeg > 0:
            # Fit a 2D polynomial of degree skyDeg on the edge pixels of a
            # given cube slice.
            cube_sky.var.T[~skySpx] = 0 # Discard central spaxels
            model_sky = pySNIFS_fit.model(data=cube_sky,
                                          func=['poly2D;%d' % skyDeg],
                                          param=[[skyLev] + [0.]*(npar_sky-1)],
                                          bounds=[[[0,None]] +
                                                  [[None,None]]*(npar_sky-1)])
            model_sky.fit()
            cube_sky.data = model_sky.evalfit() # 1st background estimate
            medstar -= cube_sky.data[0] # Subtract structured background estim.
        elif skyDeg == 0:
            medstar -= skyLev           # Subtract sky level estimate

        # Guess parameters for the current slice
        imax = medstar.max()            # Intensity
        xc = S.average(cube_star.x, weights=medstar) # Centroid [spx]
        yc = S.average(cube_star.y, weights=medstar)
        xc = S.clip(xc, -7.5,7.5)       # Put initial guess ~ in FoV
        yc = S.clip(yc, -7.5,7.5)

        # Filling in the guess parameter arrays (px) and bounds arrays (bx)
        p1 = [0., 0., xc, yc, 0., 1., 2.4, imax] # psf parameters
        b1 = [[0, 0],                   # delta (unfitted)
              [0, 0],                   # theta (unfitted)
              [None, None],             # xc
              [None, None],             # yc
              [None, None],             # PA              
              [0, None],                # ellipticity > 0
              [0, None],                # alpha > 0
              [0, None]]                # Intensity > 0

        func = ['%s;%f,%f,%f,%f' % \
                (psf_fn.name, SpaxelSize,cube_star.lbda[0],0,0)]
        param = [p1]
        bounds = [b1]

        if skyDeg >= 0:
            if skyDeg:                  # Use estimate from prev. polynomial fit
                p2 = list(model_sky.fitpar) 
            else:                       # Guess: Background=constant (>0)
                p2 = [skyLev] 
            b2 = [[0,None]] + [[None,None]]*(npar_sky-1)
            func += ['poly2D;%d' % skyDeg]
            param += [p2]
            bounds += [b2]
        else:                           # No background
            p2 = [] 
        print_msg("  Initial guess [#%d/%d, %.0fA]: %s" % \
                  (i+1,cube.nslice,cube.lbda[i],p1+p2), 2)

        # Instanciating of a model class
        model_star = pySNIFS_fit.model(data=cube_star, func=func,
                                       param=param, bounds=bounds,
                                       myfunc={psf_fn.name:psf_fn})

        # Fit of the current slice
        model_star.fit(maxfun=400, msge=int(opts.verbosity >= 3))

        # Probably one should check model_star.status...

        # Restore true chi2 (not reduced one), ie.
        # chi2 = ((cube_star.data-model_star.evalfit())**2/cube_star.var).sum()
        model_star.khi2 *= model_star.dof

        # Error computation
        if model_star.fitpar[5]>0 and \
               model_star.fitpar[6]>0 and model_star.fitpar[7]>0:
            # Cannot use model_star.param_error to compute cov, since it does
            # not handle fixed parameters (lb=ub).
            hess = pySNIFS_fit.approx_deriv(model_star.objgrad,
                                            model_star.fitpar)
            cov = 2 * S.linalg.inv(hess[2:,2:]) # Discard 1st 2 lines (unfitted)
            diag = cov.diagonal()
            if (diag>0).all():
                errorpar = S.concatenate(([0.,0.], S.sqrt(diag)))
            else:                       # Some negative diagonal elements!
                print "WARNING: negative cov. diag. elements " \
                    "in metaslice %d" % (i+1)
                model_star.khi2 *= -1   # To be discarded
                errorpar = S.zeros(len(error_mat.T))
        else:
            # Set error to 0 if alpha, intensity or ellipticity is 0.
            if model_star.fitpar[5]==0:
                print "WARNING: ellipticity of metaslice %d is null" % (i+1)
            elif model_star.fitpar[6]==0:
                print "WARNING: alpha of metaslice %d is null" % (i+1)
            elif model_star.fitpar[7]==0:
                print "WARNING: intensity of metaslice %d is null" % (i+1)
            model_star.khi2 *= -1       # To be discarded
            errorpar = S.zeros(len(error_mat.T))

        # Storing the result of the current slice parameters
        param_arr[i] = model_star.fitpar
        khi2_vec[i]  = model_star.khi2
        error_mat[i] = errorpar
        print_msg("  Fit result [%d DoF=%d chi2=%f]: %s" % \
                  (model_star.status,model_star.dof,model_star.khi2,
                   model_star.fitpar), 2)

    return param_arr,khi2_vec,error_mat


def create_2D_log_file(filename,object,airmass,efftime,
                       cube,param_arr,khi2,error_mat):

    npar_sky = (opts.skyDeg+1)*(opts.skyDeg+2)/2
    
    delta,theta  = param_arr[:2]
    xc,yc        = param_arr[2:4]
    PA,ell,alpha = param_arr[4:7]
    intensity    = param_arr[-npar_sky-1]
    sky          = param_arr[-npar_sky:]

    logfile = open(filename,'w')
    logfile.write('# cube    : %s   \n' % os.path.basename(opts.input))
    logfile.write('# object  : %s   \n' % object)
    logfile.write('# airmass : %.2f \n' % airmass)
    logfile.write('# efftime : %.2f \n' % efftime)

    logfile.write('# lbda  ' + \
                  '  '.join('%8s +/- d%-8s' % (n,n)
                            for n in ['delta','theta','xc','yc','PA',
                                      'ell','alpha','I'] + \
                            ['sky%d' % d for d in xrange(npar_sky)] ) + \
                  '        chi2\n')
    fmt = '%6.0f  ' + '  '.join(["%10.4g"]*((8+npar_sky)*2+1)) + '\n'

    for n in xrange(cube.nslice):
        list2D  = [cube.lbda[n],
                   delta[n], error_mat[n][0],
                   theta[n], error_mat[n][1],
                   xc[n]   , error_mat[n][2],
                   yc[n]   , error_mat[n][3],
                   PA[n]   , error_mat[n][4],
                   ell[n]  , error_mat[n][5],
                   alpha[n], error_mat[n][6],
                   intensity[n], error_mat[n][-npar_sky-1]]
        if npar_sky:
            tmp = S.array((sky.T[n],error_mat[n][-npar_sky:]))
            list2D += tmp.T.flatten().tolist()
        list2D += [khi2[n]]
        logfile.write(fmt % tuple(list2D))
    logfile.close()

    
def create_3D_log_file(filename,object,airmass,efftime,
                       cube,cube_fit,fitpar,khi3D,errorpar,lmin,lmax):

    logfile = open(filename,'w')
    logfile.write('# cube    : %s   \n' % os.path.basename(opts.input))
    logfile.write('# object  : %s   \n' % object)
    logfile.write('# airmass : %.2f \n' % airmass)
    logfile.write('# efftime : %.2f \n' % efftime)

    # Global parameters
    logfile.write('# lmin  lmax' + \
                  '  '.join('%8s +/- d%-8s' % (n,n)
                            for n in ['delta','theta','xc','yc','PA'] + \
                            ['ell%d' % d for d in xrange(ellDeg+1)] +
                            ['alpha%d' % d for d in xrange(alphaDeg+1)]) + \
                  '        chi2\n')
    fmt = '%6.0f  %6.0f  ' + \
          '  '.join(["%10.4g"]*((5+(ellDeg+1)+(alphaDeg+1))*2+1)) + '\n'
    list3D = [lmin,
              lmax,
              fitpar[0],errorpar[0],
              fitpar[1],errorpar[1],
              fitpar[2],errorpar[2],
              fitpar[3],errorpar[3],
              fitpar[4],errorpar[4]]
    for i in xrange(ellDeg+1):
        list3D += [fitpar[5+i],errorpar[5+i]]
    for i in xrange(alphaDeg+1):
        list3D += [fitpar[6+ellDeg+i],errorpar[6+ellDeg+i]]
    list3D += [khi3D]
    logfile.write(fmt % tuple(list3D))  

    # Megaslice parameters
    npar_psf = 7 + ellDeg + alphaDeg
    npar_sky = (opts.skyDeg+1)*(opts.skyDeg+2)/2

    logfile.write('# lbda  ' + \
                  '  '.join('%8s +/- d%-8s' % (n,n)
                            for n in ['I'] + \
                            ['sky%d' % d for d in xrange(npar_sky)] ) + \
                  '        chi2\n')
    fmt = '%6.0f  ' + '  '.join(["%10.4g"]*((1+npar_sky)*2+1)) + '\n'
    for n in xrange(cube.nslice):
        list2D  = [cube.lbda[n],
                   fitpar[npar_psf+n], errorpar[npar_psf+n]]
        for i in xrange(npar_sky):
            list2D.extend([fitpar[npar_psf+nslice+n*npar_sky+i],
                           errorpar[npar_psf+nslice+n*npar_sky+i]])
        chi2 = S.nan_to_num((cube.slice2d(n, coord='p') - \
                             cube_fit.slice2d(n, coord='p'))**2 / \
                            cube.slice2d(n, coord='p', var=True))
        list2D += [chi2.sum()]          # Slice chi2
        logfile.write(fmt % tuple(list2D))
    logfile.close()


def build_sky_cube(cube, sky, sky_var, skyDeg):

    if skyDeg < 0:
        raise ValueError("Cannot build_sky_cube with skyDeg=%d < 0." % skyDeg)

    nslices  = len(sky)
    npar_sky = int((skyDeg+1)*(skyDeg+2)/2)
    poly     = pySNIFS_fit.poly2D(skyDeg,cube)
    cube2    = pySNIFS.zerolike(cube)
    cube2.x  = (cube2.x)**2
    cube2.y  = (cube2.y)**2
    poly2    = pySNIFS_fit.poly2D(skyDeg,cube2)
    param    = S.zeros((nslices,npar_sky),'d')
    vparam   = S.zeros((nslices,npar_sky),'d')
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


def fill_header(hdr, param, adr, lrange, opts, khi2, seeing, tflux, sflux):
    """Fill header hdr with fit-related keywords."""

    # Convert reference position from lref=(lmin+lmax)/2 to LbdaRef
    lmin,lmax = lrange
    lref = (lmin+lmax)/2                # ADR reference wavelength
    x0,y0 = adr.refract(param[2],param[3], LbdaRef, unit=SpaxelSize)
    print_msg("Reference position [%.0fA]: %.2f x %.2f spx" % \
              (LbdaRef,x0,y0), 1)

    # Convert polynomial coeffs from lr=(2*lambda + (lmin+lmax))/(lmax-lmin)
    # to lr~ = lambda/LbdaRef - 1 = a + b*lr
    a = (lmin+lmax) / (2*LbdaRef) - 1
    b = (lmax-lmin) / (2*LbdaRef)
    c_ell = libES.polyConvert(param[5:6+opts.ellDeg], trans=(a,b))
    c_alp = libES.polyConvert(param[6+opts.ellDeg : \
                                    7+opts.ellDeg+opts.alphaDeg],
                              trans=(a,b))
    
    hdr.update('ES_VERS', __version__)
    hdr.update('ES_CUBE', opts.input, 'Input cube')
    hdr.update('ES_LREF', LbdaRef,    'Lambda ref. [A]')
    hdr.update('ES_SDEG', opts.skyDeg,'Polynomial bkgnd degree')
    hdr.update('ES_KHI2', khi2,       'Chi2 of 3D fit')
    hdr.update('ES_AIRM', adr.get_airmass(), 'Effective airmass')
    hdr.update('ES_PARAN',adr.get_parangle(), 'Effective parangle [deg]')
    hdr.update('ES_XC',   x0,         'xc @lbdaRef [spx]')
    hdr.update('ES_YC',   y0,         'yc @lbdaRef [spx]')
    hdr.update('ES_XY',   param[4],   'XY coeff.')
    for i in xrange(opts.ellDeg + 1):
        hdr.update('ES_E%i' % i, c_ell[i], 'Y2 coeff. e%d' % i)
    for i in xrange(opts.alphaDeg + 1):
        hdr.update('ES_A%i' % i, c_alp[i], 'Alpha coeff. a%d' % i)
    hdr.update('ES_METH', opts.method, 'Extraction method')
    if method != 'psf':
        hdr.update('ES_APRAD', opts.radius, 'Aperture radius [arcsec or sigma]')
    hdr.update('ES_TFLUX',tflux,      'Sum of the spectrum flux')
    if opts.skyDeg >= 0:
        hdr.update('ES_SFLUX',sflux,  'Sum of the sky flux')
    hdr.update('SEEING', seeing, 'Seeing @lbdaRef [arcsec] (extract_star)')
    if opts.supernova:
        hdr.update('ES_SNMOD', opts.supernova, 'Supernova mode')
    if opts.psf3Dconstraints:
        for i,constraint in enumerate(opts.psf3Dconstraints):
            hdr.update('ES_BND%d' % (i+1), constraint, "Constraint on 3D-PSF")


def setPSF3Dconstraints(psfConstraints, params, bounds):
    """Decipher psf3Dconstraints=[constraint] option and set initial
    guess params and/or bounds bounds accordingly. Each constraint is
    a string 'n:val' (strict constraint) or 'n:val1,val2' (loose
    constraint), for n=0 (delta), 1 (theta), 2,3 (position), 4 (PA),
    5...6+ellDeg (ellipticity polynomial coefficients) and
    7+ellDeg...8+ellDeg+alphaDeg (alpha polynomial coefficients)."""

    for psfConstraint in psfConstraints:
        try:
            n,constraintStr = psfConstraint.split(':')
            n = int(n)
            vals = map(float, constraintStr.split(','))
            assert len(vals) in (1,2)
        except ValueError, AssertionError:
            print "WARNING: Cannot decipher constraint '%s', discarded" % \
                psfConstraint
            continue
        else:
            if len(vals)==1:  # Strict constraint: param = val
                val = vals[0]
                params[n] = val
                bounds[n] = [val,val]
                print "WARNING: Forcing PSF param[%d] to %f" % (n,val)
            else:               # Loose constraint: vmin <= param <= vmax
                vmin,vmax = sorted(vals)
                params[n] = min(max(params[n],vmin),vmax)
                bounds[n] = [vmin,vmax]
                print "WARNING: Constraining PSF param[%d] in %f,%f" % \
                    (n,vmin,vmax)


# ########## MAIN ##############################

if __name__ == "__main__":

    # Options ================================================================

    methods = ('psf','aperture','optimal')

    usage = "usage: [%prog] [options] -i inE3D.fits " \
            "[-o outSpec.fits -s outSky.fits]"

    parser = optparse.OptionParser(usage, version=__version__)

    parser.add_option("-i", "--in", type="string", dest="input",
                      help="Input datacube (euro3d format)")
    parser.add_option("-o", "--out", type="string",
                      help="Output star spectrum")
    parser.add_option("-s", "--sky", type="string",
                      help="Output sky spectrum")
    parser.add_option("-V", "--variance", action='store_true',
                      help="Store variance spectrum in extension")

    parser.add_option("-S", "--skyDeg", type="int", dest="skyDeg",
                      help="Sky polynomial background degree [%default]",
                      default=0)
    parser.add_option("-A", "--alphaDeg", type="int",
                      help="Alpha polynomial degree [%default]",
                      default=2)
    parser.add_option("-E", "--ellDeg", type="int",
                      help="Ellipticity polynomial degree [%default]",
                      default=0)    

    parser.add_option("-m", "--method", type="string",
                      help="Extraction method ['%default']",
                      default="psf")
    parser.add_option("-r", "--radius", type="float",
                      help="Aperture radius for non-PSF extraction " \
                          "(>0: in arcsec, <0: in seeing sigma) " \
                      "[%default]", default=-5.)

    parser.add_option("-p", "--plot", action='store_true',
                      help="Plot flag (= '-g pylab')")
    parser.add_option("-g", "--graph", type="string",
                      help="Graphic output format (eps,pdf,png,pylab)")
    parser.add_option("-v", "--verbosity", type="int",
                      help="Verbosity level (<0: quiet) [%default]",
                      default=0)
    parser.add_option("-f", "--file", type="string",
                      help="Save 2D adjustment results in file.")
    parser.add_option("-F", "--File", type="string",
                      help="Save 3D adjustment results in file.")

    parser.add_option("--supernova", action='store_true',
                      help="SN mode (no final 3D fit).")
    parser.add_option("--keepmodel", action='store_true',
                      help="Store meta-slice model in 3D-cube.")
    parser.add_option("--psf3Dconstraints", type='string', action='append',
                      help="Constraints on PSF parameters (n:val,[val]).")

    opts,args = parser.parse_args()
    if not opts.input:
        parser.error("No input datacube specified.")

    if opts.graph:
        opts.plot = True
    elif opts.plot:
        opts.graph = 'pylab'

    opts.method = opts.method.lower()
    if opts.method not in methods:
        parser.error("Unrecognized extraction method '%s' %s " % \
                     (opts.method,methods))

    if opts.skyDeg < 0:
        opts.skyDeg = -1
        if opts.sky:
            print "WARNING: cannot extract sky spectrum in no-sky mode."

    # Input datacube =========================================================

    print "Opening datacube %s" % opts.input

    # The pySNIFS e3d_data_header dictionary is not enough for later
    # updates in fill_hdr, which requires a *true* pyfits header.
    inhdr = pyfits.getheader(opts.input, 1) # 1st extension

    # Test if input cube is Euro3D or plain NAXIS=3.
    isE3D = (inhdr['NAXIS'] != 3)
    if isE3D:
        full_cube = pySNIFS.SNIFS_cube(e3d_file=opts.input)
    else:
        full_cube = pySNIFS.SNIFS_cube(fits3d_file=opts.input)
    step = full_cube.lstep
    print_msg("Cube %s [%s]: %d slices [%.2f-%.2f], %d spaxels" % \
              (os.path.basename(opts.input), isE3D and 'E3D' or '3D',
               full_cube.nslice,
               full_cube.lbda[0], full_cube.lbda[-1], full_cube.nlens), 1)

    obj = inhdr.get('OBJECT', 'Unknown')
    efftime = inhdr['EFFTIME']
    airmass = inhdr['AIRMASS']
    parangle = libES.read_parangle(inhdr)
    channel = inhdr['CHANNEL'][0].upper()
    pressure,temp = libES.read_PT(inhdr)

    ellDeg   = opts.ellDeg
    alphaDeg = opts.alphaDeg
    npar_psf = 7 + ellDeg + alphaDeg

    skyDeg   = opts.skyDeg
    npar_sky = int((skyDeg+1)*(skyDeg+2)/2)

    # Select the PSF (short or long)
    psfFn = (efftime > 12.) and \
            libES.Long_ExposurePSF or \
            libES.Short_ExposurePSF

    print "  Object: %s, Airmass: %.2f, Efftime: %.1fs [%s]" % \
          (obj, airmass, efftime, psfFn.name)

    # Meta-slice definition (min,max,step [px])
    if channel == 'B':
        slices=[10, 900, 65]
    elif channel == 'R':
        slices=[10, 1500, 130]
    else:
        parser.error("Input datacube %s has no valid CHANNEL keyword (%s)" % \
                     (opts.input, channel))
    print "  Channel: '%s', extracting slices: %s" % (channel,slices)

    if isE3D:
        cube = pySNIFS.SNIFS_cube(e3d_file=opts.input, slices=slices)
    else:
        cube = pySNIFS.SNIFS_cube(fits3d_file=opts.input, slices=slices)
    cube.x = cube.i - 7                 # From arcsec to spx
    cube.y = cube.j - 7

    print_msg("  Meta-slices before selection: %d " \
              "from %.2f to %.2f by %.2f A" % \
              (len(cube.lbda), cube.lbda[0], cube.lbda[-1], cube.lstep), 0)

    # Normalisation of the signal and variance in order to avoid numerical
    # problems with too small numbers
    norm = cube.data.mean()
    cube.data /= norm
    cube.var /= norm**2

    # Computing guess parameters from slice by slice 2D fit ==================

    print "Slice-by-slice 2D-fitting..."
    param_arr,khi2_vec,error_mat = fit_slices(cube, psfFn, skyDeg=skyDeg)
    print_msg("", 1)

    param_arr = param_arr.T             # (nparam,nslice)
    delta_vec,theta_vec = param_arr[:2]
    xc_vec,yc_vec       = param_arr[2:4]
    PA_vec,ell_vec,alpha_vec = param_arr[4:7]
    int_vec = param_arr[7]
    if skyDeg >= 0:
        sky_vec = param_arr[8:]

    # Save 2D adjusted parameter file ========================================
    
    if opts.file:
        print "Producing 2D adjusted parameter file [%s]..." % opts.file        
        create_2D_log_file(opts.file,obj,airmass,efftime,
                           cube,param_arr,khi2_vec,error_mat)

    # 3D model fitting =======================================================
    
    print "Datacube 3D-fitting..."

    # Computing the initial guess for the 3D fitting from the results of the
    # slice by slice 2D fit
    nslice = cube.nslice
    lmin,lmax = cube.lbda[0],cube.lbda[-1]
    lbda_rel = ( 2*cube.lbda - (lmin+lmax) ) / ( lmax-lmin )
    lref = (lmin + lmax)/2

    # 2) Reference position
    # Convert meta-slice centroids to position at ref. lbda, and clip around
    # median position
    adr = libES.ADR_model(pressure, temp, lref=lref,
                          airmass=airmass, parangle=parangle)
    delta0 = adr.delta                  # ADR power
    theta0 = adr.theta                  # ADR angle [rad]
    print_msg(str(adr), 1)
    xref,yref = adr.refract(xc_vec,yc_vec, cube.lbda, 
                            backward=True, unit=SpaxelSize)
    valid = khi2_vec > 0                # Discard unfitted slices
    xref0 = S.median(xref[valid])       # Robust to outliers
    yref0 = S.median(yref[valid])
    r = S.hypot(xref - xref0, yref - yref0)
    rmax = 5*S.median(r[valid])         # Robust to outliers
    good = valid & (r <= rmax)          # Valid fit and reasonable position
    bad = valid & (r > rmax)            # Valid fit but discarded position
    if (valid & bad).any():
        print "WARNING: %d metaslices discarded after ADR selection" % \
              (len(S.nonzero(valid & bad)))
    print_msg("%d/%d centroids found withing %.2f spx of (%.2f,%.2f)" % \
              (len(xref[good]),len(xref),rmax,xref0,yref0), 1)
    xc,yc = xref[good].mean(),yref[good].mean()
    # We could use a weighted average, but does not make much of a difference
    # dx,dy = error_mat[:,2],error_mat[:,3]
    # xc = S.average(xref[good], weights=1/dx[good]**2)
    # yc = S.average(yref[good], weights=1/dy[good]**2)

    if not good.all():                   # Invalid slices + discarded centroids
        print "%d/%d centroid positions discarded for initial guess" % \
              (len(xc_vec[~good]),nslice)
        if len(xc_vec[good]) <= max(alphaDeg+1,ellDeg+1):
            raise ValueError('Not enough points for initial guesses')
    print_msg("  Reference position guess [%.0fA]: %.2f x %.2f spx" % \
              (lref,xc,yc), 1)
    print_msg("  ADR guess: delta=%.2f, theta=%.1f deg" % \
              (delta0, theta0/S.pi*180), 1) # S.degrees() from python-2.5 only

    # 3) Other parameters
    PA       = S.median(PA_vec[good])
    polEll   = pySNIFS.fit_poly(ell_vec[good],3,ellDeg,lbda_rel[good])
    polAlpha = pySNIFS.fit_poly(alpha_vec[good],3,alphaDeg,lbda_rel[good])

    # Filling in the guess parameter arrays (px) and bounds arrays (bx)
    p1     = [None]*(npar_psf+nslice)
    p1[:5] = [delta0, theta0, xc, yc, PA]
    p1[5:6+ellDeg]        = polEll.coeffs[::-1]
    p1[6+ellDeg:npar_psf] = polAlpha.coeffs[::-1]
    p1[npar_psf:npar_psf+nslice] = int_vec.tolist()

    if opts.supernova:                  # Fix all parameters but intensities
        print "WARNING: supernova-mode, no 3D PSF-fit"
        # This mode completely discards 3D fit. In pratice, a 3D-fit is still
        # performed on intensities, just to be coherent w/ the remaining of
        # the code.
        b1 = [[delta0, delta0],         # delta 
              [theta0, theta0],         # theta 
              [xc, xc],                 # x0 
              [yc, yc],                 # y0
              [PA, PA]]                 # PA
        for coeff in p1[5:6+ellDeg]+p1[6+ellDeg:npar_psf]:
            b1 += [[coeff,coeff]]       # ell and alpha coeff.
    else:
        b1 = [[None, None],             # delta
              [None, None],             # theta 
              [None, None],             # x0 
              [None, None],             # y0
              [None, None]]             # PA
        b1 += [[0, None]] + [[None, None]]*ellDeg   # ell0 > 0 
        b1 += [[0, None]] + [[None, None]]*alphaDeg # a0 > 0
    b1 += [[0, None]]*nslice            # Intensities

    if opts.psf3Dconstraints:   # Read and set constraints from option
        setPSF3Dconstraints(opts.psf3Dconstraints, p1, b1)

    func = [ '%s;%f,%f,%f,%f' % \
             (psfFn.name,SpaxelSize,lref,alphaDeg,ellDeg) ] # PSF
    param = [p1]
    bounds = [b1]

    if skyDeg >= 0:
        p2 = S.ravel(sky_vec.T)
        b2 = ([[0,None]] + [[None,None]]*(npar_sky-1)) * nslice 
        func += ['poly2D;%d' % skyDeg]  # Add background
        param += [p2]
        bounds += [b2]

    print_msg("  Adjusted parameters: delta,theta,xc,yc,PA,"
              "%d ellCoeffs,%d alphaCoeffs,%d intens., %d bkgndCoeffs" % \
              (ellDeg+1,alphaDeg+1,nslice,
               skyDeg>=0 and (npar_sky*nslice) or 0), 2)
    print_msg("  Initial guess [PSF]: %s" % p1[:npar_psf], 2)
    print_msg("  Initial guess [Intensities]: %s" % \
              p1[npar_psf:npar_psf+nslice], 3)
    if skyDeg >= 0:
        print_msg("  Initial guess [Background]: %s" % p2, 3)

    # Instanciating the model class and perform the fit
    data_model = pySNIFS_fit.model(data=cube, func=func,
                                   param=param, bounds=bounds,
                                   myfunc={psfFn.name:psfFn})
    data_model.fit(maxfun=2000, save=True, msge=(opts.verbosity>=3))

    # Storing result and guess parameters
    fitpar = data_model.fitpar          # Adjusted parameters
    data_model.khi2 *= data_model.dof   # Restore real chi2
    khi2 = data_model.khi2              # Total chi2 of 3D-fit
    cov = data_model.param_error(fitpar) # Covariance matrix
    errorpar = S.sqrt(cov.diagonal())

    print_msg("  Fit result [%d]: DoF=%d, chi2=%f" % \
              (data_model.status, data_model.dof, khi2), 2)
    print_msg("  Fit result [PSF param]: %s" % fitpar[:npar_psf], 2)
    print_msg("  Fit result [Intensities]: %s" % \
              fitpar[npar_psf:npar_psf+nslice], 3)
    if skyDeg >= 0:
        print_msg("  Fit result [Background]: %s" % \
                  fitpar[npar_psf+nslice:], 3)

    print_msg("  Reference position fit [%.0fA]: %.2f x %.2f spx" % \
              (lref,fitpar[2],fitpar[3]), 1)
    adr.set_param(fitpar[0], fitpar[1]) # Update ADR params
    print_msg("  ADR fit: delta=%.2f, theta=%.1f deg" % \
              (adr.delta, adr.get_parangle()), 1)
    print "  Effective airmass: %.2f" % adr.get_airmass()

    # Compute seeing (FWHM in arcsec)
    seeing = data_model.func[0].FWHM(fitpar[:npar_psf], LbdaRef) * SpaxelSize
    print '  Seeing estimate [%.0fA]: %.2f" FWHM' % (LbdaRef,seeing)

    if not 0.<seeing<4. and not 1.<adr.get_airmass()<3.:
        raise ValueError("Unphysical seeing or airmass")

    # Test positivity of alpha and ellipticity. At some point, maybe it would
    # be necessary to force positivity in the fit (e.g. fmin_cobyla).
    fit_alpha = libES.polyEval(fitpar[6+ellDeg:npar_psf], lbda_rel)
    if fit_alpha.min() < 0:
        raise ValueError("Alpha is negative (%.2f) at %.0fA" % \
                         (fit_alpha.min(), cube.lbda[fit_alpha.argmin()]))
    fit_ell = libES.polyEval(fitpar[5:6+ellDeg], lbda_rel)
    if fit_ell.min() < 0:
        raise ValueError("Ellipticity is negative (%.2f) at %.0fA" % \
                         (fit_ell.min(), cube.lbda[fit_ell.argmin()]))

    # Computing final spectra for object and background ======================

    # Compute aperture radius
    if opts.method == 'psf':
        radius = None
        method = 'psf'
    else:
        if opts.radius < 0:     # Aperture radius [sigma]
            radius = -opts.radius * seeing/2.355 # [arcsec]
            method = '%s r=%.1f sigma=%.2f"' % \
                (opts.method, -opts.radius, radius)
        else:                   # Aperture radius [arcsec]
            radius = opts.radius # [arcsec]
            method = '%s r=%.2f"' % (opts.method, radius)
    print "Extracting the spectrum [method=%s]..." % method
    if skyDeg < 0:
        print "WARNING: no background adjusted"

    psfCtes = [SpaxelSize,lref,alphaDeg,ellDeg]
    lbda,spec,var = extract_spec(full_cube, psfFn, psfCtes,
                                 fitpar[:npar_psf], skyDeg=skyDeg,
                                 method=opts.method, radius=radius)

    if skyDeg >= 0:                     # Compute background
        spec[:,1:] /= SpaxelSize**2     # Per arcsec^2
        var[:,1:]  /= SpaxelSize**4

        sky_spec_list = pySNIFS.spec_list([ pySNIFS.spectrum(data=s,
                                                             start=lbda[0],
                                                             step=step)
                                            for s in spec[:,1:] ])
        sky_var_list = pySNIFS.spec_list([ pySNIFS.spectrum(data=v,
                                                            start=lbda[0],
                                                            step=step)
                                           for v in var[:,1:] ])
        bkg_cube,bkg_spec = build_sky_cube(full_cube, sky_spec_list.list,
                                           sky_var_list.list, skyDeg)
        bkg_spec.data /= full_cube.nlens
        bkg_spec.var  /= full_cube.nlens

    # Creating a standard SNIFS cube with the adjusted data
    # We cannot directly use data_model.evalfit() because 1. we want
    # to keep psf and bkg separated; 2. cube_fit will always have 225
    # spx, data_model.evalfit() might have less.  But in the end,
    # psf+bkg ~= data_model.evalfit()
    cube_fit = pySNIFS.SNIFS_cube(lbda=cube.lbda) # Always 225 spx
    cube_fit.x = cube_fit.i - 7     # x in spaxel 
    cube_fit.y = cube_fit.j - 7     # y in spaxel
    
    psf_model = psfFn(psfCtes, cube=cube_fit)
    #psf_model = data_model.func[0]
    psf = psf_model.comp(fitpar[:psf_model.npar])
    cube_fit.data = psf.copy()

    if skyDeg >= 0:
        bkg_model = pySNIFS_fit.poly2D(skyDeg, cube_fit)
        #bkg_model = data_model.func[1]
        bkg = bkg_model.comp(fitpar[psf_model.npar: \
                                    psf_model.npar+bkg_model.npar])
        cube_fit.data += bkg

    # Update header ==========================================================
    
    tflux = spec[:,0].sum()             # Total flux of extracted spectrum
    if skyDeg >= 0:
        sflux = bkg_spec.data.sum()     # Total flux of sky (per arcsec^2)
    else:
        sflux = 0                       # Not stored anyway
    
    fill_header(inhdr,fitpar[:npar_psf],adr,(lmin,lmax),
                opts,khi2,seeing,tflux,sflux)

    # Save star spectrum =====================================================

    if not opts.out:
        opts.out = 'spec_%s.fits' % (channel)
        print "Saving output source spectrum to '%s'" % opts.out

    if not opts.variance:
        star_spec = pySNIFS.spectrum(data=spec[:,0],start=lbda[0],step=step)
        star_spec.WR_fits_file(opts.out,header_list=inhdr.items())
        star_var = pySNIFS.spectrum(data=var[:,0],start=lbda[0],step=step)
        path,name = os.path.split(opts.out)
        outname = os.path.join(path,'var_'+name)
        star_var.WR_fits_file(outname,header_list=inhdr.items())
    else:                       # Store variance as extension to signal
        star_spec = pySNIFS.spectrum(data=spec[:,0], var=var[:,0],
                                     start=lbda[0],step=step)
        star_spec.WR_fits_file(opts.out, header_list=inhdr.items())

    # Save sky spectrum/spectra ==============================================

    if skyDeg >= 0:
        if not opts.sky:
            opts.sky = 'sky_%s.fits' % (channel)
            print "Saving output sky spectrum to '%s'" % opts.sky

        if not opts.variance:
            sky_spec = pySNIFS.spectrum(data=bkg_spec.data,
                                        start=lbda[0],step=step)
            sky_spec.WR_fits_file(opts.sky,header_list=inhdr.items())
            sky_var = pySNIFS.spectrum(data=bkg_spec.var,
                                       start=lbda[0],step=step)
            path,name = os.path.split(opts.sky)
            outname = os.path.join(path,'var_'+name)
            sky_var.WR_fits_file(outname,header_list=inhdr.items())
        else:
            sky_spec = pySNIFS.spectrum(data=bkg_spec.data, var=bkg_spec.var,
                                        start=lbda[0],step=step)
            sky_spec.WR_fits_file(opts.sky,header_list=inhdr.items())

    # Save 3D adjusted parameter file ========================================
    
    if opts.File:
        print "Producing 3D adjusted parameter file '%s'..." % opts.File
        create_3D_log_file(opts.File,obj,airmass,efftime,
                           cube,cube_fit,fitpar,khi2,errorpar,lmin,lmax)
    
    # Save adjusted PSF ==============================

    if opts.keepmodel:
        path,name = os.path.split(opts.out)
        outpsf = os.path.join(path,'psf_'+name)
        print "Saving adjusted meta-slice PSF in 3D-fits cube '%s'..." % outpsf
        cube_fit.WR_3d_fits(outpsf)
    
    # Create output graphics =================================================

    if opts.plot:
        print "Producing output figures [%s]..." % opts.graph

        import matplotlib
        backends = {'png':'Agg','eps':'PS','pdf':'PDF','svg':'SVG'}
        if opts.graph.lower() in backends:
            matplotlib.use(backends[opts.graph.lower()])
            basename = os.path.splitext(opts.out)[0]
            plot1 = os.path.extsep.join((basename+"_plt" , opts.graph))
            plot2 = os.path.extsep.join((basename+"_fit1", opts.graph))
            plot3 = os.path.extsep.join((basename+"_fit2", opts.graph))
            plot4 = os.path.extsep.join((basename+"_fit3", opts.graph))
            plot6 = os.path.extsep.join((basename+"_fit4", opts.graph))
            plot7 = os.path.extsep.join((basename+"_fit5", opts.graph))
            plot8 = os.path.extsep.join((basename+"_fit6", opts.graph))
            plot5 = os.path.extsep.join((basename+"_fit7", opts.graph))
        else:
            opts.graph = 'pylab'
            plot1 = plot2 = plot3 = plot4 = plot6 = plot7 = plot8 = plot5 = ''
        import pylab

        # Plot of the star and sky spectra -----------------------------------

        print_msg("Producing spectra plot %s..." % plot1, 1)

        fig1 = pylab.figure()
        if skyDeg >= 0:
            axS = fig1.add_subplot(3, 1, 1)
            axB = fig1.add_subplot(3, 1, 2)
            axN = fig1.add_subplot(3, 1, 3)
        else:
            axS = fig1.add_subplot(2, 1, 1)
            axN = fig1.add_subplot(2, 1, 2)

        axS.plot(star_spec.x, star_spec.data, 'b')
        axS.text(0.95,0.8, os.path.basename(opts.input), fontsize='small',
                 horizontalalignment='right', transform=axS.transAxes)

        axN.plot(star_spec.x, star_spec.data/S.sqrt(var[:,0]), 'b')

        if skyDeg >= 0:
            axB.plot(bkg_spec.x, bkg_spec.data, 'g')
            axB.set(title=u"Background spectrum (per arcsec²)",
                    xlim=(bkg_spec.x[0],bkg_spec.x[-1]),
                    xticklabels=[])
            axN.plot(bkg_spec.x, bkg_spec.data/S.sqrt(bkg_spec.var), 'g')

        axS.set(title="Point-source spectrum [%s, %s]" % (obj,method),
                xlim=(star_spec.x[0],star_spec.x[-1]), xticklabels=[])
        axN.set(title="Signal/Noise",
                xlabel=u"Wavelength [Å]", 
                xlim=(star_spec.x[0],star_spec.x[-1]),
                yscale='log')

        # Plot of the fit on each slice --------------------------------------

        print_msg("Producing slice fit plot %s..." % plot2, 1)

        ncol = S.floor(S.sqrt(nslice))
        nrow = S.ceil(nslice/float(ncol))

        fig2 = pylab.figure()
        fig2.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.97)
        mod = data_model.evalfit()      # Total model (same nb of spx as cube)
        fmin = 0
        for i in xrange(nslice):        # Loop over meta-slices
            data = cube.data[i,:]
            fit = mod[i,:]
            #fmin = min(data.min(),fit.min()) - max(data.max(),fit.max())/1e4
            ax = fig2.add_subplot(nrow, ncol, i+1, 
                                  xlim=(0,len(data)),
                                  yscale='log')
            ax.plot(data - fmin, 'b-', lw=2)  # Signal
            ax.plot(fit - fmin, 'r-')         # Model
            ax.set_autoscale_on(False)
            if skyDeg >= 0:
                ax.plot(psf[i,:] - fmin, 'g-') # PSF alone
                ax.plot(bkg[i,:] - fmin, 'y-') # Background
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.1,0.8, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='left', transform=ax.transAxes)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Spaxel ID", fontsize=8)
                ax.set_ylabel("Flux + cte", fontsize=8)

        # Plot of the fit on rows and columns sum ----------------------------

        print_msg("Producing profile plot %s..." % plot3, 1)

        if not opts.verbosity:  # Plot fit on rows and columns sum

            fig3 = pylab.figure()
            fig3.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.97)
            for i in xrange(nslice):        # Loop over slices
                ax = fig3.add_subplot(nrow, ncol, i+1)
                sigSlice = cube.slice2d(i, coord='p', NAN=False)
                varSlice = cube.slice2d(i, coord='p', var=True, NAN=False)
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
                pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(),
                           fontsize=6)
                ax.text(0.1,0.8, "%.0f" % cube.lbda[i], fontsize=8,
                        horizontalalignment='left', transform=ax.transAxes)
                if ax.is_last_row() and ax.is_first_col():
                    ax.set_xlabel("I (blue) or J (red)", fontsize=8)
                    ax.set_ylabel("Flux", fontsize=8)

        else:                           # Plot correlation matrix

            corr = cov / S.outer(errorpar,errorpar) # Correlation matrix
            parnames = data_model.func[0].parnames # PSF param names
            if skyDeg >= 0:                 # Add background param names
                coeffnames = [ "00" ] + \
                             [ "%d%d" % (d-j,j)
                               for d in range(1,skyDeg+1) for j in range(d+1) ]
                parnames += [ "b%02d_%s" % (s+1,c)
                              for c in coeffnames for s in range(nslice) ]

            assert len(parnames)==corr.shape[0]
            # Remove some of the names for clarity
            parnames[npar_psf+1::2] = ['']*len(parnames[npar_psf+1::2])

            fig3 = pylab.figure(figsize=(6,6))
            ax3 = fig3.add_subplot(1,1,1, title="Correlation matrix for 3D-fit")
            im3 = ax3.imshow(S.absolute(corr),
                             vmin=max(1e-3,corr.min()), vmax=1,
                             aspect='equal', origin='upper',
                             norm=pylab.matplotlib.colors.LogNorm(),
                             )
            ax3.set_xticks(range(len(parnames)))
            ax3.set_xticklabels(parnames,
                                va='top', fontsize='x-small', rotation=90)
            ax3.set_yticks(range(len(parnames)))
            ax3.set_yticklabels(parnames,
                                ha='right', fontsize='x-small')

            cb3 = fig3.colorbar(im3, ax=ax3, orientation='horizontal')
            cb3.set_label("|Correlation|")

        # Plot of the star center of gravity and adjusted center -------------

        print_msg("Producing ADR plot %s..." % plot4, 1)

        xguess = xc + delta0*psf_model.ADR_coeff[:,0]*S.sin(theta0)
        yguess = yc - delta0*psf_model.ADR_coeff[:,0]*S.cos(theta0)
        xfit = fitpar[2] + fitpar[0]*psf_model.ADR_coeff[:,0]*S.sin(fitpar[1])
        yfit = fitpar[3] - fitpar[0]*psf_model.ADR_coeff[:,0]*S.cos(fitpar[1])

        fig4 = pylab.figure()
        ax4a = fig4.add_subplot(2, 2, 1,
                                xlabel=u"Wavelength [Å]",
                                ylabel="X center [spaxels]")
        ax4b = fig4.add_subplot(2, 2, 2,
                                xlabel=u"Wavelength [Å]",
                                ylabel="Y center [spaxels]")
        ax4c = fig4.add_subplot(2, 1, 2, 
                                aspect='equal', adjustable='datalim',
                                xlabel="X center [spaxels]",
                                ylabel="Y center [spaxels]")

        ax4a.errorbar(cube.lbda[good], xc_vec[good], yerr=error_mat[good,2],
                      fmt='b.',ecolor='b',label="Fit 2D")
        if bad.any():
            ax4a.plot(cube.lbda[bad],xc_vec[bad],'r.', label='_nolegend_')
        ax4a.plot(cube.lbda, xguess, 'k--', label="Guess 3D")
        ax4a.plot(cube.lbda, xfit, 'g', label="Fit 3D")
        pylab.setp(ax4a.get_xticklabels()+ax4a.get_yticklabels(), fontsize=8)
        leg = ax4a.legend(loc='best')
        pylab.setp(leg.get_texts(), fontsize='small')

        ax4b.errorbar(cube.lbda[good], yc_vec[good], yerr=error_mat[good,3],
                      fmt='b.',ecolor='b')
        if bad.any():
            ax4b.plot(cube.lbda[bad],yc_vec[bad],'r.')
        ax4b.plot(cube.lbda, yfit, 'g')
        ax4b.plot(cube.lbda, yguess, 'k--')
        pylab.setp(ax4b.get_xticklabels()+ax4b.get_yticklabels(), fontsize=8)

        ax4c.errorbar(xc_vec[valid],yc_vec[valid],
                      xerr=error_mat[valid,2],yerr=error_mat[valid,3],
                      fmt=None, ecolor='g')
        ax4c.scatter(xc_vec[good],yc_vec[good], faceted=True,
                     c=cube.lbda[good][::-1],
                     cmap=matplotlib.cm.Spectral, zorder=3)
        # Plot position selection process
        ax4c.plot(xref[good],yref[good],'b.') # Selected ref. positions
        ax4c.plot(xref[bad],yref[bad],'r.')   # Discarded ref. positions
        ax4c.plot((xref0,xc),(yref0,yc),'k-')
        ax4c.plot(xguess, yguess, 'k--') # Guess ADR
        ax4c.plot(xfit, yfit, 'g')       # Adjusted ADR
        ax4c.set_autoscale_on(False)
        ax4c.plot((xc,),(yc,),'k+')
        ax4c.add_patch(matplotlib.patches.Circle((xref0,yref0),radius=rmax,
                                                 ec='0.8',fc='None'))
        ax4c.add_patch(matplotlib.patches.Rectangle((-7.5,-7.5),15,15,
                                                 ec='0.8',lw=2,fc='None')) # FoV
        ax4c.text(0.03, 0.85,
                  u'Guess: x0,y0=%4.2f,%4.2f  airmass=%.2f parangle=%.1f°' % \
                  (xc, yc, airmass, parangle),
                  transform=ax4c.transAxes, fontsize='small')
        ax4c.text(0.03, 0.75,
                  u'Fit: x0,y0=%4.2f,%4.2f  airmass=%.2f parangle=%.1f°' % \
                  (fitpar[2], fitpar[3], adr.get_airmass(), adr.get_parangle()),
                  transform=ax4c.transAxes, fontsize='small')
        fig4.text(0.5, 0.93, "ADR plot [%s, airmass=%.2f]" % (obj, airmass),
                  horizontalalignment='center', size='large')

        # Plot of the other model parameters ---------------------------------

        print_msg("Producing model parameter plot %s..." % plot6, 1)

        guess_ell   = S.polyval(polEll.coeffs,   lbda_rel)
        guess_alpha = S.polyval(polAlpha.coeffs, lbda_rel)

        # err_ell and err_alpha are definitely wrong, and not only because
        # they do not include correlations between parameters!
        # d = cov.diagonal()
        # err_ell   = S.sqrt(libES.polyEval(d[5:6+ellDeg],lbda_rel))
        # err_alpha = S.sqrt(libES.polyEval(d[6+ellDeg:npar_psf],lbda_rel))
        err_PA = errorpar[4]

        def plot_conf_interval(ax, x, y, dy):
            ax.plot(x, y, 'g', label="Fit 3D")
            if dy is not None:
                ax.plot(x, y+dy, 'g:', label='_nolegend_')
                ax.plot(x, y-dy, 'g:', label='_nolegend_')

        fig6 = pylab.figure()
        ax6a = fig6.add_subplot(2, 1, 1,
                                title='Model parameters ' \
                                    '[%s, seeing %.2f" FWHM]' % \
                                    (obj,seeing),
                                xticklabels=[],
                                ylabel=r'$\alpha$ [spx]')
        ax6b = fig6.add_subplot(4, 1, 3,
                                xticklabels=[],
                                ylabel=u'y² coeff.')
        ax6c = fig6.add_subplot(4, 1, 4,
                                xlabel=u"Wavelength [Å]",
                                ylabel=u'xy coeff.')

        # WARNING: the so-called PA parameter is not the PA of the
        # adjusted ellipse, but half the x*y coefficient. Similarly,
        # ell is not the ellipticity, but the y**2 coefficient: x2 +
        # ell*y2 + 2*PA*x*y + ... = 0. One should use quadEllipse for
        # conversion, and use full covariance matrix to compute
        # associated errors.
        # Since 
        # rell2 = (x-x0)**2 + ell*(y-y0)**2 + 2*q*(x-x0)*(y-y0)
        #       = x2 + 2q*x*y + ell*y2 - 2x*(x0 + q*y0) - 2y*(ell*y0 + q*x0)
        #         + x02 +ell*y02 +2*q*x0*y0
        #       = a*x2 + 2b*x*y * c*y2 + 2d*x + 2f*y + g
        # with a=1, b=q, c=ell, d=-(x0 + q*y0), f=-(ell*y0 + q*x0), 
        # and g=x0**2 + ell*y0**2 + 2*q*x0*y0
        # so one should compute:
        # elldata = S.array([ quadEllipse(1, q, ell,
        #                                 -(x0 + q*y0), -(ell*y0 + q*x0),
        #                                 x0**2 + ell*y0**2 + 2*q*x0*y0 - 1) 
        #                     for x0,y0,ell,q in 
        #                     zip(xfit,yfit,fit_ell,[fitpar[4]]*nslice)])
        # and associated errors.

        ax6a.errorbar(cube.lbda[good], alpha_vec[good], error_mat[good,6],
                      fmt='b.', ecolor='b', label="Fit 2D")
        if bad.any():
            ax6a.plot(cube.lbda[bad],alpha_vec[bad],'r.', label="_nolegend_")
        ax6a.plot(cube.lbda, guess_alpha, 'k--', label="Guess 3D")
        #plot_conf_interval(ax6a, cube.lbda, fit_alpha, err_alpha)
        plot_conf_interval(ax6a, cube.lbda, fit_alpha, None)
        ax6a.text(0.03, 0.15,
                  'Guess: %s' % \
                  (', '.join([ 'a%d=%.2f' % (i,a) for i,a in
                              enumerate(polAlpha.coeffs[::-1]) ]) ),
                  transform=ax6a.transAxes, fontsize='small')
        ax6a.text(0.03, 0.05,
                  'Fit: %s' % \
                  (', '.join(['a%d=%.2f' % (i,a) for i,a in
                             enumerate(fitpar[6+ellDeg:npar_psf])])),
                  transform=ax6a.transAxes, fontsize='small')
        leg = ax6a.legend(loc='best')
        pylab.setp(leg.get_texts(), fontsize='small')

        ax6b.errorbar(cube.lbda[good], ell_vec[good], error_mat[good,5],
                      fmt='b.',ecolor='blue')
        if bad.any():
            ax6b.plot(cube.lbda[bad],ell_vec[bad],'r.')
        ax6b.plot(cube.lbda, guess_ell, 'k--')
        #plot_conf_interval(ax6b, cube.lbda, fit_ell, err_ell)
        plot_conf_interval(ax6b, cube.lbda, fit_ell, None)
        ax6b.text(0.03, 0.3,
                  'Guess: %s' % \
                  (', '.join([ 'e%d=%.2f' % (i,e)
                              for i,e in enumerate(polEll.coeffs[::-1]) ]) ),
                  transform=ax6b.transAxes, fontsize='small')
        ax6b.text(0.03, 0.1,
                  'Fit: %s' % \
                  (', '.join([ 'e%d=%.2f' % (i,e)
                              for i,e in enumerate(fitpar[5:6+ellDeg]) ])),
                  transform=ax6b.transAxes, fontsize='small')

        ax6c.errorbar(cube.lbda[good], PA_vec[good], error_mat[good,4], 
                      fmt='b.', ecolor='b')
        if bad.any():
            ax6c.plot(cube.lbda[bad],PA_vec[bad],'r.')
        ax6c.plot([cube.lbda[0],cube.lbda[-1]], [PA]*2, 'k--')
        plot_conf_interval(ax6c, S.asarray([cube.lbda[0],cube.lbda[-1]]),
                           S.ones(2)*fitpar[4], S.ones(2)*err_PA)
        ax6c.text(0.03, 0.1,
                  u'Guess: xy=%4.2f  Fit: xy=%4.2f' % (PA,fitpar[4]),
                  transform=ax6c.transAxes, fontsize='small')

        # Plot of the radial profile -----------------------------------------

        print_msg("Producing radial profile plot %s..." % plot7, 1)

        fig7 = pylab.figure()
        fig7.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.97)

        def ellRadius(x,y, x0,y0, ell, q):
            dx = x - x0
            dy = y - y0
            return S.sqrt(dx**2 + ell*dy**2 + 2*q*dx*dy)
        
        def radialbin(r,f, binsize=20, weighted=True):
            rbins = S.sort(r)[::binsize] # Bin limits, starting from min(r)
            ibins = S.digitize(r, rbins) # WARNING: ibins(min(r)) = 1
            ib = S.arange(len(rbins))+1 # Bin index
            rb = S.array([ r[ibins==b].mean() for b in ib ]) # Mean radius
            if weighted:
                fb = S.array([ S.average(f[ibins==b], weights=r[ibins==b]) 
                               for b in ib ]) # Mean radius-weighted data
            else:
                fb = S.array([ f[ibins==b].mean() for b in ib ]) # Mean data
            # Error on bin mean quantities
            #snb = S.sqrt([ len(r[ibins==b]) for b in ib ]) # sqrt(nb of points)
            #drb = S.array([ r[ibins==b].std()/n for b,n in zip(ib,snb) ])
            #dfb = S.array([ f[ibins==b].std()/n for b,n in zip(ib,snb) ])
            return rb,fb

        fmin = 0
        for i in xrange(nslice):        # Loop over slices
            ax = fig7.add_subplot(nrow, ncol, i+1, yscale='log')
            # Use adjusted elliptical radius instead of plain radius
            #r = S.hypot(cube.x-xfit[i],cube.y-yfit[i])
            #rfit = S.hypot(cube_fit.x-xfit[i],cube_fit.y-yfit[i])
            r = ellRadius(cube.x,cube.y, xfit[i],yfit[i], fit_ell[i], fitpar[4])
            rfit = ellRadius(cube_fit.x,cube_fit.y, xfit[i],yfit[i],
                             fit_ell[i], fitpar[4])
            #fmin = min(cube.data[i].min(),cube_fit.data[i].min()) - \
            #    max(cube.data[i].max(),cube_fit.data[i].max())/1e4
            ax.plot(r, cube.data[i] - fmin, 'b,')             # Data
            ax.plot(rfit, cube_fit.data[i] - fmin, 'r.',ms=1) # Model
            ax.set_autoscale_on(False)
            if skyDeg >= 0:
                ax.plot(rfit, psf[i] - fmin,'g.',ms=1) # PSF alone
                ax.plot(rfit, bkg[i] - fmin,'y.',ms=1) # Sky
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.9,0.8, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='right', transform=ax.transAxes)
            if method!='psf':
                ax.axvline(radius/SpaxelSize, color='y', lw=2)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Elliptical radius [spaxels]", fontsize=8)
                ax.set_ylabel("Flux + cte", fontsize=8)
            # ax.axis([0, rfit.max()*1.1, 
            #          cube.data[i][cube.data[i]>0].min()/1.2,
            #          cube.data[i].max()*1.2])

            # Binned values
            rb,db = radialbin(r, cube.data[i])
            ax.plot(rb, db - fmin, 'c.')
            rfb,fb = radialbin(rfit, cube_fit.data[i])
            ax.plot(rfb, fb - fmin, 'm.')

        # Missing energy (not activated by default)
        if opts.verbosity>=1:
            print_msg("Producing missing energy plot...", 1)
            figB = pylab.figure()
            figB.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.97)
            for i in xrange(nslice):        # Loop over slices
                ax = figB.add_subplot(nrow, ncol, i+1,
                                      yscale='log')
                r = ellRadius(cube.x,cube.y, xfit[i],yfit[i], 
                              fit_ell[i], fitpar[4])
                rfit = ellRadius(cube_fit.x,cube_fit.y, xfit[i],yfit[i],
                                 fit_ell[i], fitpar[4])
                # Binned values
                rb,db = radialbin(r, cube.data[i])
                rfb,fb = radialbin(rfit, cube_fit.data[i])
                tb = S.cumsum(rb*db)
                norm = tb.max()
                ax.plot(rb, 1 - tb/norm, 'c.')
                ax.plot(rfb, 1 - S.cumsum(rfb*fb)/norm, 'm.')
                if skyDeg >= 0:
                    rfb,pb = radialbin(rfit, psf[i])
                    rfb,bb = radialbin(rfit, bkg[i])
                    ax.plot(rfb, 1 - S.cumsum(rfb*pb)/norm, 'g.')
                    ax.plot(rfb, 1 - S.cumsum(rfb*bb)/norm, 'y.')
                pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), 
                           fontsize=6)
                ax.text(0.9,0.8, "%.0f" % cube.lbda[i], fontsize=8,
                        horizontalalignment='right', transform=ax.transAxes)
                if method!='psf':
                    ax.axvline(radius/SpaxelSize, color='y', lw=2)
                if ax.is_last_row() and ax.is_first_col():
                    ax.set_xlabel("Elliptical radius [spaxels]", fontsize=8)
                    ax.set_ylabel("Missing energy [fraction]", fontsize=8)

        # Radial Chi2 plot (not activated by default)
        if opts.verbosity>=1:
            print_msg("Producing radial chi2 plot...", 1)
            figA = pylab.figure()
            figA.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.97)
            for i in xrange(nslice):        # Loop over slices
                ax = figA.add_subplot(nrow, ncol, i+1, yscale='log')
                rfit = ellRadius(cube_fit.x,cube_fit.y, xfit[i],yfit[i],
                                 fit_ell[i], fitpar[4])
                chi2 = (cube.slice2d(i,coord='p') - \
                        cube_fit.slice2d(i,coord='p'))**2 / \
                        cube.slice2d(i,coord='p',var=True)
                ax.plot(rfit, chi2.flatten(), 'b.')
                pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), 
                           fontsize=6)
                ax.text(0.9,0.8, "%.0f" % cube.lbda[i], fontsize=8,
                        horizontalalignment='right', transform=ax.transAxes)
                if method!='psf':
                    ax.axvline(radius/SpaxelSize, color='y', lw=2)
                if ax.is_last_row() and ax.is_first_col():
                    ax.set_xlabel("Elliptical radius [spaxels]", fontsize=8)
                    ax.set_ylabel(ur"$\chi$²", fontsize=8)

        # Contour plot of each slice -----------------------------------------

        print_msg("Producing PSF contour plot %s..." % plot8, 1)

        fig8 = pylab.figure()
        fig8.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.97,
                             hspace=0.02, wspace=0.02)
        extent = (cube.x.min()-0.5,cube.x.max()+0.5,
                  cube.y.min()-0.5,cube.y.max()+0.5)
        for i in xrange(nslice):        # Loop over meta-slices
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
                        fmt=None, ecolor=good[i] and 'k' or 'r')
            ax.plot((xfit[i],),(yfit[i],), 'g+')
            if opts.method != 'psf':
                ax.add_patch(matplotlib.patches.Circle((xfit[i],yfit[i]),
                                                       radius/SpaxelSize,
                                                       fc='None', ec='y', lw=2))
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.1,0.1, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='left', transform=ax.transAxes)
            ax.axis(extent)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I", fontsize=8)
                ax.set_ylabel("J", fontsize=8)
            if not ax.is_last_row():
                ax.set_xticks([])
            if not ax.is_first_col():
                ax.set_yticks([])

        # Residuals of each slice --------------------------------------------

        print_msg("Producing residuals plot %s..." % plot5, 1)

        fig5 = pylab.figure()
        fig5.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.97,
                             hspace=0.02, wspace=0.02)
        for i in xrange(nslice):        # Loop over meta-slices
            ax   = fig5.add_subplot(ncol, nrow, i+1, aspect='equal')
            data = cube.slice2d(i, coord='p')
            var  = cube.slice2d(i, coord='p', var=True)
            fit  = cube_fit.slice2d(i, coord='p')
            res  = S.nan_to_num((data - fit)/S.sqrt(var))
            vmin,vmax = pylab.prctile(res, (3.,97.)) # Percentiles
            ax.imshow(res, origin='lower', extent=extent,
                      vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.plot((xfit[i],),(yfit[i],), 'g+')
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.1,0.1, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='left', transform=ax.transAxes)
            ax.axis(extent)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I", fontsize=8)
                ax.set_ylabel("J", fontsize=8)
            if not ax.is_last_row():
                ax.set_xticks([])
            if not ax.is_first_col():
                ax.set_yticks([])

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
