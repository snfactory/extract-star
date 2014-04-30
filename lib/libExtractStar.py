#!/usr/bin/env python
##############################################################################
## Filename:      libExtractStar.py
## Version:       $Revision$
## Description:   Extract_star utilities module
## Author:        $Author$
## $Id$
##############################################################################

"""Extract_star classes and functions."""

__author__ = "Y. Copin, C. Buton, E. Pecontal"
__version__ = '$Id$'

import re
import numpy as N
import scipy.linalg as SL               # More complete than numpy.linalg
from pySNIFS import SNIFS_cube
import ToolBox.Atmosphere as TA

def print_msg(str, limit, verb=0):
    """Print message 'str' if verbosity level (typically
    opts.verbosity) >= limit."""

    if verb >= limit:
        print str

def get_slices_lrange(cube, nmeta=12):
    """
    Get wavelength range from meta-sliced cube, because
    pySNIFS.SNIFS_cube manners are just too obfuscated.

    :param cube: pySNIFS.SNIFS_cube instance
    :param nmeta: number of meta-slices
    :return: (lstart, lend) of sliced cube.
    """

    from ToolBox.Arrays import metaslice

    # Should be the same definition as in extract_star
    slices = metaslice(cube.nslice, nmeta, trim=10)
    try:
        slice_cube = SNIFS_cube(e3d_file=cube.e3d_file, slices=slices)
    except (ValueError, AttributeError,):
        slice_cube = SNIFS_cube(fits3d_file=cube.fits3d_file, slices=slices)
        
    return slice_cube.lstart, slice_cube.lend

# PSF fitting ==============================

def fit_metaslices(cube, psf_fn, skyDeg=0, nsky=2, chi2fit=True, verbosity=1):
    """Adjust PSF parameters on (meta)slices of (meta)*cube* using PSF
    *psf_fn* and a background of polynomial degree *skyDeg*."""

    from ToolBox.Optimizer import approx_deriv
    from scipy.ndimage.filters import median_filter
    import pySNIFS_fit

    assert skyDeg >= -1, \
           "skyDeg=%d is invalid (should be >=-1)" % skyDeg

    npar_psf = 7                        # Number of parameters of the psf
    npar_sky = (skyDeg+1)*(skyDeg+2)/2  # Nb. param. in polynomial bkgnd

    cube_sky = SNIFS_cube()
    cube_sky.x = cube.x
    cube_sky.y = cube.y
    cube_sky.i = cube.i
    cube_sky.j = cube.j
    cube_sky.nslice = 1
    cube_sky.nlens = cube.nlens

    cube_star = SNIFS_cube()
    cube_star.x = cube.x
    cube_star.y = cube.y
    cube_star.i = cube.i
    cube_star.j = cube.j
    cube_star.nslice = 1
    cube_star.nlens = cube.nlens

    # PSF intensity + Sky + Bkgnd coeffs
    params  = N.zeros((cube.nslice,npar_psf+1+npar_sky), dtype='d')
    dparams = N.zeros((cube.nslice,npar_psf+1+npar_sky), dtype='d')
    chi2s   = N.zeros( cube.nslice,                      dtype='d')

    if nsky>7:                          # Nb of edge spx used for sky estimate
        raise ValueError('The number of edge pixels should be less than 7')
    skySpx = (cube_sky.i < nsky) | (cube_sky.i >= 15-nsky) | \
             (cube_sky.j < nsky) | (cube_sky.j >= 15-nsky)

    print_msg("  Set initial guess from 2D-Gaussian + Cte fit",
              2, verbosity)
    print_msg("  Adjusted parameters: [delta=0,theta=0],xc,yc,PA,ell,alpha,I,"
              "%d bkgndCoeffs" % (npar_sky if skyDeg>=0 else 0),
              2, verbosity)

    xc,yc = None,None                   # Position guess
    alpha = None                        # Alpha guess
    if cube.lstart < 5000.:             # Blue cube
        loop = range(cube.nslice)[::-1] # Blueward loop
    else:                               # Red cube
        loop = range(cube.nslice)       # Redward loop
    
    for i in loop:                      # Loop over cube slices
        print_msg("Meta-slice #%d/%d, %.0fA:" % (i+1,cube.nslice,cube.lbda[i]),
                  2, verbosity)
        cube_star.lbda = N.array([cube.lbda[i]])
        cube_star.data = cube.data[i, N.newaxis]
        cube_star.var  = cube.var[i,  N.newaxis]
        cube_sky.data  = cube.data[i, N.newaxis].copy() # To be modified
        cube_sky.var   = cube.var[i,  N.newaxis].copy()

        # Sky median estimate (from FoV edge spx)
        skyLev = N.median(cube_sky.data.T[skySpx].squeeze())

        if skyDeg > 0:
            # Fit a 2D polynomial of degree skyDeg on the edge pixels of a
            # given cube slice.
            cube_sky.var.T[~skySpx] = 0 # Discard central spaxels
            if not chi2fit:
                cube_sky.var[cube_sky.var > 0] = 1 # Least-square
            model_sky = pySNIFS_fit.model(data=cube_sky,
                                          func=['poly2D;%d' % skyDeg],
                                          param=[[skyLev] + [0.]*(npar_sky-1)],
                                          bounds=[[[0,None]] +
                                                  [[None,None]]*(npar_sky-1)])
            model_sky.fit()
            skyLev = model_sky.evalfit().squeeze() # Structured bkgnd estimate

        # Guess parameters for the current slice
        medstar = median_filter(cube_star.data[0], 3) - skyLev # (nspx,)
        imax = medstar.max()                    # Intensity
        if (xc,yc)==(None,None):                # No previous estimate
            xc = N.average(cube.x, weights=medstar) # Flux-weighted centroid
            yc = N.average(cube.y, weights=medstar)

        cube_sky.data -= skyLev
        if chi2fit:
            cube_sky.var = cube.var[i, N.newaxis] # Reset to cube.var for chi2
        else:
            cube_sky.var = None                   # Least-square

        model = pySNIFS_fit.model(data=cube_sky,
                                  func=['gaus2D','poly2D;0'],
                                  param=[[xc,yc,1,1,imax],[0]],
                                  bounds=[ [[-7.5,7.5]]*2 + # x0,y0
                                           [[0.1,5]]*2 +    # sx,sy
                                           [[0,None]],
                                           [[None,None]] ])
        model.fit(msge=(verbosity >= 4))

        if model.status:
            print "WARNING: gaussian fit on meta-slice " \
                      "did not converge [%d]" % model.status
            if alpha is None:
                alpha = 2.4
        else:
            xc,yc = model.fitpar[:2]    # Update centroid
            alpha = max(N.hypot(*model.fitpar[2:4]), 1.)

        # Filling in the guess parameter arrays (px) and bounds arrays (bx)
        p1 = [0., 0., xc, yc, 0., 1., alpha, imax] # psf parameters
        b1 = [[0, 0],                   # delta (unfitted)
              [0, 0],                   # theta (unfitted)
              [-10, 10],                # xc
              [-10, 10],                # yc
              [None, None],             # PA
              [0, None],                # ellipticity > 0
              [0.1, 10],                # alpha > 0
              [0, None]]                # Intensity > 0

        # alphaDeg & ellDeg set to 0 for meta-slice fits
        func = ['%s;%f,%f,%f,%f' % 
                (psf_fn.name, cube_star.spxSize, cube_star.lbda[0], 0, 0)]
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
        print_msg("  Initial guess: %s" % (p1+p2), 2, verbosity)

        # Chi2 vs. Least-square fit
        if not chi2fit:
            cube_star.var = None # Will be handled by pySNIFS_fit.model

        # Instantiate the model class and fit current slice
        model_star = pySNIFS_fit.model(data=cube_star, func=func,
                                       param=param, bounds=bounds,
                                       myfunc={psf_fn.name:psf_fn})
        model_star.fit(maxfun=400, msge=(verbosity >= 4))

        # Restore true chi2 (not reduced one), ie.
        # chi2 = ((cube_star.data-model_star.evalfit())**2/cube_star.var).sum()
        # If least-square fitting, this actually corresponds to
        # RSS=residual sum of squares
        model_star.khi2 *= model_star.dof

        # Check fit results
        if model_star.status>0:
            print "WARNING: fit of metaslice %d failed (status=%d=%s) " % \
                (i+1, model_star.status,
                 pySNIFS_fit.S.optimize.tnc.RCSTRINGS[model_star.status])
        elif not model_star.fitpar[5]>0:
            print "WARNING: ellipticity of metaslice %d is null" % (i+1)
            model_star.status = -1
        elif not 0<model_star.fitpar[6]<10:
            print "WARNING: alpha of metaslice %d is null" % (i+1)
            model_star.status = -1
        elif not model_star.fitpar[7]>0:
            print "WARNING: intensity of metaslice %d is null" % (i+1)
            model_star.status = -1

        # Error computation
        if not model_star.status:
            # Cannot use model_star.param_error to compute cov, since
            # it does not handle fixed parameters (lb=ub).
            hess = approx_deriv(model_star.objgrad, model_star.fitpar)
            cov = 2*SL.pinv2(hess[2:,2:]) # Discard 1st 2 lines (unfitted)
            diag = cov.diagonal()
            if (diag>0).all():
                dpar = N.concatenate(([0.,0.], N.sqrt(diag)))
            else:                       # Some negative diagonal elements!
                print "WARNING: negative covariance diagonal elements " \
                    "in metaslice %d" % (i+1)
                model_star.khi2 *= -1   # To be discarded
                dpar = N.zeros(len(dparams.T))
        else:                           # Set error to 0 if status
            model_star.khi2 *= -1       # To be discarded
            dpar = N.zeros(len(dparams.T))

        # Storing the result of the current slice parameters
        params[i]  = model_star.fitpar
        dparams[i] = dpar
        chi2s[i]   = model_star.khi2
        if chi2fit:
            print_msg("  Chi2 fit [%d, chi2/dof=%.2f/%d]: %s" % 
                      (model_star.status,model_star.khi2,model_star.dof,
                       model_star.fitpar), 1, verbosity)
        else:
            print_msg("  Lsq fit [%d, RSS/dof=%.2f/%d]: %s" % 
                      (model_star.status,model_star.khi2,model_star.dof,
                       model_star.fitpar), 1, verbosity)

        if verbosity >= 3:
            print "Gradient checks:"
            model_star.check_grad()

    return params,chi2s,dparams

# Point-source extraction ==============================

def extract_specs(cube, psf, skyDeg=0,
                  method='psf', radius=5., chi2fit=True, verbosity=0):
    """Extract object and sky spectra from *cube* using PSF --
    described by *psf*=(psf_fn,psf_ctes,psf_param) -- in presence of
    sky (polynomial degree *skyDeg*) using *method* ('psf':
    PSF-photometry, 'aperture': aperture photometry, or
    'optimal'). For aperture related methods, *radius* gives aperture
    radius in arcsec.

    Returns (lbda,sigspecs,varspecs) where sigspecs and varspecs are
    (nslice,npar+1).
    """

    assert method in ('psf','aperture','subaperture','optimal'), \
           "Unknown extraction method '%s'" % method
    assert skyDeg >= -1, \
           "skyDeg=%d is invalid (should be >=-1)" % skyDeg

    if (cube.var>1e20).any():
        print "WARNING: discarding infinite variances in extract_specs"
        cube.var[cube.var>1e20] = 0
    if (cube.var<0).any():              # There should be none anymore
        print "WARNING: discarding negative variances in extract_specs"
        cube.var[cube.var<0] = 0

    psf_fn, psf_ctes, psf_param = psf   # Unpack PSF description

    # The PSF parameters are only the shape parameters. We arbitrarily
    # set the intensity of each slice to 1.
    param = N.concatenate((psf_param, N.ones(cube.nslice)))

    # General linear least-squares fit: data = I*PSF + sky [ + a*x + b*y + ...]
    # See Numerical Recipes (2nd ed.), sect.15.4

    spxSize = psf_ctes[0]               # Spaxel size [arcsec]
    cube.x  = cube.i - 7                # x in spaxel
    cube.y  = cube.j - 7                # y in spaxel
    model = psf_fn(psf_ctes, cube)
    psf   = model.comp(param, normed=True) # nslice,nlens

    npar_sky = (skyDeg+1)*(skyDeg+2)/2  # Nb param. in polynomial bkgnd

    # Basis function matrix: BF (nslice,nlens,npar+1) (so-called X in NR)
    BF = N.zeros((cube.nslice,cube.nlens,npar_sky+1),'d')
    BF[:,:,0] = psf                     # Intensity    
    if npar_sky:                        # =0 when no background (skyDeg<=-1)
        BF[:,:,1] = 1                   # Constant background
        n = 2
        for d in xrange(1,skyDeg+1):
            for j in xrange(d+1):
                # Background polynomials as function of spaxel (centered)
                # position [spx]
                BF[:,:,n] = cube.x**(d-j) * cube.y**j 
                n += 1                  # Finally: n = npar_sky + 1

    # Chi2 (variance-weighted) vs. Least-square (unweighted) fit
    # *Note* that weight is actually 1/sqrt(var) (see NR)
    if chi2fit:
        weight = N.where(cube.var>0, 1/N.sqrt(cube.var), 0) # nslice,nlens
    else:
        weight = N.where(cube.var>0, 1, 0) # nslice,nlens

    # Design matrix (basis functions normalized by std errors)
    A = BF * weight[...,N.newaxis]      # nslice,nlens,npar+1
    b = weight*cube.data                # nslice,nlens

    # The linear least-squares fit AX = b could be done directly using
    #
    #   sigspecs = N.array([ N.linalg.lstsq(aa,bb)[0] for aa,bb in zip(A,b) ])
    #
    # but Alpha=dot(A.T,A) is needed anyway to compute covariance
    # matrix Cov=1/Alpha. Furthermore, linear resolution
    #
    #   [ N.linalg.solve(aa,bb) for aa,bb in zip(Alpha,Beta) ]
    #
    # can be replace by faster (~x10) matrix product
    #
    #   [ N.dot(cc,bb) for cc,bb in zip(Cov,Beta) ]
    #
    # since Cov=1/Alpha is readily available.
    #
    # "Solving Ax = b: inverse vs cholesky factorization" thread
    # (http://thread.gmane.org/gmane.comp.python.numeric.general/41365)
    # advocates to never invert a matrix directly: that's why we use
    # SVD-based inversion SL.pinv2.

    #Alpha = N.einsum('...jk,...jl',A,A)              # ~x2 slower
    Alpha = N.array([ N.dot(aa.T, aa) for aa in A ]) # nslice,npar+1,npar+1
    #Beta = N.einsum('...jk,...j',A,b)
    Beta = N.array([ N.dot(aa.T, bb) for aa,bb in zip(A,b) ]) # nslice,npar+1
    try:
        Cov = N.array([ SL.pinv2(aa) for aa in Alpha ])  # ns,np+1,np+1
    except SL.LinAlgError:
        raise SL.LinAlgError("Singular matrix during spectrum extraction")
    # sigspecs & varspecs = nslice x [Star,Sky,[slope_x...]]
    sigspecs = N.array([ N.dot(cc,bb)
                         for cc,bb in zip(Cov,Beta) ]) # nslice,npar+1
    varspecs  = N.array([ N.diag(cc) for cc in Cov ])  # nslice,npar+1

    # Compute the least-square variance using the chi2-case method
    # (errors are meaningless in pure least-square case)
    if not chi2fit:
        weight = N.where(cube.var>0, 1/N.sqrt(cube.var), 0)
        A = BF * weight[...,N.newaxis]
        Alpha = N.array([ N.dot(aa.T, aa) for aa in A ])
        try:
            Cov = N.array([ SL.pinv2(aa) for aa in Alpha ])
        except SL.LinAlgError:
            raise SL.LinAlgError("Singular matrix "
                                 "during variance extraction")
        varspecs = N.array([ N.diag(cc) for cc in Cov ])

    # Now, what about negative sky? The pb arises for short-exposures,
    # where there's probably no sky whatsoever (except if taken during
    # twilight), and where a (significantly) negative sky is actually
    # a shortcoming of the PSF. For long exposures, one expects "some"
    # negative sky values, where sky is compatible to 0.
    #
    # One could also use a NNLS fit to force parameter non-negativity:
    #
    #   [ pySNIFS_fit.fnnls(aa,bb)[0] for aa,bb in zip(Alpha,Beta) ]
    #
    # *BUT*:
    # 1. It is incompatible w/ non-constant sky (since it will force
    #    all sky coeffs to >0). This can therefore be done only if
    #    skyDeg=0 (it would otherwise involve optimization with
    #    constraints on sky positivity).
    # 2. There is no easy way to estimate covariance matrix from NNLS
    #    fit. Since an NNLS fit on a negative sky slice would probably
    #    always lead to a null sky, an NNLS fit is then equivalent to
    #    a standard 'PSF' fit without sky.

    if skyDeg==0:
        negSky = sigspecs[:,1]<0   # Test for presence of negative sky
        if negSky.any(): # and 'long' not in psf_fn.name.lower():
            print "WARNING: %d slices w/ sky<0 in extract_specs" % \
                  (len(negSky.nonzero()[0]))
            print_msg(str(cube.lbda[negSky]), 2, verbosity)
        #if 'short' in psf_fn.name:
            # For slices w/ sky<0, fit only PSF without background
            Alpha = N.array([ N.dot(aa,aa) for aa in A[negSky,:,0] ])
            Beta = N.array([ N.dot(aa,bb)
                             for aa,bb in zip(A[negSky,:,0],b[negSky]) ])
            Cov = 1/Alpha
            sigspecs[negSky,0] = Cov*Beta   # Linear fit without sky
            sigspecs[negSky,1] = 0          # Set sky to null
            varspecs[negSky,0] = Cov
            varspecs[negSky,1] = 0

    if method == 'psf':
        return cube.lbda,sigspecs,varspecs  # PSF extraction

    # Reconstruct background and subtract it from cube
    bkgnd     = N.zeros_like(cube.data)
    var_bkgnd = N.zeros_like(cube.var)
    if npar_sky:
        for d in xrange(1,npar_sky+1):  # Loop over sky components
            bkgnd     += (BF[:,:,d].T    * sigspecs[:,d]).T
            var_bkgnd += (BF[:,:,d].T**2 * varspecs[:,d]).T
    subData = cube.data - bkgnd         # Bkgnd subtraction (nslice,nlens)
    subVar = cube.var.copy()
    good = cube.var>0
    subVar[good] += var_bkgnd[good]     # Variance of bkgnd-sub. signal

    # Replace invalid data (var=0) by model PSF = Intensity*PSF
    if not good.all():
        print_msg("Replacing %d vx with modeled signal" % 
                  len((~good).nonzero()[0]), 1, verbosity)
        subData[~good] = (sigspecs[:,0]*psf.T).T[~good]

    # Plain summation over aperture

    # Aperture radius in spaxels
    aperRad = radius / spxSize
    print_msg("Aperture radius: %.2f arcsec = %.2f spx" % (radius,aperRad),
              1, verbosity)
    # Aperture center after ADR offset [spx] (nslice)
    xc = psf_param[2] + psf_param[0]*model.ADRcoeffs[:,0]*N.cos(psf_param[1])
    yc = psf_param[3] + psf_param[0]*model.ADRcoeffs[:,0]*N.sin(psf_param[1])
    # Radial distance from center [spx] (nslice,nlens)
    r = N.hypot((model.x.T - xc).T, (model.y.T - yc).T)
    # Circular aperture (nslice,nlens)
    # Use r<aperRad[:,N.newaxis] if radius is a (nslice,) vec.
    frac = (r < aperRad).astype('float')

    if method == 'subaperture':
        # Fractions accounting for subspaxels (a bit slow)
        newfrac = subaperture(xc, yc, aperRad, 4)
        # Remove bad spaxels since subaperture returns the full spaxel grid
        w = (~N.isnan(cube.slice2d(0).ravel())).nonzero()[0]
        frac = newfrac[:,w]

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

        if method == 'optimal':
            print "WARNING: Model extrapolation outside FoV " \
                  "not implemented for optimal summation."
        elif method == 'subaperture':
            print "WARNING: Model extrapolation outside FoV " \
                  "not implemented for sub-aperture summation."

    if hit.any() and method == 'aperture':

        # Extrapolate signal from PSF model
        print_msg("Signal extrapolation outside FoV...", 1, verbosity)

        # Extend usual range by ns spx on each side
        nw = 15 + 2*ns                  # New FoV size in spaxels
        mid = (7 + ns)                  # FoV center
        extRange = N.arange(nw) - mid
        extx,exty = N.meshgrid(extRange[::-1],extRange) # nw,nw
        extnlens = extx.size                 # = nlens' = nw**2
        print_msg("  Extend FoV by %d spx: nlens=%d -> %d" % 
                  (ns, model.nlens, extnlens), 1, verbosity)

        # Compute PSF on extended range (nslice,extnlens)
        extModel = psf_fn(psf_ctes, cube, coords=(extx,exty)) # Extended model
        extPsf   = extModel.comp(param, normed=True) # nslice,extnlens

        # Embed background-subtracted data in extended model PSF
        origData = subData.copy()
        origVar  = subVar.copy()
        subData  = (sigspecs[:,0]*extPsf.T).T # Extended model, nslice,extnlens
        subVar   = N.zeros((extModel.nslice,extModel.nlens))
        for i in xrange(model.nlens):
            # Embeb original spx i in extended model array by finding
            # corresponding index j in new array
            j, = ((extModel.x[0]==model.x[0,i]) & 
                  (extModel.y[0]==model.y[0,i])).nonzero()
            subData[:,j[0]] = origData[:,i]
            subVar[:,j[0]]  = origVar[:,i]

        r = N.hypot((extModel.x.T - xc).T, (extModel.y.T - yc).T)
        frac = (r < aperRad).astype('float')

    if method.endswith('aperture'):
        # Replace signal and variance estimates from plain summation
        sigspecs[:,0] = (frac    * subData).sum(axis=1)
        varspecs[:,0] = (frac**2 * subVar ).sum(axis=1)

        return cube.lbda,sigspecs,varspecs       # [Sub]Aperture extraction

    if method == 'optimal':

        from scipy.ndimage.filters import median_filter
        
        # Model signal = Intensity*PSF + bkgnd
        modsig = (sigspecs[:,0]*psf.T).T + bkgnd # nslice,nlens

        # One has to have a model of the variance. This can be estimated from
        # a simple 'photon noise + RoN' model on each slice: signal ~ alpha*N
        # (alpha = 1/flat-field coeff and N = photon counts) and variance ~ (N
        # + RoN**2) * alpha**2 = (signal/alpha + RoN**2) * alpha**2 =
        # alpha*signal + beta. This model disregards spatial component of
        # flat-field, which is supposed to be constant on FoV.

        # Model variance = alpha*Signal + beta
        coeffs = N.array([ polyfit_clip(modsig[s], cube.var[s], 1, clip=5)
                           for s in xrange(cube.nslice) ])
        coeffs = median_filter(coeffs, (5,1)) # A bit of smoothing...
        modvar = N.array([ N.polyval(coeffs[s], modsig[s])
                           for s in xrange(cube.nslice) ]) # nslice,nlens

        # Optimal weighting
        norm = (frac * psf).sum(axis=1) # PSF norm, nslice
        npsf = (psf.T / norm).T         # nslice,nlens
        weight = frac * npsf / modvar   # Unormalized weights, nslice,nlens
        norm = (weight * npsf).sum(axis=1) # Weight norm, nslice
        weight = (weight.T / norm).T    # Normalized weights, nslice,nlens

        # Replace signal and variance estimates from optimal summation
        sigspecs[:,0] = (weight     * subData).sum(axis=1)
        varspecs[:,0]  = (weight**2 * subVar ).sum(axis=1)
        
        return cube.lbda,sigspecs,varspecs       # Optimal extraction

# Resampling ========================================================

def subaperture(xc, yc, rc, f=0, nspaxel=15):
    """
    Compute aperture fraction for each spaxel with resampling

    :param xc: aperture X center
    :param yc: aperture Y center
    :param rc: aperture radius
    :param f: resampling factor (power of 2)
    :param nspaxel: spaxel grid side
    :return: spaxel flux fraction on original 15x15 grid
    """

    from ToolBox.Arrays import rebin
    
    # Resample spaxel center positions, originally [-7:7]
    f = 2.**f
    epsilon = 0.5 / f
    border = nspaxel / 2.
    r = N.linspace(-border + epsilon, border - epsilon, nspaxel*f)

    x,y = N.meshgrid(r, r)        # (x,y) positions of resampled array
    frac = N.ones(x.shape) / f**2 # Spaxel fraction

    xc = N.atleast_1d(xc)
    yc = N.atleast_1d(yc)
    assert xc.shape == yc.shape

    rc = N.atleast_1d(rc)
    if len(rc) == 1:              # One single radius?
        rc = N.repeat(rc, xc.shape)

    out = []
    # This loop could possibly be achieved with some higher order matrix
    for i,j,k in zip(xc, yc, rc):
        fr = frac.copy()
        fr[N.hypot(x-i, y-j) > k] = 0.  # subspaxels outside circle
        # Resample back to original size and sum
        out.append(rebin(fr, f).ravel())

    return N.array(out)

# Header information access utilities ===============================

def read_PT(hdr, MK_pressure=616., MK_temp=2.):
    """Read pressure [mbar] and temperature [C] from hdr (or use default
    Mauna-Kea values), and check value consistency."""

    if hdr is None:
        return MK_pressure, MK_temp

    pressure = hdr.get('PRESSURE', N.nan)
    if not 550 < pressure < 650:        # Non-std pressure
        print "WARNING: non-std pressure (%.0f mbar) updated to %.0f mbar" % \
              (pressure, MK_pressure)
        if isinstance(hdr, dict):       # pySNIFS.SNIFS_cube.e3d_data_header
            hdr['PRESSURE'] = MK_pressure
        else:                           # True pyfits header, add comment
            hdr['PRESSURE'] = (MK_pressure,"Default MK pressure [mbar]")
        pressure = MK_pressure

    temp = hdr.get('TEMP', N.nan)
    if not -20 < temp < 20:             # Non-std temperature
        print "WARNING: non-std temperature (%.0f C) updated to %.0f C" % \
              (temp, MK_temp)
        if isinstance(hdr, dict):       # pySNIFS.SNIFS_cube.e3d_data_header
            hdr['TEMP'] = MK_temp
        else:                           # True pyfits header, add comment
            hdr['TEMP'] = (MK_temp, "Default MK temperature [C]")
        temp = MK_temp

    return pressure,temp


def read_psf_name(hdr):
    """Return PSF class as read (or guessed) from header."""

    assert hdr['ES_METH']=='psf', \
           "PSF reconstruction only works for PSF spectro-photometry"

    try:
        psfname = hdr['ES_PSF']
    except KeyError:
        efftime = hdr['EFFTIME']
        print "WARNING: cannot read 'ES_PSF' keyword, " \
              "guessing from EFFTIME=%.0fs" % efftime
        # Assert it's an 'classic' PSF model (i.e. 'long' or 'short')
        psfname = 'long' if efftime > 12. else 'short'

    try:
        psfname, psfmodel = psfname.split(', ') # "name, model"
    except ValueError:
        if len(psfname.split())==2:     # Chromatic PSF: 'short|long blue|red'
            psfmodel = 'chromatic'
        else:                           # Classic PSF: 'short|long'
            psfmodel = 'classic'

    # Convert PSF name (e.g. 'short red') to PSF class name
    # ('ShortRed_ExposurePSF')
    fnname = ''.join(map(str.capitalize,psfname.split())) + '_ExposurePSF'
    print "PSF name/model: %s/%s [%s]" % (psfname, psfmodel, fnname)

    psffn = eval(fnname)
    if psfmodel.endswith('powerlaw'):
        psffn.model += '-powerlaw'

    return psffn


def read_psf_ctes(hdr, lrange=()):
    """Read PSF constants [lRef,alphaDeg,ellDeg] from header."""

    assert ('ES_LMIN' in hdr and 'ES_LMAX' in hdr) or lrange, \
           'ES_LMIN/ES_LMAX not found and lrange not set'

    # This reproduces exactly the PSF parameters used by
    # extract_specs(full_cube...)
    if lrange:
        lmin, lmax = lrange
    else:
        lmin = hdr['ES_LMIN']
        lmax = hdr['ES_LMAX']
    lref = (lmin+lmax)/2.

    # Count up alpha/ell coefficients (ES_Ann/ES_Enn) to get the
    # polynomial degrees
    countKeys = lambda regexp: \
                len([ k for k in hdr.keys() if re.match(regexp,k) ])

    adeg = countKeys('ES_A\d+$') - 1
    edeg = countKeys('ES_E\d+$') - 1
    print "PSF constants: lRef=%.2fA, alphaDeg=%d, ellDeg=%d" % (lref,adeg,edeg)

    return [lref,adeg,edeg]


def read_psf_param(hdr, lrange=()):
    """Read (7+ellDeg+alphaDeg) PSF parameters from header:
    delta,theta,x0,y0,PA,e0,...en,a0,...an."""

    assert ('ES_LMIN' in hdr and 'ES_LMAX' in hdr) or lrange, \
           'ES_LMIN/ES_LMAX not found and lrange not set'

    # Polynomial coeffs in lr~ = lambda/LbdaRef - 1
    c_ell = [ v for k,v in hdr.items() if re.match('ES_E\d+$',k) ]
    c_alp = [ v for k,v in hdr.items() if re.match('ES_A\d+$',k) ]

    if lrange:
        lmin, lmax = lrange
    else:
        lmin = hdr['ES_LMIN']
        lmax = hdr['ES_LMAX']
    lref = hdr['ES_LREF']       # Reference wavelength [A]

    psfname = hdr['ES_PSF']

    # Convert polynomial coeffs from lr~ = lambda/LbdaRef - 1 = a+b*lr
    # back to lr = (2*lambda - (lmin+lmax))/(lmax-lmin)
    a = (lmin+lmax) / (2*lref) - 1
    b = (lmax-lmin) / (2*lref)
    ecoeffs = polyConvert(c_ell, trans=(a,b), backward=True).tolist()
    if 'powerlaw' not in psfname:
        acoeffs = polyConvert(c_alp, trans=(a,b), backward=True).tolist()
    else:
        acoeffs = c_alp

    x0 = hdr['ES_XC']           # Reference position [spx]
    y0 = hdr['ES_YC']
    pa = hdr['ES_XY']           # (Nearly) position angle

    # This reproduces exactly the PSF parameters used by
    # extract_specs(full_cube...)
    pressure,temp = read_PT(hdr)
    airmass = hdr['ES_AIRM']    # Effective airmass
    parang = hdr['ES_PARAN']    # Effective parallactic angle [deg]
    adr = TA.ADR(pressure, temp, lref=(lmin+lmax)/2,
                 airmass=airmass, parangle=parang)
    x0,y0 = adr.refract(x0, y0, lref, unit=0.43, backward=True)

    print "PSF parameters: airmass=%.3f, parangle=%.1fdeg, " \
          "refpos=%.2fx%.2f spx @%.2fA" % (airmass,parang,x0,y0,(lmin+lmax)/2)

    return [adr.delta,adr.theta,x0,y0,pa] + ecoeffs + acoeffs

# Polynomial utilities ======================================================

def polyEval(coeffs, x):
    """Evaluate polynom sum_i ci*x**i on x. It uses 'natural' convention for
    polynomial coeffs: [c0,c1...,cn] (opposite to N.polyfit)."""

    if N.isscalar(x):
        y = 0                           # Faster on scalar
        for i,c in enumerate(coeffs):
            # Incremental computation of x**i is only slightly faster
            y += c * x**i
    else:                               # Faster on arrays
        y = N.polyval(coeffs[::-1], x)  # Beware coeffs order!

    return y

def polyConvMatrix(n, trans=(0,1)):
    """Return the upper triangular matrix (i,k) * b**k * a**(i-k), that
    converts polynomial coeffs for x~:=a+b*x (P~ = a0~ + a1~*x~ + a2~*x~**2 +
    ...) in polynomial coeffs for x (P = a0 + a1*x + a2*x**2 +
    ...). Therefore, (a,b)=(0,1) gives identity."""

    from scipy.misc import comb
    a,b = trans
    m = N.zeros((n,n), dtype='d')
    for r in range(n):
        for c in range(r,n):
            m[r,c] = comb(c,r) * b**r * a**(c-r)
    return m

def polyConvert(coeffs, trans=(0,1), backward=False):
    """Converts polynomial coeffs for x (P = a0 + a1*x + a2*x**2 + ...) in
    polynomial coeffs for x~:=a+b*x (P~ = a0~ + a1~*x~ + a2~*x~**2 +
    ...). Therefore, (a,b)=(0,1) makes nothing. If backward, makes the
    opposite transformation.

    Note: backward transformation could be done using more general
    polynomial composition polyval, but forward transformation is a
    long standing issue in the general case (functional decomposition
    of univariate polynomial)."""

    a,b = trans
    if not backward:
        a = -float(a)/float(b)
        b = 1/float(b)
    return N.dot(polyConvMatrix(len(coeffs), (a,b)),coeffs)

def polyfit_clip(x, y, deg, clip=3, nitermax=10):
    """Least squares polynomial fit with sigma-clipping (if clip>0). Returns
    polynomial coeffs w/ same convention as N.polyfit: [cn,...,c1,c0]."""

    good = N.ones(y.shape, dtype='bool')
    niter = 0
    while True:
        niter += 1
        coeffs = N.polyfit(x[good], y[good], deg)
        old = good
        if clip:
            dy = N.polyval(coeffs, x) - y
            good = N.absolute(dy) < clip*N.std(dy)
        if (good==old).all(): break     # No more changes, stop there
        if niter > nitermax:            # Max. # of iter, stop there
            print "polyfit_clip reached max. # of iterations: " \
                  "deg=%d, clip=%.2f x %f, %d px removed" % \
                  (deg, clip, N.std(dy), len((~old).nonzero()[0]))
            break
        if y[good].size <= deg+1:
            raise ValueError("polyfit_clip: Not enough points left (%d) " 
                             "for degree %d" % (y[good].size,deg))
    return coeffs

def chebNorm(x, xmin, xmax):
    """Normalization [xmin,xmax] to [-1,1]"""

    return ( 2*x - (xmax+xmin) ) / (xmax-xmin)

def chebEval(pars, nx, chebpolys=[]):
    """Orthogonal Chebychev polynomial expansion, x should be already
    normalized in [-1,1]."""

    from scipy.special import chebyu

    if len(chebpolys)<len(pars):
        print "Initializing Chebychev polynomials up to order %d" % len(pars)
        chebpolys[:] = [ chebyu(i) for i in range(len(pars)) ]

    return N.sum( [ par*cheb(nx) for par,cheb in zip(pars,chebpolys) ], axis=0)

def powerLawEval(coeffs, x):
    """Evaluate (curved) power-law:
    coeffs[-1] * x**(coeffs[-2] + coeffs[-3]*(x-1) + ...)
    """

    return coeffs[-1] * x**N.polyval(coeffs[:-1], x-1)

def powerLawFit(x, y, deg=2, guess=None):

    import ToolBox.Optimizer as TO
    import scipy.optimize as O

    if guess is None:
        guess = [0.]*(deg-1) + [-1.,2.]
    else:
        assert len(guess)==(deg+1)

    model = TO.Model(powerLawEval)
    data = TO.DataSet(y, x=x)
    fit = TO.Fitter(model, data)
    lsqPars,msg = O.leastsq(fit.residuals, guess, args=(x,))

    if msg <= 4:
        return lsqPars
    else:
        raise ValueError("powerLawFit did not converge")

# Ellipse utilities ==============================

def quadEllipse(a,b,c,d,f,g):
    """Ellipse elements (center, semi-axes and PA) from the general
    quadratic curve a*x2 + 2*b*x*y + c*y2 + 2*d*x + 2*f*y + g = 0.

    http://mathworld.wolfram.com/Ellipse.html"""

    D = N.linalg.det([[a,b,d],[b,c,f],[d,f,g]])
    J = N.linalg.det([[a,b],[b,c]])
    I = a+c
    if not (D!=0 and J>0 and D/I<0):
        #raise ValueError("Input quadratic curve does not correspond to "
        #                 "an ellipse: D=%f!=0, J=%f>0, D/I=%f<0" % (D,J,D/I))
        return 0,0,-1,-1,0
    elif a==c and b==0:
        #raise ValueError("Input quadratic curve correspond to a circle")
        pass

    b2mac = b**2 - a*c
    # Center of the ellipse
    x0 = (c*d - b*f) / b2mac
    y0 = (a*f - b*d) / b2mac
    # Semi-axes lengthes
    ap = N.sqrt( 2*(a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g) /
                 (b2mac * (N.sqrt((a-c)**2 + 4*b**2) - (a+c))) )
    bp = N.sqrt( 2*(a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g) /
                 (b2mac * (-N.sqrt((a-c)**2 + 4*b**2) - (a+c))) )
    # Position angle
    if b==0:
        phi = 0
    else:
        phi = N.tan((a-c)/(2*b))/2
    if a>c:
        phi += N.pi/2

    return x0,y0,ap,bp,phi

def flatAndPA(cy2, c2xy):
    """Return flattening q=b/a and position angle PA [deg] for ellipse defined
    by x**2 + cy2*y**2 + 2*c2xy*x*y = 1.
    """

    x0,y0,a,b,phi = quadEllipse(1,c2xy,cy2,0,0,-1)
    assert a>0 and b>0, "Input equation does not correspond to an ellipse"
    q = b/a                             # Flattening
    pa = phi*TA.RAD2DEG                 # From rad to deg

    return q,pa

# PSF classes ================================================================

class ExposurePSF:
    """
    Empirical PSF 3D function used by the L{model} class.

    Note that the so-called PA parameter is not the PA of the adjusted
    ellipse, but half the x*y coefficient. Similarly, ell is not the
    ellipticity, but the y**2 coefficient: x2 + ell*y2 + 2*PA*x*y + ... = 0.
    See quadEllipse/flatAndPA for conversion.
    """

    def __init__(self, psf_ctes, cube, coords=None):
        """Initiating the class.
        @param psf_ctes: Internal parameters (pixel size in cube spatial unit,
                         reference wavelength and polynomial degrees).
        @param cube: Input cube. This is a L{SNIFS_cube} object.
        @param coords: if not None, should be (x,y).
        """
        self.spxSize  = psf_ctes[0]     # Spaxel size [arcsec]
        self.lref = psf_ctes[1]         # Reference wavelength [AA]
        self.alphaDeg = int(psf_ctes[2]) # Alpha polynomial degree
        self.ellDeg   = int(psf_ctes[3]) # Ellip polynomial degree

        self.npar_cor = 7 + self.ellDeg + self.alphaDeg # PSF parameters
        self.npar_ind = 1               # Intensity parameters per slice
        self.nslice = cube.nslice
        self.npar = self.npar_cor + self.npar_ind*self.nslice

        # Name of PSF parameters
        self.parnames = ['delta','theta','x0','y0','PA'] + \
                        ['e%d' % i for i in range(self.ellDeg+1)] + \
                        ['a%d' % i for i in range(self.alphaDeg+1)] + \
                        ['i%02d' % (i+1) for i in range(self.nslice)]

        if coords is None:
            self.nlens = cube.nlens
            self.x = N.resize(cube.x, (self.nslice,self.nlens)) # nslice,nlens
            self.y = N.resize(cube.y, (self.nslice,self.nlens))
        else:
            x = coords[0].ravel()
            y = coords[1].ravel()
            assert len(x)==len(y), \
                   "Incompatible coordinates (%d/%d)" % (len(x),len(y))
            self.nlens = len(x)
            self.x = N.resize(x, (self.nslice,self.nlens)) # nslice,nlens
            self.y = N.resize(y, (self.nslice,self.nlens))
        self.l = N.resize(cube.lbda, (self.nlens,self.nslice)).T # nslice,nlens

        if self.nslice > 1:
            self.lmin = cube.lstart
            self.lmax = cube.lend
            self.lrel = chebNorm(self.l, self.lmin, self.lmax) # From -1 to +1
        else:
            self.lmin,self.lmax = -1,+1
            self.lrel = self.l

        # ADR in spaxels (nslice,nlens)
        if hasattr(cube,'e3d_data_header'): # Read from cube if possible
            pressure,temp = read_PT(cube.e3d_data_header)
        else:
            pressure,temp = read_PT(None)   # Get default values for P and T
        self.nRef = TA.atmosphericIndex(self.lref, P=pressure, T=temp)
        self.ADRcoeffs = ( self.nRef - 
                           TA.atmosphericIndex(self.l, P=pressure, T=temp) ) * \
                           TA.RAD2ARC / self.spxSize # l > l_ref <=> coeff > 0

    def comp(self, param, normed=False):
        """
        Compute the function.
        @param param: Input parameters of the polynomial. A list of numbers:
        - C{param[0:7+n+m]}: The n parameters of the PSF shape
          - C{param[0]}: Atmospheric dispersion power
          - C{param[1]}: Atmospheric dispersion position angle
          - C{param[2]}: X center at the reference wavelength
          - C{param[3]}: Y center at the reference wavelength
          - C{param[4]}: Position angle
          - C{param[5:6+n]}: Ellipticity
                             (n:polynomial degree of ellipticity)
          - C{param[6+n:7+n+m]}: Moffat scale
                                 (m:polynomial degree of alpha)
        - C{param[7+m+n:]}: Intensity parameters
                            (one for each slice in the cube).
        @param normed: Should the function be normalized (integral)
        """

        self.param = N.asarray(param)

        # ADR params
        delta = self.param[0]
        theta = self.param[1]
        xc    = self.param[2]
        yc    = self.param[3]
        x0 = xc + delta*self.ADRcoeffs*N.sin(theta) # nslice,nlens
        y0 = yc - delta*self.ADRcoeffs*N.cos(theta)

        # Other params
        PA          = self.param[4]
        ellCoeffs   = self.param[5:6+self.ellDeg]
        alphaCoeffs = self.param[6+self.ellDeg:self.npar_cor]

        ell = polyEval(ellCoeffs, self.lrel) # nslice,nlens
        if self.model.endswith('powerlaw'):
            alpha = powerLawEval(alphaCoeffs, self.l/self.lref)
        else:
            alpha = polyEval(alphaCoeffs, self.lrel)

        # PSF model
        if self.model=='chromatic':  # Includes chromatic correlations
            lcheb = chebNorm(self.l, *self.chebRange)
            b0 = chebEval(self.beta0,  lcheb)
            b1 = chebEval(self.beta1,  lcheb)
            s0 = chebEval(self.sigma0, lcheb)
            s1 = chebEval(self.sigma1, lcheb)
            e0 = chebEval(self.eta0,   lcheb)
            e1 = chebEval(self.eta1,   lcheb)
        else:                        # Achromatic correlations
            b0 = self.beta0
            b1 = self.beta1
            s0 = self.sigma0
            s1 = self.sigma1
            e0 = self.eta0
            e1 = self.eta1
        sigma = s0 + s1*alpha
        beta  = b0 + b1*alpha
        eta   = e0 + e1*alpha

        # Gaussian + Moffat
        dx = self.x - x0
        dy = self.y - y0
        # CAUTION: ell & PA are not the true ellipticity and position angle!
        r2 = dx**2 + ell*dy**2 + 2*PA*dx*dy
        gaussian = N.exp(-0.5*r2/sigma**2)
        moffat = (1 + r2/alpha**2)**(-beta)

        # Function
        val = self.param[self.npar_cor:,N.newaxis] * (moffat + eta*gaussian)

        # The 3D psf model is not normalized to 1 in integral. The result must
        # be renormalized by (2*eta*sigma**2 + alpha**2/(beta-1)) *
        # N.pi/sqrt(ell)
        if normed:
            val /= N.pi*( 2*eta*sigma**2 + alpha**2/(beta-1) )/N.sqrt(ell)

        return val

    def deriv(self, param):
        """
        Compute the derivative of the function with respect to its parameters.
        @param param: Input parameters of the polynomial.
                      A list numbers (see L{SNIFS_psf_3D.comp}).
        """

        self.param = N.asarray(param)

        # ADR params
        delta = self.param[0]
        theta = self.param[1]
        xc    = self.param[2]
        yc    = self.param[3]
        costheta = N.cos(theta)
        sintheta = N.sin(theta)
        x0 = xc + delta*self.ADRcoeffs*sintheta
        y0 = yc - delta*self.ADRcoeffs*costheta

        # Other params
        PA  = self.param[4]
        ellCoeffs   = self.param[5:6+self.ellDeg]
        alphaCoeffs = self.param[6+self.ellDeg:self.npar_cor]

        ell = polyEval(ellCoeffs, self.lrel)
        if self.model.endswith('powerlaw'):
            alpha = powerLawEval(alphaCoeffs, self.l/self.lref)
        else:
            alpha = polyEval(alphaCoeffs, self.lrel)
        
        # PSF model
        if self.model=='chromatic':  # Includes chromatic correlations
            lcheb = chebNorm(self.l, *self.chebRange)
            b0 = chebEval(self.beta0,  lcheb)
            b1 = chebEval(self.beta1,  lcheb)
            s0 = chebEval(self.sigma0, lcheb)
            s1 = chebEval(self.sigma1, lcheb)
            e0 = chebEval(self.eta0,   lcheb)
            e1 = chebEval(self.eta1,   lcheb)
        else:                        # Achromatic correlations
            b0 = self.beta0
            b1 = self.beta1
            s0 = self.sigma0
            s1 = self.sigma1
            e0 = self.eta0
            e1 = self.eta1
        sigma = s0 + s1*alpha
        beta  = b0 + b1*alpha
        eta   = e0 + e1*alpha

        # Gaussian + Moffat
        dx = self.x - x0
        dy = self.y - y0
        dy2 = dy**2
        r2 = dx**2 + ell*dy2 + 2*PA*dx*dy
        sigma2 = sigma**2
        gaussian = N.exp(-0.5*r2/sigma2)
        alpha2 = alpha**2
        ea = 1 + r2/alpha2
        moffat = ea**(-beta)

        # Derivatives
        grad = N.zeros((self.npar_cor+self.npar_ind,)+self.x.shape,'d')
        j1 = eta/sigma2
        j2 = 2*beta/ea/alpha2
        tmp = gaussian*j1 + moffat*j2
        grad[2] = tmp*(    dx + PA*dy)  # dPSF/dx0
        grad[3] = tmp*(ell*dy + PA*dx)  # dPSF/dy0
        grad[0] =       self.ADRcoeffs*(sintheta*grad[2] - costheta*grad[3])
        grad[1] = delta*self.ADRcoeffs*(sintheta*grad[3] + costheta*grad[2])
        grad[4] = -tmp   * dx*dy        # dPSF/dPA
        for i in xrange(self.ellDeg + 1): # dPSF/dei
            grad[5+i] = -tmp/2 * dy2 * self.lrel**i
        dalpha = gaussian * ( e1 + s1*r2*j1/sigma ) + \
                 moffat * ( -b1*N.log(ea) + r2*j2/alpha ) # dPSF/dalpha
        if self.model.endswith('powerlaw'):
            lrel = self.l/self.lref
            imax = 6+self.ellDeg+self.alphaDeg
            grad[imax] = dalpha * lrel**N.polyval(alphaCoeffs[:-1], lrel-1)
            if self.alphaDeg:
                grad[imax-1] = grad[imax] * alphaCoeffs[-1] * N.log(lrel)
                for i in range(imax-2, imax-self.alphaDeg-1, -1):
                    grad[i] = grad[i+1] * (lrel-1)  # dPSF/dai, i=<0,alphaDeg>
        else:
            for i in xrange(self.alphaDeg + 1): # dPSF/dai, i=<0,alphaDeg>
                grad[6+self.ellDeg+i] = dalpha * self.lrel**i
        grad[:self.npar_cor] *= self.param[N.newaxis,self.npar_cor:,N.newaxis]
        grad[self.npar_cor] = moffat + eta*gaussian # dPSF/dI

        return grad

    def _HWHM_fn(self, r, alphaCoeffs, lbda):
        """Half-width at half maximum function (=0 at HWHM)."""

        if self.model.endswith('powerlaw'):
            alpha = powerLawEval(alphaCoeffs, lbda/self.lref)
        else:
            alpha = polyEval(alphaCoeffs, chebNorm(lbda, self.lmin, self.lmax))

        if self.model=='chromatic':
            lcheb = chebNorm(lbda, *self.chebRange)
            b0 = chebEval(self.beta0,  lcheb)
            b1 = chebEval(self.beta1,  lcheb)
            s0 = chebEval(self.sigma0, lcheb)
            s1 = chebEval(self.sigma1, lcheb)
            e0 = chebEval(self.eta0,   lcheb)
            e1 = chebEval(self.eta1,   lcheb)
        else:
            b0 = self.beta0
            b1 = self.beta1
            s0 = self.sigma0
            s1 = self.sigma1
            e0 = self.eta0
            e1 = self.eta1
        sigma = s0 + s1*alpha
        beta  = b0 + b1*alpha
        eta   = e0 + e1*alpha
        gaussian = N.exp(-0.5*r**2/sigma**2)
        moffat = (1 + r**2/alpha**2)**(-beta)

        # PSF=moffat + eta*gaussian, maximum is 1+eta
        return moffat + eta*gaussian - (eta + 1)/2

    def FWHM(self, param, lbda):
        """Estimate FWHM of PSF at wavelength lbda."""

        from scipy.optimize import fsolve

        alphaCoeffs = param[6+self.ellDeg:self.npar_cor]
        # Compute FWHM from radial profile
        fwhm = 2*fsolve(func=self._HWHM_fn, x0=1., args=(alphaCoeffs,lbda))

        # Beware: scipy-0.8.0 fsolve returns a size 1 array
        return N.squeeze(fwhm)  # In spaxels


class Long_ExposurePSF(ExposurePSF):
    """Classic PSF model (achromatic correlations) for long exposures."""

    name = 'long'
    model = 'classic'

    beta0  = 1.685
    beta1  = 0.345
    sigma0 = 0.545
    sigma1 = 0.215
    eta0   = 1.04
    eta1   = 0.00

class Short_ExposurePSF(ExposurePSF):
    """Classic PSF model (achromatic correlations) for short
    exposures."""

    name = 'short'
    model = 'classic'

    beta0  = 1.395
    beta1  = 0.415
    sigma0 = 0.56
    sigma1 = 0.2
    eta0   = 0.6
    eta1   = 0.16


class LongBlue_ExposurePSF(ExposurePSF):
    """PSF model with chromatic correlations (2nd order Chebychev
    polynomial) for long, blue exposures."""

    name = 'long blue'
    model = 'chromatic'
    chebRange = (3399.,5100.)      # Domain of validity of Chebychev expansion

    beta0  = [ 1.220, 0.016,-0.056] # b00,b01,b02
    beta1  = [ 0.590, 0.004, 0.014] # b10,b11,b12
    sigma0 = [ 0.710,-0.024, 0.016] # s00,s01,s02
    sigma1 = [ 0.119, 0.001,-0.004] # s10,s11,s12
    eta0   = [ 0.544,-0.090, 0.039] # e00,e01,e02
    eta1   = [ 0.223, 0.060,-0.020] # e10,e11,e12

class LongRed_ExposurePSF(ExposurePSF):
    """PSF model with chromatic correlations (2nd order Chebychev
    polynomial) for long, red exposures."""

    name = 'long red'
    model = 'chromatic'
    chebRange = (5318.,9508.)      # Domain of validity of Chebychev expansion

    beta0  = [ 1.205,-0.100,-0.031] # b00,b01,b02
    beta1  = [ 0.578, 0.062, 0.028] # b10,b11,b12
    sigma0 = [ 0.596, 0.044, 0.011] # s00,s01,s02
    sigma1 = [ 0.173,-0.035,-0.008] # s10,s11,s12
    eta0   = [ 1.366,-0.184,-0.126] # e00,e01,e02
    eta1   = [-0.134, 0.121, 0.054] # e10,e11,e12

class ShortBlue_ExposurePSF(ExposurePSF):
    """PSF model with chromatic correlations (2nd order Chebychev
    polynomial) for short, blue exposures."""

    name = 'short blue'
    model = 'chromatic'
    chebRange = (3399.,5100.)      # Domain of validity of Chebychev expansion

    beta0  = [ 1.355, 0.023,-0.042] # b00,b01,b02
    beta1  = [ 0.524,-0.012, 0.020] # b10,b11,b12
    sigma0 = [ 0.492,-0.037, 0.000] # s00,s01,s02
    sigma1 = [ 0.176, 0.016, 0.000] # s10,s11,s12
    eta0   = [ 0.499, 0.080, 0.061] # e00,e01,e02
    eta1   = [ 0.316,-0.015,-0.050] # e10,e11,e12

class ShortRed_ExposurePSF(ExposurePSF):
    """PSF model with chromatic correlations (2nd order Chebychev
    polynomial) for short, red exposures."""

    name = 'short red'
    model = 'chromatic'
    chebRange = (5318.,9508.)      # Domain of validity of Chebychev expansion

    beta0  = [ 1.350,-0.030,-0.012] # b00,b01,b02
    beta1  = [ 0.496, 0.032, 0.020] # b10,b11,b12
    sigma0 = [ 0.405,-0.003, 0.000] # s00,s01,s02
    sigma1 = [ 0.212,-0.017, 0.000] # s10,s11,s12
    eta0   = [ 0.704,-0.060, 0.044] # e00,e01,e02
    eta1   = [ 0.343, 0.113,-0.045] # e10,e11,e12
