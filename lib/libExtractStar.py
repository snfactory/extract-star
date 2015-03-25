#!/usr/bin/env python
# -*- coding: utf-8 -*-
##############################################################################
# Filename:      libExtractStar.py
# Version:       $Revision$
# Description:   Extract_star utilities module
# Modified at:   $Date$
# Author:        $Author$
# $Id$
##############################################################################

"""Extract_star classes and functions."""

__author__ = "Y. Copin, C. Buton, E. Pecontal"
__version__ = '$Id$'

import re
import itertools                        # product
import numpy as N
import scipy.linalg as SL               # More complete than numpy.linalg
import scipy.optimize as SO

from pySNIFS import SNIFS_cube

import ToolBox.Atmosphere as TA

LbdaRef = 5000.               # Use constant ref. wavelength for easy comparison
SpxSize = SNIFS_cube.spxSize  # Spaxel size in arcsec


def print_msg(str, limit, verb=0):
    """
    Print message 'str' if verbosity level (typically opts.verbosity)
    >= limit.
    """

    if verb >= limit:
        print str

# PSF fitting ==============================


def fit_metaslices(cube, psf_fn, skyDeg=0, nsky=2, chi2fit=True,
                   scalePriors=0., seeingPrior=None, posPrior=None,
                   airmass=1., verbosity=1):
    """
    Adjust PSF parameters on each (meta)slices of (meta)*cube* using
    PSF *psf_fn* and a background of polynomial degree *skyDeg*. Add
    priors on seeing and position if any.
    """

    from scipy.ndimage.filters import median_filter
    import pySNIFS_fit

    assert skyDeg >= -1, \
        "skyDeg=%d is invalid (should be >=-1)" % skyDeg

    npar_psf = 7                        # Number of parameters of the psf
    # Nb. param. in polynomial bkgnd
    npar_sky = (skyDeg + 1) * (skyDeg + 2) / 2

    cube_sky = SNIFS_cube()             # 1-slice cube for background fit
    cube_sky.x = cube.x
    cube_sky.y = cube.y
    cube_sky.i = cube.i
    cube_sky.j = cube.j
    cube_sky.nslice = 1
    cube_sky.nlens = cube.nlens

    cube_star = SNIFS_cube()            # 1-slice cube for point-source fit
    cube_star.x = cube.x
    cube_star.y = cube.y
    cube_star.i = cube.i
    cube_star.j = cube.j
    cube_star.nslice = 1
    cube_star.nlens = cube.nlens

    # PSF intensity + Sky + Bkgnd coeffs
    params = N.zeros((cube.nslice, npar_psf + 1 + npar_sky), dtype='d')
    dparams = N.zeros((cube.nslice, npar_psf + 1 + npar_sky), dtype='d')
    chi2s = N.zeros(cube.nslice, dtype='d')

    # Nb of edge spx used for sky estimate
    if nsky > 7:
        raise ValueError('The number of edge pixels should be less than 7')
    skySpx = (cube_sky.i < nsky) | (cube_sky.i >= 15 - nsky) | \
             (cube_sky.j < nsky) | (cube_sky.j >= 15 - nsky)

    print_msg("  Set initial guess from 2D-Gaussian + Cte fit", 2, verbosity)
    parnames = ['delta=0', 'theta=0', 'xc', 'yc', 'xy', 'y2', 'alpha', 'I'] + \
               ['B%02d' % j for j in range(npar_sky)]
    print_msg("  Adjusted parameters: %s" % ','.join(parnames), 2, verbosity)

    xc, yc = None, None                  # Position guess at lmid
    alpha = None                         # Alpha guess
    channel = 'B' if cube.lstart < 5000. else 'R'
    if channel == 'B':                   # Blue cube
        loop = range(cube.nslice)[::-1]  # Blueward loop
    else:                                # Red cube
        loop = range(cube.nslice)        # Redward loop

    # Hyper-term on alpha from seeing prior and on position
    if scalePriors:
        if not psf_fn.model.endswith('powerlaw'):
            raise NotImplementedError
        if seeingPrior:
            print "  Seeing prior: %.2f\"" % seeingPrior
        if posPrior:
            print "  Position prior: %+.2f x %+.2f spx" % posPrior

    for i in loop:                      # Loop over cube slices
        # Fill-in the meta-slice
        print_msg((" Meta-slice #%d/%d, %.0f A " %
                   (i + 1, cube.nslice, cube.lbda[i])).center(50, '-'),
                  2, verbosity)
        # Single-slice cubes
        cube_star.lbda = N.array([cube.lbda[i]])
        cube_star.data = cube.data[i, N.newaxis]
        cube_star.var = cube.var[i, N.newaxis]
        cube_sky.data = cube.data[i, N.newaxis].copy()  # Will be modified
        cube_sky.var = cube.var[i, N.newaxis].copy()

        # Sky median estimate from FoV edge spx
        skyLev = N.median(cube_sky.data[:, skySpx], axis=None)

        if skyDeg > 0:
            # Fit a 2D polynomial of degree skyDeg on the edge pixels
            # of a given cube slice.
            cube_sky.var[:, ~skySpx] = 0  # Discard central spaxels
            if not chi2fit:
                cube_sky.var[cube_sky.var > 0] = 1  # Least-square
            model_sky = pySNIFS_fit.model(
                data=cube_sky, func=['poly2D;%d' % skyDeg],
                param=[[skyLev] + [0.] * (npar_sky - 1)],
                bounds=[[[0, None]] +
                        [[None, None]] * (npar_sky - 1)])
            model_sky.fit()
            skyLev = model_sky.evalfit().squeeze()  # Structured bkgnd estimate

        # Rough guess parameters for the current slice
        medstar = median_filter(cube_star.data[0], 3) - skyLev  # (nspx,)
        imax = medstar.max()                    # Intensity
        if posPrior is not None:                # Use prior on position
            # Note the prior on position (formally at ADR reference wavelength
            # lmid) is not differentially refracted to current wavelength
            xc, yc = posPrior
        elif (xc, yc) == (None, None) or (xc, yc) == (0, 0):
            # No prior nor previous estimate: use flux-weighted centroid on
            # central part
            xc = N.average(cube.x[~skySpx], weights=medstar[~skySpx])
            yc = N.average(cube.y[~skySpx], weights=medstar[~skySpx])
            if not (-7 + nsky < xc < 7 - nsky and -7 + nsky < yc < 7 - nsky):
                xc, yc = 0., 0.

        cube_sky.data -= skyLev         # Subtract background level
        if chi2fit:
            cube_sky.var = cube.var[i, N.newaxis]  # Reset to cube.var for chi2
        else:
            cube_sky.var = None                    # Least-square

        # Guess parameters from 2D-Gaussian + polynomial background fit
        model_gauss = pySNIFS_fit.model(data=cube_sky,
                                        func=['gaus2D', 'poly2D;0'],
                                        param=[[xc, yc, 1, 1, imax], [0]],
                                        bounds=[[[-7, +7]] * 2 +    # xc,yc
                                                [[0.4, 4]] * 2 +    # sx,sy
                                                # intensity
                                                [[0, 5 * imax]],
                                                [[None, None]]])  # background
        model_gauss.minimize(verbose=(verbosity >= 3), tol=1e-4)
        print_msg(model_gauss.facts(params=verbosity >= 3,
                                    names=['xc', 'yc', 'sx', 'sy', 'I', 'B']),
                  2, verbosity)

        if model_gauss.success:
            xc, yc = model_gauss.fitpar[:2]  # Update centroid position
            alpha = max(N.hypot(*model_gauss.fitpar[2:4]), 1.)
        else:
            print "WARNING: gaussian fit failed (status=%d: %s) " % \
                (model_gauss.status, model_gauss.res.message)
            if alpha is None:
                alpha = 2.4             # Educated guess from median seeing

        # Hyper-term on alpha from seeing and position prior
        hyper = {}
        if scalePriors:
            hterm = Hyper_PSF2D_PL(cube.lbda[i], seeingPrior, airmass, channel,
                                   position=posPrior, scale=scalePriors)
            print_msg(str(hterm), 2, verbosity)
            hyper = {psf_fn.name: hterm}
            if seeingPrior:
                alpha = hterm.predict_alpha(cube.lbda[i])
            if posPrior:
                xc, yc = posPrior

        # Filling in the guess parameter arrays (px) and bounds arrays (bx)
        p1 = [0., 0., xc, yc, 0., 1., alpha, imax]  # psf parameters
        b1 = [[0, 0],                   # delta (unfitted)
              [0, 0],                   # theta (unfitted)
              [-10, 10],                # xc
              [-10, 10],                # yc
              [-0.6, +0.6],             # xy parameter
              [0.2, 5],                 # Ellipticity parameter > 0
              [0.1, 15],                # alpha > 0
              [0, None]]                # Intensity > 0

        # alphaDeg & ellDeg set to 0 for meta-slice fits
        func = ['%s;%f,%f,%f,%f' %
                (psf_fn.name, cube_star.spxSize, cube_star.lbda[0], 0, 0)]
        param = [p1]
        bounds = [b1]

        # Background initial guess
        if skyDeg >= 0:
            # Use estimate from prev. polynomial fit
            if skyDeg:
                p2 = list(model_sky.fitpar)
            else:                       # Guess: Background=constant (>0)
                p2 = [skyLev]
            b2 = [[0, None]] + [[None, None]] * (npar_sky - 1)
            func += ['poly2D;%d' % skyDeg]
            param += [p2]
            bounds += [b2]
        else:                           # No background
            p2 = []

        # print_msg("  Initial guess: %s" % (p1+p2), 2, verbosity)

        # Chi2 vs. Least-square fit
        if not chi2fit:
            cube_star.var = None  # Will be handled by pySNIFS_fit.model

        # Instantiate the model class and fit current slice
        model_star = pySNIFS_fit.model(data=cube_star, func=func,
                                       param=param, bounds=bounds,
                                       myfunc={psf_fn.name: psf_fn},
                                       hyper=hyper)

        if verbosity >= 4:
            print "Gradient checks:"    # Includes hyper-term if any
            model_star.check_grad()

        model_star.minimize(verbose=(verbosity >= 2), tol=1e-6,
                            options={'maxiter': 400})

        print_msg(model_star.facts(params=(verbosity >= 2), names=parnames),
                  1, verbosity)
        if scalePriors:
            print_msg("  Hyper-term: h=%f" % hterm.comp(model_star.fitpar),
                      2, verbosity)

        # Restore true chi2 (not reduced one), ie. chi2 =
        # ((cube_star.data-model_star.evalfit())**2/cube_star.var).sum()
        # For least-square fitting, this actually corresponds to
        # RSS=residual sum of squares
        model_star.khi2 *= model_star.dof

        # Check fit results
        if not model_star.success:      # Fit failure
            pass
        elif not 0.2 < model_star.fitpar[5] < 5:
            model_star.success = False
            model_star.status = -1
            model_star.res.message = "ellipticity is invalid (%.2f)" % \
                                     model_star.fitpar[5]
        elif not 0.1 < model_star.fitpar[6] < 15:
            model_star.success = False
            model_star.status = -2
            model_star.res.message = "alpha is invalid (%.2f)" % \
                                     model_star.fitpar[6]
        elif not model_star.fitpar[7] > 0:
            model_star.success = False
            model_star.status = -3
            model_star.res.message = "intensity is null"
        elif not (abs(model_star.fitpar[2]) < 9 and
                  abs(model_star.fitpar[3]) < 9):
            model_star.success = False
            model_star.status = -3
            model_star.res.message = "source is outside FoV (%.2f,%.2f)" % \
                                     (model_star.fitpar[2], model_star.fitpar[3])

        # Error computation and metaslice clipping
        if model_star.success:
            cov = model_star.param_cov()
            diag = cov.diagonal()
            if (diag >= 0).all():
                dpar = N.sqrt(diag)
            else:                       # Some negative diagonal elements!
                model_star.success = False
                model_star.status = -4
                model_star.res.message = "negative covariance diagonal elements"

        if not model_star.success:      # Set error to 0 if status
            print "WARNING: metaslice #%d, status=%d: %s" % \
                  (i + 1, model_star.status, model_star.res.message,)
            model_star.khi2 *= -1       # To be discarded
            dpar = N.zeros(len(dparams.T))
        else:
            xc, yc = model_star.fitpar[2:4]  # Update centroid position

        # Storing the result of the current slice parameters
        params[i] = model_star.fitpar
        dparams[i] = dpar
        chi2s[i] = model_star.khi2

    return params, chi2s, dparams

# Point-source extraction ==============================


def extract_specs(cube, psf, skyDeg=0,
                  method='psf', radius=5., chi2fit=True, verbosity=0):
    """
    Extract object and sky spectra from *cube* using PSF -- described
    by *psf*=(psf_fn,psf_ctes,psf_param) -- in presence of sky
    (polynomial degree *skyDeg*) using *method* ('psf':
    PSF-photometry, 'aperture': aperture photometry, or
    'optimal'). For aperture related methods, *radius* gives aperture
    radius in arcsec.

    Returns (lbda,sigspecs,varspecs) where sigspecs and varspecs are
    (nslice,npar+1).
    """

    assert method in ('psf', 'aperture', 'subaperture', 'optimal'), \
        "Unknown extraction method '%s'" % method
    assert skyDeg >= -1, \
        "skyDeg=%d is invalid (should be >=-1)" % skyDeg

    if (N.isnan(cube.var).any()):
        print "WARNING: discarding NaN variances in extract_specs"
        cube.var[N.isnan(cube.var)] = 0
    if (cube.var > 1e20).any():
        print "WARNING: discarding infinite variances in extract_specs"
        cube.var[cube.var > 1e20] = 0
    if (cube.var < 0).any():              # There should be none anymore
        print "WARNING: discarding negative variances in extract_specs"
        cube.var[cube.var < 0] = 0

    psf_fn, psf_ctes, psf_param = psf   # Unpack PSF description

    # The PSF parameters are only the shape parameters. We arbitrarily
    # set the intensity of each slice to 1.
    param = N.concatenate((psf_param, N.ones(cube.nslice)))

    # General linear least-squares fit: data = I*PSF + sky [ + a*x + b*y + ...]
    # See Numerical Recipes (2nd ed.), sect.15.4

    spxSize = psf_ctes[0]               # Spaxel size [arcsec]
    cube.x = cube.i - 7                # x in spaxel
    cube.y = cube.j - 7                # y in spaxel
    model = psf_fn(psf_ctes, cube)
    psf = model.comp(param, normed=True)  # nslice,nlens

    npar_sky = (skyDeg + 1) * (skyDeg + 2) / 2  # Nb param. in polynomial bkgnd

    # Basis function matrix: BF (nslice,nlens,npar+1) (so-called X in NR)
    BF = N.zeros((cube.nslice, cube.nlens, npar_sky + 1), 'd')
    BF[:, :, 0] = psf                     # Intensity
    if npar_sky:                        # =0 when no background (skyDeg<=-1)
        BF[:, :, 1] = 1                   # Constant background
        n = 2
        for d in xrange(1, skyDeg + 1):
            for j in xrange(d + 1):
                # Background polynomials as function of spaxel (centered)
                # position [spx]
                BF[:, :, n] = cube.x ** (d - j) * cube.y ** j
                n += 1                  # Finally: n = npar_sky + 1

    # Chi2 (variance-weighted) vs. Least-square (unweighted) fit
    # *Note* that weight is actually 1/sqrt(var) (see NR)
    if chi2fit:
        weight = N.where(cube.var > 0, 1 / N.sqrt(cube.var), 0)  # nslice,nlens
    else:
        weight = N.where(cube.var > 0, 1, 0)  # nslice,nlens

    # Design matrix (basis functions normalized by std errors)
    A = BF * weight[..., N.newaxis]      # nslice,nlens,npar+1
    b = weight * cube.data                # nslice,nlens

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

    # Alpha = N.einsum('...jk,...jl',A,A)              # ~x2 slower
    Alpha = N.array([N.dot(aa.T, aa) for aa in A])  # nslice,npar+1,npar+1
    #Beta = N.einsum('...jk,...j',A,b)
    Beta = N.array([N.dot(aa.T, bb) for aa, bb in zip(A, b)])  # nslice,npar+1
    try:
        Cov = N.array([SL.pinv2(aa) for aa in Alpha])  # ns,np+1,np+1
    except SL.LinAlgError:
        raise SL.LinAlgError("Singular matrix during spectrum extraction")
    # sigspecs & varspecs = nslice x [Star,Sky,[slope_x...]]
    sigspecs = N.array([N.dot(cc, bb)
                        for cc, bb in zip(Cov, Beta)])  # nslice,npar+1
    varspecs = N.array([N.diag(cc) for cc in Cov])  # nslice,npar+1

    # Compute the least-square variance using the chi2-case method
    # (errors are meaningless in pure least-square case)
    if not chi2fit:
        weight = N.where(cube.var > 0, 1 / N.sqrt(cube.var), 0)
        A = BF * weight[..., N.newaxis]
        Alpha = N.array([N.dot(aa.T, aa) for aa in A])
        try:
            Cov = N.array([SL.pinv2(aa) for aa in Alpha])
        except SL.LinAlgError:
            raise SL.LinAlgError("Singular matrix "
                                 "during variance extraction")
        varspecs = N.array([N.diag(cc) for cc in Cov])

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

    if skyDeg == 0:
        negSky = sigspecs[:, 1] < 0   # Test for presence of negative sky
        if negSky.any():  # and 'long' not in psf_fn.name.lower():
            print "WARNING: %d slices w/ sky<0 in extract_specs" % \
                  (len(negSky.nonzero()[0]))
            print_msg(str(cube.lbda[negSky]), 3, verbosity)
        # if 'short' in psf_fn.name:
        if False:
            # For slices w/ sky<0, fit only PSF without background
            Alpha = N.array([N.dot(aa, aa) for aa in A[negSky, :, 0]])
            Beta = N.array([N.dot(aa, bb)
                            for aa, bb in zip(A[negSky, :, 0], b[negSky])])
            Cov = 1 / Alpha
            sigspecs[negSky, 0] = Cov * Beta   # Linear fit without sky
            sigspecs[negSky, 1] = 0          # Set sky to null
            varspecs[negSky, 0] = Cov
            varspecs[negSky, 1] = 0

    if method == 'psf':
        return cube.lbda, sigspecs, varspecs  # PSF extraction

    # Reconstruct background and subtract it from cube
    bkgnd = N.zeros_like(cube.data)
    var_bkgnd = N.zeros_like(cube.var)
    if npar_sky:
        for d in xrange(1, npar_sky + 1):  # Loop over sky components
            bkgnd += (BF[:, :, d].T * sigspecs[:, d]).T
            var_bkgnd += (BF[:, :, d].T ** 2 * varspecs[:, d]).T
    subData = cube.data - bkgnd         # Bkgnd subtraction (nslice,nlens)
    subVar = cube.var.copy()
    good = cube.var > 0
    subVar[good] += var_bkgnd[good]     # Variance of bkgnd-sub. signal

    # Replace invalid data (var=0) by model PSF = Intensity*PSF
    if not good.all():
        print_msg("Replacing %d vx with modeled signal" %
                  len((~good).nonzero()[0]), 1, verbosity)
        subData[~good] = (sigspecs[:, 0] * psf.T).T[~good]

    # Plain summation over aperture

    # Aperture radius in spaxels
    aperRad = radius / spxSize
    print_msg("Aperture radius: %.2f arcsec = %.2f spx" % (radius, aperRad),
              1, verbosity)
    # Aperture center after ADR offset from lmid [spx] (nslice,)
    x0 = psf_param[2] + psf_param[0] * \
        N.cos(psf_param[1]) * model.ADRscale[:, 0]
    y0 = psf_param[3] - psf_param[0] * \
        N.sin(psf_param[1]) * model.ADRscale[:, 0]
    # Radial distance from center [spx] (nslice,nlens)
    r = N.hypot((model.x.T - x0).T, (model.y.T - y0).T)
    # Circular aperture (nslice,nlens)
    # Use r<aperRad[:,N.newaxis] if radius is a (nslice,) vec.
    frac = (r < aperRad).astype('float')

    if method == 'subaperture':
        # Fractions accounting for subspaxels (a bit slow)
        newfrac = subaperture(x0, y0, aperRad, 4)
        # Remove bad spaxels since subaperture returns the full spaxel grid
        w = (~N.isnan(cube.slice2d(0).ravel())).nonzero()[0]
        frac = newfrac[:, w]

    # Check if aperture hits the FoV edges
    hit = ((x0 - aperRad) < -7.5) | ((x0 + aperRad) > 7.5) | \
          ((y0 - aperRad) < -7.5) | ((y0 + aperRad) > 7.5)
    if hit.any():
        # Find the closest edge
        ld = (x0 - aperRad + 7.5).min()  # Dist. to left edge (<0 if outside)
        rd = -(x0 + aperRad - 7.5).max()  # Dist. to right edge
        bd = (y0 - aperRad + 7.5).min()  # Dist. to bottom edge
        td = -(y0 + aperRad - 7.5).max()  # Dist. to top edge
        cd = -min(ld, rd, bd, td)          # Should be positive
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
        nw = 15 + 2 * ns                  # New FoV size in spaxels
        mid = (7 + ns)                  # FoV center
        extRange = N.arange(nw) - mid
        extx, exty = N.meshgrid(extRange[::-1], extRange)  # nw,nw
        extnlens = extx.size                 # = nlens' = nw**2
        print_msg("  Extend FoV by %d spx: nlens=%d -> %d" %
                  (ns, model.nlens, extnlens), 1, verbosity)

        # Compute PSF on extended range (nslice,extnlens)
        # Extended model
        extModel = psf_fn(psf_ctes, cube, coords=(extx, exty))
        extPsf = extModel.comp(param, normed=True)  # nslice,extnlens

        # Embed background-subtracted data in extended model PSF
        origData = subData.copy()
        origVar = subVar.copy()
        # Extended model, nslice,extnlens
        subData = (sigspecs[:, 0] * extPsf.T).T
        subVar = N.zeros((extModel.nslice, extModel.nlens))
        for i in xrange(model.nlens):
            # Embeb original spx i in extended model array by finding
            # corresponding index j in new array
            j, = ((extModel.x[0] == model.x[0, i]) &
                  (extModel.y[0] == model.y[0, i])).nonzero()
            subData[:, j[0]] = origData[:, i]
            subVar[:, j[0]] = origVar[:, i]

        r = N.hypot((extModel.x.T - x0).T, (extModel.y.T - y0).T)
        frac = (r < aperRad).astype('float')

    if method.endswith('aperture'):
        # Replace signal and variance estimates from plain summation
        sigspecs[:, 0] = (frac * subData).sum(axis=1)
        varspecs[:, 0] = (frac ** 2 * subVar).sum(axis=1)

        return cube.lbda, sigspecs, varspecs       # [Sub]Aperture extraction

    if method == 'optimal':

        from scipy.ndimage.filters import median_filter

        # Model signal = Intensity*PSF + bkgnd
        modsig = (sigspecs[:, 0] * psf.T).T + bkgnd  # nslice,nlens

        # One has to have a model of the variance. This can be estimated from
        # a simple 'photon noise + RoN' model on each slice: signal ~ alpha*N
        # (alpha = 1/flat-field coeff and N = photon counts) and variance ~ (N
        # + RoN**2) * alpha**2 = (signal/alpha + RoN**2) * alpha**2 =
        # alpha*signal + beta. This model disregards spatial component of
        # flat-field, which is supposed to be constant on FoV.

        # Model variance = alpha*Signal + beta
        coeffs = N.array([polyfit_clip(modsig[s], cube.var[s], 1, clip=5)
                          for s in xrange(cube.nslice)])
        coeffs = median_filter(coeffs, (5, 1))  # A bit of smoothing...
        modvar = N.array([N.polyval(coeffs[s], modsig[s])
                          for s in xrange(cube.nslice)])  # nslice,nlens

        # Optimal weighting
        norm = (frac * psf).sum(axis=1)  # PSF norm, nslice
        npsf = (psf.T / norm).T         # nslice,nlens
        weight = frac * npsf / modvar   # Unormalized weights, nslice,nlens
        norm = (weight * npsf).sum(axis=1)  # Weight norm, nslice
        weight = (weight.T / norm).T    # Normalized weights, nslice,nlens

        # Replace signal and variance estimates from optimal summation
        sigspecs[:, 0] = (weight * subData).sum(axis=1)
        varspecs[:, 0] = (weight ** 2 * subVar).sum(axis=1)

        return cube.lbda, sigspecs, varspecs       # Optimal extraction

# Resampling ========================================================


def subaperture(xc, yc, rc, f=0, nspaxel=15):
    """
    Compute aperture fraction for each spaxel with resampling

    :param xc: aperture X center
    :param yc: aperture Y center
    :param rc: aperture radius
    :param f: resampling factor (e.g. 3 for 2**3-resampling)
    :param nspaxel: spaxel grid side
    :return: spaxel flux fraction on original 15x15 grid
    """

    from ToolBox.Arrays import rebin

    # Resample spaxel center positions, originally [-7:7]
    f = 2 ** f
    epsilon = 0.5 / f
    border = nspaxel / 2.
    r = N.linspace(-border + epsilon, border - epsilon, nspaxel * f)

    x, y = N.meshgrid(r, r)        # (x,y) positions of resampled array
    frac = N.ones(x.shape) / f ** 2  # Spaxel fraction

    xc = N.atleast_1d(xc)
    yc = N.atleast_1d(yc)
    assert xc.shape == yc.shape

    rc = N.atleast_1d(rc)
    if len(rc) == 1:              # One single radius?
        rc = N.repeat(rc, xc.shape)

    out = []
    # This loop could possibly be achieved with some higher order matrix
    for i, j, k in zip(xc, yc, rc):
        fr = frac.copy()
        fr[N.hypot(x - i, y - j) > k] = 0.  # subspaxels outside circle
        # Resample back to original size and sum
        out.append(rebin(fr, f).ravel())

    return N.array(out)

# Header information access utilities ===============================


def read_PT(hdr, MK_pressure=616., MK_temp=2.):
    """
    Read pressure [mbar] and temperature [C] from hdr (or use default
    Mauna-Kea values), and check value consistency.
    """

    if hdr is None:
        return MK_pressure, MK_temp

    pressure = hdr.get('PRESSURE', N.nan)
    if not 550 < pressure < 650:        # Non-std pressure
        print "WARNING: non-std pressure (%.0f mbar) updated to %.0f mbar" % \
              (pressure, MK_pressure)
        if isinstance(hdr, dict):       # pySNIFS.SNIFS_cube.e3d_data_header
            hdr['PRESSURE'] = MK_pressure
        else:                           # True pyfits header, add comment
            hdr['PRESSURE'] = (MK_pressure, "Default MK pressure [mbar]")
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

    return pressure, temp


def read_psf(hdr):
    """Return PSF class as read (or guessed) from header."""

    assert hdr['ES_METH'] == 'psf', \
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
        psfname, psfmodel = psfname.split(', ')  # "name, model"
    except ValueError:
        # Chromatic PSF: 'short|long blue|red'
        if len(psfname.split()) == 2:
            psfmodel = 'chromatic'
        else:                           # Classic PSF: 'short|long'
            psfmodel = 'classic'

    # Convert PSF name (e.g. 'short red') to PSF class name
    # ('ShortRed_ExposurePSF')
    fnname = ''.join(map(str.capitalize, psfname.split())) + '_ExposurePSF'

    psffn = eval(fnname)
    if psfmodel.endswith('powerlaw'):
        psffn.model = psffn.model + '-powerlaw'

    try:
        subsampling = hdr['ES_SUB']
    except KeyError:
        subsampling = 1
    psffn.subsampling = subsampling

    print "PSF name/model: %s/%s [%s], sub x%d" % \
          (psfname, psfmodel, fnname, subsampling)

    return psffn


def read_psf_ctes(hdr):
    """Read PSF constants [lmid,alphaDeg,ellDeg] from header."""

    lmin = hdr['ES_LMIN']
    lmax = hdr['ES_LMAX']
    lmid = (lmin + lmax) / 2.

    # Count up alpha/ell coefficients (ES_Ann/ES_Enn) to get the
    # polynomial degrees
    countKeys = lambda regexp: \
        len([k for k in hdr.keys() if re.match(regexp, k)])

    adeg = countKeys('ES_A\d+$') - 1
    edeg = countKeys('ES_E\d+$') - 1
    print "PSF constants: lMid=%.2f A, alphaDeg=%d, ellDeg=%d" % \
          (lmid, adeg, edeg)

    return [lmid, adeg, edeg]


def read_psf_param(hdr):
    """
    Read (7+ellDeg+alphaDeg) PSF parameters from header:
    delta,theta,xc,yc,xy,e0,...en,a0,...an.
    """

    # Chromatic expansion coefficients
    c_ell = [v for k, v in hdr.items() if re.match('ES_E\d+$', k)]
    c_alp = [v for k, v in hdr.items() if re.match('ES_A\d+$', k)]

    lmin = hdr['ES_LMIN']
    lmax = hdr['ES_LMAX']
    lmid = (lmin + lmax) / 2.       # Middle wavelength [A]
    lref = hdr['ES_LREF']       # Reference wavelength [A]

    # Convert public polynomial coeffs from lr~ = lambda/LbdaRef - 1 =
    # a+b*lr back to internal lr = (2*lambda -
    # (lmin+lmax))/(lmax-lmin)
    a = (lmin + lmax) / (2. * lref) - 1
    b = (lmax - lmin) / (2. * lref)
    ecoeffs = polyConvert(c_ell, trans=(a, b), backward=True).tolist()
    if 'powerlaw' not in hdr['ES_PSF']:
        acoeffs = polyConvert(c_alp, trans=(a, b), backward=True).tolist()
    else:                       # Not needed for powerlaw expansion
        acoeffs = c_alp

    xref = hdr['ES_XC']  # Reference position [spx] at ref. wavelength
    yref = hdr['ES_YC']
    try:
        xy = hdr['ES_XY']       # xy parameter
    except KeyError:
        xy = hdr['ES_PA']       # Old name

    # This reproduces exactly the PSF parameters used by
    # extract_specs(full_cube...)
    pressure, temp = read_PT(hdr)
    airmass = hdr['ES_AIRM']    # Effective airmass
    parang = hdr['ES_PARAN']    # Effective parallactic angle [deg]
    adr = TA.ADR(pressure, temp, lref=lmid, airmass=airmass, parangle=parang)
    xmid, ymid = adr.refract(
        xref, yref, lref, unit=SpxSize, backward=True)  # [spx]

    print "PSF parameters: airmass=%.3f, parangle=%.1f deg, " \
          "refpos=%.2fx%.2f spx @%.2f A" % (airmass, parang, xmid, ymid, lmid)

    return [adr.delta, adr.theta, xmid, ymid, xy] + ecoeffs + acoeffs


def estimate_zdpar(inhdr):
    """
    Estimate zenithal distance [deg] and parallactic angle [deg] from
    header.
    """

    from ToolBox.Astro import Coords
    ha, dec = Coords.altaz2hadec(inhdr['ALTITUDE'], inhdr['AZIMUTH'],
                                 phi=inhdr['LATITUDE'], deg=True)
    zd, parangle = Coords.hadec2zdpar(ha, dec,
                                      phi=inhdr['LATITUDE'], deg=True)

    return zd, parangle          # [deg]


def read_DDTpos(inhdr):
    """
    Read reference wavelength and DDT-estimated position from DDTLREF
    and DDT[X|Y]P keywords. Will raise KeyError if keywords are not
    available.
    """

    try:
        lddt = inhdr['DDTLREF']  # Ref. wavelength [A]
        xddt = inhdr['DDTXP']    # Predicted position [spx]
        yddt = inhdr['DDTYP']
    except KeyError as err:
        raise KeyError("File has no DDT-related keywords (%s)" % err)

    # Some sanity check
    if not (abs(xddt) < 7 and abs(yddt) < 7):
        raise KeyError(
            "Invalid DDT position: %.2f x %.2f is outside FoV" % (xddt, yddt))

    return lddt, xddt, yddt

# Polynomial utilities ======================================================


def polyEval(coeffs, x):
    """
    Evaluate polynom sum_i ci*x**i on x. It uses 'natural' convention
    for polynomial coeffs: [c0,c1...,cn] (opposite to N.polyfit).
    """

    if N.isscalar(x):
        y = 0                           # Faster on scalar
        for i, c in enumerate(coeffs):
            # Incremental computation of x**i is only slightly faster
            y += c * x ** i
    else:                               # Faster on arrays
        y = N.polyval(coeffs[::-1], x)  # Beware coeffs order!

    return y


def polyConvMatrix(n, trans=(0, 1)):
    """
    Return the upper triangular matrix (i,k) * b**k * a**(i-k), that
    converts polynomial coeffs for x~:=a+b*x (P~ = a0~ + a1~*x~ +
    a2~*x~**2 + ...) in polynomial coeffs for x (P = a0 + a1*x +
    a2*x**2 + ...). Therefore, (a,b)=(0,1) gives identity.
    """

    from scipy.misc import comb
    a, b = trans
    m = N.zeros((n, n), dtype='d')
    for r in range(n):
        for c in range(r, n):
            m[r, c] = comb(c, r) * b ** r * a ** (c - r)
    return m


def polyConvert(coeffs, trans=(0, 1), backward=False):
    """
    Converts polynomial coeffs for x (P = a0 + a1*x + a2*x**2 + ...) in
    polynomial coeffs for x~:=a+b*x (P~ = a0~ + a1~*x~ + a2~*x~**2 +
    ...). Therefore, (a,b)=(0,1) makes nothing. If backward, makes the
    opposite transformation.

    Note: backward transformation could be done using more general
    polynomial composition `polyval`, but forward transformation is a
    long standing issue in the general case (look for functional
    decomposition of univariate polynomial).
    """

    a, b = trans
    if not backward:
        a = -float(a) / float(b)
        b = 1 / float(b)
    return N.dot(polyConvMatrix(len(coeffs), (a, b)), coeffs)


def polyfit_clip(x, y, deg, clip=3, nitermax=10):
    """
    Least squares polynomial fit with sigma-clipping (if
    clip>0). Returns polynomial coeffs w/ same convention as
    N.polyfit: [cn,...,c1,c0].
    """

    good = N.ones(y.shape, dtype='bool')
    niter = 0
    while True:
        niter += 1
        coeffs = N.polyfit(x[good], y[good], deg)
        old = good
        if clip:
            dy = N.polyval(coeffs, x) - y
            good = N.absolute(dy) < clip * N.std(dy)
        if (good == old).all():
            break     # No more changes, stop there
        if niter > nitermax:            # Max. # of iter, stop there
            print "polyfit_clip reached max. # of iterations: " \
                  "deg=%d, clip=%.2f x %f, %d px removed" % \
                  (deg, clip, N.std(dy), len((~old).nonzero()[0]))
            break
        if y[good].size <= deg + 1:
            raise ValueError("polyfit_clip: Not enough points left (%d) "
                             "for degree %d" % (y[good].size, deg))
    return coeffs


def chebNorm(x, xmin, xmax):
    """Normalization [xmin,xmax] to [-1,1]"""

    if xmin != xmax:
        return (2 * x - (xmax + xmin)) / (xmax - xmin)
    elif x == xmin:
        return N.zeros_like(x)
    else:
        raise ValueError("Invalid Chebychev normalization.")


def chebEval(pars, nx, chebpolys=[]):
    """
    Orthogonal Chebychev polynomial expansion, x should be already
    normalized in [-1,1].
    """

    from scipy.special import chebyu

    if len(chebpolys) < len(pars):
        print "Initializing Chebychev polynomials up to order %d" % len(pars)
        chebpolys[:] = [chebyu(i) for i in range(len(pars))]

    return N.sum([par * cheb(nx) for par, cheb in zip(pars, chebpolys)], axis=0)


def powerLawEval(coeffs, x):
    """
    Evaluate (curved) power-law: coeffs[-1] * x**(coeffs[-2] +
    coeffs[-3]*(x-1) + ...)

    Note that f(1) = pars[-1] = alpha(lref) with x = lbda/lref.
    """

    return coeffs[-1] * x ** N.polyval(coeffs[:-1], x - 1)


def powerLawJac(coeffs, x):

    ncoeffs = len(coeffs)                          # M
    jac = N.empty((ncoeffs, len(x)), dtype=x.dtype)  # M×N
    jac[-1] = x ** N.polyval(coeffs[:-1], x - 1)       # df/dcoeffs[-1]
    jac[-2] = coeffs[-1] * jac[-1] * N.log(x)        # df/dcoeffs[-2]
    for i in range(-3, -ncoeffs - 1, -1):
        jac[i] = jac[i + 1] * (x - 1)

    return jac                          # M×N


def powerLawFit(x, y, deg=2, guess=None):

    import ToolBox.Optimizer as TO

    if guess is None:
        guess = [0.] * (deg - 1) + [-1., 2.]
    else:
        assert len(guess) == (deg + 1)

    model = TO.Model(powerLawEval, jac=powerLawJac)
    data = TO.DataSet(y, x=x)
    fit = TO.Fitter(model, data)
    lsqPars, msg = SO.leastsq(fit.residuals, guess, args=(x,))

    if msg <= 4:
        return lsqPars
    else:
        raise ValueError("powerLawFit did not converge")

# Ellipse utilities ==============================


def quadEllipse(a, b, c, d, f, g):
    """
    Ellipse elements (center, semi-axes and PA) from the general
    quadratic curve a*x2 + 2*b*x*y + c*y2 + 2*d*x + 2*f*y + g = 0.

    http://mathworld.wolfram.com/Ellipse.html
    """

    D = N.linalg.det([[a, b, d], [b, c, f], [d, f, g]])
    J = N.linalg.det([[a, b], [b, c]])
    I = a + c
    if not (D != 0 and J > 0 and D / I < 0):
        # raise ValueError("Input quadratic curve does not correspond to "
        #                 "an ellipse: D=%f!=0, J=%f>0, D/I=%f<0" % (D,J,D/I))
        return 0, 0, -1, -1, 0
    elif a == c and b == 0:
        #raise ValueError("Input quadratic curve correspond to a circle")
        pass

    b2mac = b ** 2 - a * c
    # Center of the ellipse
    x0 = (c * d - b * f) / b2mac
    y0 = (a * f - b * d) / b2mac
    # Semi-axes lengthes
    ap = N.sqrt(2 * (a * f ** 2 + c * d ** 2 + g * b ** 2 - 2 * b * d * f - a * c * g) /
                (b2mac * (N.sqrt((a - c) ** 2 + 4 * b ** 2) - (a + c))))
    bp = N.sqrt(2 * (a * f ** 2 + c * d ** 2 + g * b ** 2 - 2 * b * d * f - a * c * g) /
                (b2mac * (-N.sqrt((a - c) ** 2 + 4 * b ** 2) - (a + c))))
    # Position angle
    if b == 0:
        phi = 0
    else:
        phi = N.tan((a - c) / (2 * b)) / 2
    if a > c:
        phi += N.pi / 2

    return x0, y0, ap, bp, phi


def flatAndPA(cy2, c2xy):
    """
    Return flattening q=b/a and position angle PA [deg] for ellipse
    defined by x**2 + cy2*y**2 + 2*c2xy*x*y = 1.
    """

    x0, y0, a, b, phi = quadEllipse(1, c2xy, cy2, 0, 0, -1)
    assert a > 0 and b > 0, "Input equation does not correspond to an ellipse"
    q = b / a                             # Flattening
    pa = phi * TA.RAD2DEG                 # From rad to deg

    return q, pa

# PSF classes ================================================================


class ExposurePSF:

    """
    Empirical PSF-3D function used by the `model` class.

    Note that the so-called `PA` or `xy` parameter is *not* the PA of
    the adjusted ellipse, but half the x*y coefficient. Similarly,
    'ell' is not the ellipticity, but the y**2 coefficient: x2 +
    ell*y2 + 2*xy*x*y + ... = 0.  See `quadEllipse`/`flatAndPA` for
    conversion routines.
    """

    subsampling = 1                     # No subsampling by default

    def __init__(self, psf_ctes, cube, coords=None):
        """
        Initiating the class.

        psf_ctes: Internal parameters (pixel size in cube spatial unit,
                  reference wavelength and polynomial degrees).
        cube:     Input cube. This is a `SNIFS_cube` object.
        coords:   if not None, should be (x,y).
        """
        self.spxSize = psf_ctes[0]      # Spaxel size [arcsec]
        self.lmid = psf_ctes[1]      # Reference wavelength [AA]
        self.alphaDeg = int(psf_ctes[2])  # Alpha polynomial degree
        self.ellDeg = int(psf_ctes[3])  # y**2 (aka 'Ell') polynomial degree

        self.npar_cor = 7 + self.ellDeg + self.alphaDeg  # PSF parameters
        self.npar_ind = 1               # Intensity parameters per slice
        self.nslice = cube.nslice
        self.npar = self.npar_cor + self.npar_ind * self.nslice

        # Name of PSF parameters
        self.parnames = ['delta', 'theta', 'xc', 'yc', 'xy'] + \
                        ['e%d' % i for i in range(self.ellDeg + 1)] + \
                        ['a%d' % i for i in range(self.alphaDeg + 1)] + \
                        ['i%02d' % (i + 1) for i in range(self.nslice)]

        # Spaxel coordinates [spx]
        if coords is None:
            self.nlens = cube.nlens
            # nslice,nlens
            self.x = N.resize(cube.x, (self.nslice, self.nlens))
            self.y = N.resize(cube.y, (self.nslice, self.nlens))
        else:
            x = coords[0].ravel()
            y = coords[1].ravel()
            assert len(x) == len(y), \
                "Incompatible coordinates (%d/%d)" % (len(x), len(y))
            self.nlens = len(x)
            self.x = N.resize(x, (self.nslice, self.nlens))  # nslice,nlens
            self.y = N.resize(y, (self.nslice, self.nlens))
        # nslice,nlens
        self.l = N.resize(cube.lbda, (self.nlens, self.nslice)).T

        if self.nslice > 1:
            self.lmin = cube.lstart
            self.lmax = cube.lend
            self.lrel = chebNorm(self.l, self.lmin, self.lmax)  # From -1 to +1
        else:
            self.lmin, self.lmax = -1, +1
            self.lrel = self.l

        # ADR in spaxels (nslice,nlens)
        if hasattr(cube, 'e3d_data_header'):  # Read from cube if possible
            pressure, temp = read_PT(cube.e3d_data_header)
        else:
            pressure, temp = read_PT(None)   # Get default values for P and T
        self.ADRscale = TA.ADR(
            P=pressure, T=temp, lref=self.lmid).get_scale(self.l) / self.spxSize

        # Sub-sampling grid: decompose the spaxels into n×n sub-spaxels
        eps = N.linspace(-0.5, +0.5, self.subsampling * 2 + 1)[1::2]
        # Offsets from center
        self.subgrid = tuple(itertools.product(eps, eps))

    def comp(self, param, normed=False):
        """
        Compute the function.

        param: Input parameters for the PSF model:

        - param[0:7+n+m]: parameters of the PSF shape
        - param[0,1]: Atmospheric dispersion power and parall. angle [rad]
        - param[2,3]: X,Y position at reference wavelength
        - param[4]: xy parameter
        - param[5:6+n]: Ellipticity param. expansion (n+1: # of coeffs)
        - param[6+n:7+n+m]: Moffat scale alpha expansion (m+1: # of coeffs)
        - param[7+m+n:]: Intensity parameters (one for each slice in the cube)

        normed: Should the function be normalized (integral)
        """

        self.param = N.asarray(param)

        # ADR params
        delta = self.param[0]
        theta = self.param[1]
        xc = self.param[2]           # Position at lmid
        yc = self.param[3]
        # Position at current wavelength
        x0 = xc + delta * N.sin(theta) * self.ADRscale  # nslice,nlens
        y0 = yc - delta * N.cos(theta) * self.ADRscale

        # Other params
        xy = self.param[4]
        ellCoeffs = self.param[5:6 + self.ellDeg]
        alphaCoeffs = self.param[6 + self.ellDeg:self.npar_cor]

        ell = polyEval(ellCoeffs, self.lrel)  # nslice,nlens
        if not self.model.endswith('powerlaw'):
            alpha = polyEval(alphaCoeffs, self.lrel)
        else:
            alpha = powerLawEval(alphaCoeffs, self.l / LbdaRef)

        # PSF model
        if self.model == 'chromatic':  # Includes chromatic correlations
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
        sigma = s0 + s1 * alpha
        beta = b0 + b1 * alpha
        eta = e0 + e1 * alpha

        val = 0.
        for epsx, epsy in self.subgrid:
            # Gaussian + Moffat
            dx = self.x - x0 + epsx     # Center of sub-spaxel
            dy = self.y - y0 + epsy
            # CAUTION: ell & PA are not the true ellipticity and position
            # angle!
            r2 = dx ** 2 + ell * dy ** 2 + 2 * xy * dx * dy
            gaussian = N.exp(-0.5 * r2 / sigma ** 2)
            moffat = (1 + r2 / alpha ** 2) ** (-beta)
            # Function
            val += moffat + eta * gaussian

        val *= self.param[self.npar_cor:, N.newaxis] / self.subsampling ** 2

        # The 3D psf model is not normalized to 1 in integral. The result must
        # be renormalized by (2*eta*sigma**2 + alpha**2/(beta-1)) *
        # N.pi/sqrt(ell)
        if normed:
            val /= N.pi * \
                (2 * eta * sigma ** 2 + alpha ** 2 / (beta - 1)) / N.sqrt(ell)

        return val

    def deriv(self, param):
        """
        Compute the derivative of the function with respect to its parameters.

        param: Input parameters of the polynomial.
               A list numbers (see `SNIFS_psf_3D.comp`).
        """

        self.param = N.asarray(param)

        # ADR params
        delta = self.param[0]
        theta = self.param[1]
        xc = self.param[2]
        yc = self.param[3]
        costheta = N.cos(theta)
        sintheta = N.sin(theta)
        x0 = xc + delta * sintheta * self.ADRscale  # nslice,nlens
        y0 = yc - delta * costheta * self.ADRscale

        # Other params
        xy = self.param[4]
        ellCoeffs = self.param[5:6 + self.ellDeg]
        alphaCoeffs = self.param[6 + self.ellDeg:self.npar_cor]

        ell = polyEval(ellCoeffs, self.lrel)
        if not self.model.endswith('powerlaw'):
            alpha = polyEval(alphaCoeffs, self.lrel)
        else:
            alpha = powerLawEval(alphaCoeffs, self.l / LbdaRef)

        # PSF model
        if self.model == 'chromatic':  # Includes chromatic correlations
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
        sigma = s0 + s1 * alpha
        beta = b0 + b1 * alpha
        eta = e0 + e1 * alpha

        totgrad = N.zeros((self.npar_cor + self.npar_ind,) + self.x.shape, 'd')
        for epsx, epsy in self.subgrid:
            # Gaussian + Moffat
            dx = self.x - x0 + epsx
            dy = self.y - y0 + epsy
            dy2 = dy ** 2
            r2 = dx ** 2 + ell * dy2 + 2 * xy * dx * dy
            sigma2 = sigma ** 2
            gaussian = N.exp(-0.5 * r2 / sigma2)
            alpha2 = alpha ** 2
            ea = 1 + r2 / alpha2
            moffat = ea ** (-beta)

            # Derivatives
            grad = N.zeros(
                (self.npar_cor + self.npar_ind,) + self.x.shape, 'd')
            j1 = eta / sigma2
            j2 = 2 * beta / ea / alpha2
            tmp = gaussian * j1 + moffat * j2
            grad[2] = tmp * (dx + xy * dy)  # dPSF/dxc
            grad[3] = tmp * (ell * dy + xy * dx)  # dPSF/dyc
            grad[0] = self.ADRscale * \
                (sintheta * grad[2] - costheta * grad[3])
            grad[1] = delta * self.ADRscale * \
                (sintheta * grad[3] + costheta * grad[2])
            grad[4] = -tmp * dx * dy        # dPSF/dxy
            for i in xrange(self.ellDeg + 1):  # dPSF/dei
                grad[5 + i] = -tmp / 2 * dy2 * self.lrel ** i
            dalpha = gaussian * ( e1 + s1 * r2 * j1 / sigma ) + \
                moffat * (-b1 * N.log(ea) + r2 * j2 / alpha)  # dPSF/dalpha
            if not self.model.endswith('powerlaw'):
                for i in xrange(self.alphaDeg + 1):  # dPSF/dai, i=<0,alphaDeg>
                    grad[6 + self.ellDeg + i] = dalpha * self.lrel ** i
            else:
                lrel = self.l / LbdaRef
                imax = 6 + self.ellDeg + self.alphaDeg
                grad[imax] = dalpha * \
                    lrel ** N.polyval(alphaCoeffs[:-1], lrel - 1)
                if self.alphaDeg:
                    grad[imax - 1] = grad[imax] * alphaCoeffs[-1] * N.log(lrel)
                    for i in range(imax - 2, imax - self.alphaDeg - 1, -1):
                        # dPSF/dai, i=0..alphaDeg
                        grad[i] = grad[i + 1] * (lrel - 1)
            grad[self.npar_cor] = moffat + eta * gaussian  # dPSF/dI

            totgrad += grad

        totgrad[:self.npar_cor] *= self.param[N.newaxis,
                                              self.npar_cor:, N.newaxis]
        totgrad /= self.subsampling ** 2

        return totgrad

    def _HWHM_fn(self, r, alphaCoeffs, lbda):
        """Half-width at half maximum function (=0 at HWHM)."""

        if not self.model.endswith('powerlaw'):
            alpha = polyEval(alphaCoeffs, chebNorm(lbda, self.lmin, self.lmax))
        else:
            alpha = powerLawEval(alphaCoeffs, lbda / LbdaRef)

        if self.model == 'chromatic':
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
        sigma = s0 + s1 * alpha
        beta = b0 + b1 * alpha
        eta = e0 + e1 * alpha
        gaussian = N.exp(-0.5 * r ** 2 / sigma ** 2)
        moffat = (1 + r ** 2 / alpha ** 2) ** (-beta)

        # PSF=moffat + eta*gaussian, maximum is 1+eta
        return moffat + eta * gaussian - (eta + 1) / 2

    def FWHM(self, param, lbda):
        """Estimate FWHM of PSF at wavelength lbda."""

        alphaCoeffs = param[6 + self.ellDeg:self.npar_cor]
        # Compute FWHM from radial profile
        fwhm = 2 * \
            SO.fsolve(func=self._HWHM_fn, x0=1., args=(alphaCoeffs, lbda))

        # Beware: scipy-0.8.0 fsolve returns a size 1 array
        return N.squeeze(fwhm)  # In spaxels

    @classmethod
    def seeing_powerlaw(cls, lbda, alphaCoeffs):
        """
        Estimate power-law chromatic model seeing FWHM [arcsec] at
        wavelength `lbda` for alpha coefficients `alphaCoeffs`.
        """

        def hwhm(r, alphaCoeffs, lbda):

            alpha = powerLawEval(alphaCoeffs, lbda / LbdaRef)
            sigma = cls.sigma0 + cls.sigma1 * alpha
            beta = cls.beta0 + cls.beta1 * alpha
            eta = cls.eta0 + cls.eta1 * alpha
            gaussian = N.exp(-0.5 * r ** 2 / sigma ** 2)
            moffat = (1 + r ** 2 / alpha ** 2) ** (-beta)
            # PSF=moffat + eta*gaussian, maximum is 1+eta
            return moffat + eta * gaussian - (eta + 1) / 2

        # Compute FWHM from radial profile [spx]
        seeing = 2 * SO.fsolve(func=hwhm, x0=1., args=(alphaCoeffs, lbda))

        return N.squeeze(seeing) * SpxSize  # Spx → arcsec


class Long_ExposurePSF(ExposurePSF):

    """Classic PSF model (achromatic correlations) for long exposures."""

    name = 'long'
    model = 'classic'

    beta0 = 1.685
    beta1 = 0.345
    sigma0 = 0.545
    sigma1 = 0.215
    eta0 = 1.04
    eta1 = 0.00


class Short_ExposurePSF(ExposurePSF):

    """
    Classic PSF model (achromatic correlations) for short exposures.
    """

    name = 'short'
    model = 'classic'

    beta0 = 1.395
    beta1 = 0.415
    sigma0 = 0.56
    sigma1 = 0.2
    eta0 = 0.6
    eta1 = 0.16


class LongBlue_ExposurePSF(ExposurePSF):

    """
    PSF model with chromatic correlations (2nd order Chebychev
    polynomial) for long, blue exposures.
    """

    name = 'long blue'
    model = 'chromatic'
    chebRange = (3399., 5100.)      # Domain of validity of Chebychev expansion

    beta0 = [1.220, 0.016, -0.056]  # b00,b01,b02
    beta1 = [0.590, 0.004, 0.014]  # b10,b11,b12
    sigma0 = [0.710, -0.024, 0.016]  # s00,s01,s02
    sigma1 = [0.119, 0.001, -0.004]  # s10,s11,s12
    eta0 = [0.544, -0.090, 0.039]  # e00,e01,e02
    eta1 = [0.223, 0.060, -0.020]  # e10,e11,e12


class LongRed_ExposurePSF(ExposurePSF):

    """
    PSF model with chromatic correlations (2nd order Chebychev
    polynomial) for long, red exposures.
    """

    name = 'long red'
    model = 'chromatic'
    chebRange = (5318., 9508.)      # Domain of validity of Chebychev expansion

    beta0 = [1.205, -0.100, -0.031]  # b00,b01,b02
    beta1 = [0.578, 0.062, 0.028]  # b10,b11,b12
    sigma0 = [0.596, 0.044, 0.011]  # s00,s01,s02
    sigma1 = [0.173, -0.035, -0.008]  # s10,s11,s12
    eta0 = [1.366, -0.184, -0.126]  # e00,e01,e02
    eta1 = [-0.134, 0.121, 0.054]  # e10,e11,e12


class ShortBlue_ExposurePSF(ExposurePSF):

    """
    PSF model with chromatic correlations (2nd order Chebychev
    polynomial) for short, blue exposures.
    """

    name = 'short blue'
    model = 'chromatic'
    chebRange = (3399., 5100.)      # Domain of validity of Chebychev expansion

    beta0 = [1.355, 0.023, -0.042]  # b00,b01,b02
    beta1 = [0.524, -0.012, 0.020]  # b10,b11,b12
    sigma0 = [0.492, -0.037, 0.000]  # s00,s01,s02
    sigma1 = [0.176, 0.016, 0.000]  # s10,s11,s12
    eta0 = [0.499, 0.080, 0.061]  # e00,e01,e02
    eta1 = [0.316, -0.015, -0.050]  # e10,e11,e12


class ShortRed_ExposurePSF(ExposurePSF):

    """
    PSF model with chromatic correlations (2nd order Chebychev
    polynomial) for short, red exposures.
    """

    name = 'short red'
    model = 'chromatic'
    chebRange = (5318., 9508.)      # Domain of validity of Chebychev expansion

    beta0 = [1.350, -0.030, -0.012]  # b00,b01,b02
    beta1 = [0.496, 0.032, 0.020]  # b10,b11,b12
    sigma0 = [0.405, -0.003, 0.000]  # s00,s01,s02
    sigma1 = [0.212, -0.017, 0.000]  # s10,s11,s12
    eta0 = [0.704, -0.060, 0.044]  # e00,e01,e02
    eta1 = [0.343, 0.113, -0.045]  # e10,e11,e12


class Hyper_PSF3D_PL(object):

    """
    Hyper-term to be added to 3D-PSF fit: priors on ADR parameters,
    alpha power-law chromatic expansion, PSF shape parameters and
    point-source position.
    """

    positionAccuracy = 0.5     # Rather arbitrary position prior accuracy [spx]

    def __init__(self, psf_ctes, inhdr, seeing=None, position=None,
                 scale=1., verbose=False):

        alphaDeg = psf_ctes[2]          # Alpha expansion degree
        ellDeg = psf_ctes[3]            # Ellipticity expansion degree
        if alphaDeg != 2:
            raise NotImplementedError("Hyper-term trained for alphaDeg=2 only")
        if ellDeg != 0:
            raise NotImplementedError("Hyper-term trained for ellDeg=0 only")
        self.alphaSlice = slice(6 + ellDeg, 7 + ellDeg + alphaDeg)

        self.X = inhdr['CHANNEL'][0].upper()  # 'B' or 'R'
        if self.X not in ('B', 'R'):
            raise KeyError("Unknown channel '%s'" % inhdr['CHANNEL'])

        self.scale = scale              # Global hyper-scaling

        # Compute predictions and associated accuracy
        self._predict_ADR(inhdr, verbose=verbose)    # ADR parameters
        self._predict_shape(inhdr, verbose=verbose)  # Shape (xy & y2) params
        self._predict_PL(seeing, verbose=verbose)    # Power-law expansion coeffs
        # Position at ref. wavelength
        self._predict_pos(position, verbose=verbose)

    @classmethod
    def predict_adr_params(cls, inhdr):
        """
        Predict ADR parameters delta and theta [rad] from header `inhdr`
        including standard keywords `AIRMASS`, `PARANG` (parallactic
        angle [deg]), and `CHANNEL`.
        """

        # 0th-order estimates
        delta0 = N.tan(N.arccos(1. / inhdr['AIRMASS']))
        theta0 = inhdr['PARANG'] / TA.RAD2DEG  # Parallactic angle [rad]

        # 1st-order corrections from ad-hoc linear regressions
        sinpar = N.sin(theta0)
        cospar = N.cos(theta0)
        X = inhdr['CHANNEL'][0].upper()  # 'B' or 'R'
        if X == 'B':                      # Blue
            ddelta1 = -0.00734 * sinpar + 0.00766
            dtheta1 = -0.554 * cospar + 3.027  # [deg]
        elif X == 'R':                    # Red
            ddelta1 = +0.04674 * sinpar + 0.00075
            dtheta1 = +3.078 * cospar + 4.447  # [deg]
        else:
            raise KeyError("Unknown channel '%s'" % inhdr['CHANNEL'])

        # Final predictions
        delta = delta0 + ddelta1
        theta = theta0 + dtheta1 / TA.RAD2DEG  # [rad]

        return delta, theta

    @classmethod
    def predict_alpha_coeffs(cls, seeing, channel):
        """
        Predict power-law expansion alpha coefficients from seeing for
        given channel.
        """

        if channel == 'B':
            coeffs = N.array([
                -0.134 * seeing + 0.5720,    # p0
                -0.134 * seeing - 0.0913,    # p1
                +3.474 * seeing - 1.3880])   # p2
        elif channel == 'R':
            coeffs = N.array([
                -0.0777 * seeing + 0.1741,   # p0
                -0.0202 * seeing - 0.3434,   # p1
                +3.4000 * seeing - 1.352])   # p2
        else:
            raise KeyError("Unknown channel '%s'" % channel)

        return coeffs

    @classmethod
    def predict_y2_param(cls, inhdr):
        """Predict shape parameter y2."""

        # Ad-hoc linear regressions
        airmass = inhdr['AIRMASS']
        X = inhdr['CHANNEL'][0].upper()  # 'B' or 'R'
        if X == 'B':                      # Blue
            y2 = -0.323 * airmass + 1.730
        elif X == 'R':                    # Red
            y2 = -0.442 * airmass + 1.934
        else:
            raise KeyError("Unknown channel '%s'" % inhdr['CHANNEL'])

        return y2

    def _predict_ADR(self, inhdr, verbose=False):
        """
        Predict ADR parameters delta,theta and prediction accuracy
        ddelta,dtheta, for use in hyper-term computation. 1st-order
        corrections and model dispersions were obtained from faint
        standard star ad-hoc analysis (`adr.py` and `runaway.py`).
        """

        self.delta, self.theta = self.predict_adr_params(inhdr)

        # Final model dispersion
        if self.X == 'B':                 # Blue
            self.ddelta = 0.0173
            self.dtheta = 1.651         # [deg]
        else:                           # Red
            self.ddelta = 0.0122
            self.dtheta = 1.453         # [deg]
        self.dtheta /= TA.RAD2DEG       # [rad]

        if verbose:
            print "ADR parameter predictions:"
            print "  Header:     δ=% .2f,  θ=%+.2f°" % \
                  (N.tan(N.arccos(1. / inhdr['AIRMASS'])),
                   inhdr['PARANG'] / TA.RAD2DEG)
            print "  Parameters: δ=% .2f,  θ=%+.2f°" % \
                  (self.delta, self.theta * TA.RAD2DEG)
            print "  dParam:    Δδ=% .2f, Δθ=% .2f°" % \
                  (self.ddelta, self.dtheta * TA.RAD2DEG)

    def _predict_PL(self, seeing, verbose=False):
        """
        Predict ADR parameters power-law parameters {p_i} and prediction
        precision matrix cov^{-1}({p_i}), for use in hyper-term
        computation. 1st-order corrections and model dispersions were
        obtained from faint standard star ad-hoc analysis (`adr.py`
        and `runaway.py`).
        """

        if seeing is None:
            self.plpars = None  # No prediction
            return

        # Predict power-law expansion coefficients and precision matrix
        if self.X == 'B':                 # Blue
            self.plpars = self.predict_alpha_coeffs(seeing, self.X)
            self.plicov = N.array(              # Precision matrix = 1/Cov
                [[43.33738708, -66.87684631, -0.23146413],
                 [-66.87684631, 242.87202454,  4.43127346],
                 [-0.231464,     4.43127346, 12.65395737]])
        else:                           # Red
            self.plpars = self.predict_alpha_coeffs(seeing, self.X)
            self.plicov = N.array(              # Precision matrix = 1/Cov
                [[476.81713867,  19.62824821, 23.05086708],
                 [19.62825203, 612.26849365, 11.54866409],
                 [23.05086899,  11.54866314, 11.4956665]])

        if verbose:
            print "Power-law expansion coefficient predictions:"
            print "  Seeing prior: %.2f\"" % seeing
            print "  Parameters: p0=%+.3f  p1=%+.3f  p2=%+.3f" % \
                  tuple(self.plpars)
            print "  ~dParams:  dp0=% .3f dp1=% .3f dp2=% .3f" % \
                  tuple(self.plicov.diagonal() ** -0.5)

    def _predict_shape(self, inhdr, verbose=False):
        """
        Predict shape parameters y2,xy and prediction accuracy dy2,dxy, for
        use in hyper-term computation. 1st-order corrections and model
        dispersions were obtained from faint standard star ad-hoc
        analysis (`runaway.py`).
        """

        self.y2 = self.predict_y2_param(inhdr)
        self.xy = 0.                    # Pure dispersion

        # Final model dispersion
        if self.X == 'B':                 # Blue
            self.dy2 = 0.221
            self.dxy = 0.041
        else:                           # Red
            self.dy2 = 0.269
            self.dxy = 0.050

        if verbose:
            print "Shape parameter predictions:"
            print "  Airmass:       %+.2f" % inhdr['AIRMASS']
            print "  Parameters: y²=% .3f,  xy=% .3f" % (self.y2, self.xy)
            print "  dParam:    Δy²=% .3f, Δxy=% .3f" % (self.dy2, self.dxy)

    def _predict_pos(self, position, verbose=False):
        """
        Predict position (x,y) and prediction accuracy (dx,dy) at
        reference wavelength, for use in hyper-term computation.
        """

        self.position = position             # None or (x,y)
        self.dposition = (
            self.positionAccuracy, self.positionAccuracy)  # [spx]

        if verbose and self.position is not None:
            print "Position predictions:"
            print "  Parameters: x=% .3f,  y=% .3f" % self.position
            print "  dParam:    Δx=% .3f, Δy=% .3f" % self.dposition

    def comp(self, param):
        """
        Input parameters, same as `ExposurePSF.comp`, notably:

        - param[0,1]: ADR power (delta) and parallactic angle (theta[rad])
        - param[2,3]: X,Y position at reference wavelength
        - param[4]: xy parameter
        - param[5:6+n]: Ellipticity param. expansion (n+1: # of coeffs)
        - param[6+n:7+n+m]: Moffat scale alpha expansion (m+1: # of coeffs)
        """

        # Term from ADR parameters
        hadr = ( (param[0] - self.delta) / self.ddelta ) ** 2 + \
               ((param[1] - self.theta) / self.dtheta) ** 2
        # Term from shape parameters
        hsha = ( (param[4] - self.xy) / self.dxy ) ** 2 + \
               ((param[5] - self.y2) / self.dy2) ** 2
        if self.plpars is not None:
            # Term from PL parameters
            dalpha = param[self.alphaSlice] - self.plpars
            # Faster than dalpha.dot(self.plicov).dot(dalpha)
            hpl = N.dot(N.dot(dalpha, self.plicov), dalpha)
        else:
            hpl = 0.
        # Term from position
        if self.position is not None:
            hpos = ( (param[2] - self.position[0]) / self.dposition[0] ) ** 2 + \
                   ((param[3] - self.position[1]) / self.dposition[1]) ** 2
        else:
            hpos = 0.

        return self.scale * (hadr + hsha + hpl + hpos)  # Scalar ()

    def deriv(self, param):

        hjac = N.zeros(len(param))      # Half jacobian

        # ADR parameter jacobian
        hjac[0] = (param[0] - self.delta) / self.ddelta ** 2
        hjac[1] = (param[1] - self.theta) / self.dtheta ** 2
        # Shape parameter jacobian
        hjac[4] = (param[4] - self.xy) / self.dxy ** 2
        hjac[5] = (param[5] - self.y2) / self.dy2 ** 2
        if self.plpars is not None:
            # PL-expansion parameter jacobian
            hjac[self.alphaSlice] = N.dot(
                self.plicov, param[self.alphaSlice] - self.plpars)
        # Position jacobian
        if self.position is not None:
            hjac[2] = (param[2] - self.position[0]) / self.dposition[0] ** 2
            hjac[3] = (param[3] - self.position[1]) / self.dposition[1] ** 2

        return self.scale * 2 * hjac        # (npar,)

    def __str__(self):

        s = "PSF3D_PL hyper-term: hyper-scale=%.2f" % self.scale
        s += "\n  ADR: delta=% 7.2f +/- %.2f" % (self.delta, self.ddelta)
        s += "\n       theta=%+7.2f +/- %.2f deg" % \
             (self.theta * TA.RAD2DEG, self.dtheta * TA.RAD2DEG)
        s += "\n  Shape: xy=% 5.3f +/- %.3f" % (self.xy, self.dxy)
        s += "\n         y2=% 5.3f +/- %.3f" % (self.y2, self.dy2)
        if self.plpars is not None:
            dplpars = self.plicov.diagonal() ** -0.5  # Approximate variance
            s += "\n  PL: p0=%+.3f +/- %.3f" % (self.plpars[0], dplpars[0])
            s += "\n      p1=%+.3f +/- %.3f" % (self.plpars[1], dplpars[1])
            s += "\n      p2=%+.3f +/- %.3f" % (self.plpars[2], dplpars[2])
        if self.position is not None:
            s += "\n  Position: x=% 5.3f +/- %.3f" % \
                 (self.position[0], self.dposition[0])
            s += "\n            y=% 5.3f +/- %.3f" % \
                 (self.position[1], self.dposition[1])

        return s


class Hyper_PSF2D_PL(Hyper_PSF3D_PL):

    """
    Hyper-term to be added to 2D-PSF fit: priors on alpha (seeing), xy
    and y2 shape terms.
    """

    dalpha = 0.15               # Relaxed achromatic accuracy

    def __init__(self, lbda, seeing, airmass, channel,
                 position=None, scale=1., verbose=False):

        # Mimic PSF constantes and input header
        psf_ctes = [None, None, 2, 0]
        inhdr = {'CHANNEL': channel,
                 'AIRMASS': airmass,
                 'PARANG': 0.}

        Hyper_PSF3D_PL.__init__(self, psf_ctes, inhdr, seeing,
                                position=position, scale=scale, verbose=False)

        self.alpha = self.predict_alpha(lbda)

    def predict_alpha(self, lbda):
        """Predict alpha at wavelength `lbda`."""

        if self.plpars is None:
            return None

        alpha = powerLawEval(self.plpars, lbda / LbdaRef)
        # Adjust prediction with linear regression in wavelength
        dalpha = -2.35e-05 * lbda + 0.156  # Up to 0.1 correction

        return alpha - dalpha              # Total prediction

    def comp(self, param):
        """
        Input parameters, notably:

        - param[0,1]: ADR delta and theta (kept fixed to 0)
        - param[2,3]: X,Y position at reference wavelength
        - param[4]: xy parameter
        - param[5]: Ellipticity parameter
        - param[6]: Moffat scale alpha
        - param[7]: Pount-source intensity
        """

        # Terms from xy- and y2-parameters
        h = (((param[4] - self.xy) / self.dxy) ** 2 +
             ((param[5] - self.y2) / self.dy2) ** 2)
        if self.plpars is not None:
            # Term from alpha
            h += ((param[6] - self.alpha) / self.dalpha) ** 2
        if self.position is not None:
            # Term from point-source position
            h += (((param[2] - self.position[0]) / self.dposition[0] ) ** 2 +
                  ((param[3] - self.position[1]) / self.dposition[1]) ** 2)

        return self.scale * h

    def deriv(self, param):

        hjac = N.zeros(len(param))      # Half jacobian
        hjac[4] = (param[4] - self.xy) / self.dxy ** 2
        hjac[5] = (param[5] - self.y2) / self.dy2 ** 2
        if self.plpars is not None:
            hjac[6] = (param[6] - self.alpha) / self.dalpha ** 2
        if self.position is not None:
            hjac[2] = (param[2] - self.position[0]) / self.dposition[0] ** 2
            hjac[3] = (param[3] - self.position[1]) / self.dposition[1] ** 2

        return self.scale * 2 * hjac        # (npar,)

    def __str__(self):

        s = "PSF2D_PL hyper-term: hyper-scale=%.2f" % self.scale
        s += "\n  Pred. xy:   %+.3f +/- %.3f" % (self.xy, self.dxy)
        s += "\n  Pred. y2:   %+.3f +/- %.3f" % (self.y2, self.dy2)
        if self.plpars is not None:
            s += "\n  Pred. alpha:  %.2f +/- %.2f (%.2f\" at %.0f A)" % \
                 (self.alpha, self.dalpha,
                  Long_ExposurePSF.seeing_powerlaw(LbdaRef, self.plpars),
                  LbdaRef)
        if self.position is not None:
            s += "\n  Pred. x:   %+.2f +/- %.2f" % (self.position[0],
                                                    self.dposition[0])
            s += "\n  Pred. y:   %+.2f +/- %.2f" % (self.position[1],
                                                    self.dposition[1])

        return s
