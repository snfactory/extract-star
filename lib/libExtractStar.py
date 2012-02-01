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
import scipy as S
import scipy.special
from scipy.ndimage import filters as F

from ToolBox import Atmosphere as A

def print_msg(str, limit, verb=0):
    """Print message 'str' if verbosity level (opts.verbosity) >= limit."""

    if verb >= limit:
        print str

# Extraction ========================================================

def extract_spec(cube, psf_fn, psf_ctes, psf_param, skyDeg=0,
                 method='psf', radius=5., chi2fit=True, verbosity=0):
    """Extract object and sky spectra from cube according to PSF (described by
    psf_fn, psf_ctes and psf_params) in presence of sky (polynomial degree
    skyDeg) using method ('psf' or 'aperture' or 'optimal'). For aperture
    related methods, radius gives aperture radius in arcsec.

    Returns spec,var where spec and var are (nslice,npar+1)."""

    assert method in ('psf','aperture','subaperture','optimal'), \
           "Extraction method '%s' unrecognized" % method
    assert skyDeg >= -1, \
           "skyDeg=%d is invalid (should be >=-1)" % skyDeg

    if (cube.var>1e20).any():
        print "WARNING: discarding infinite variances in extract_spec"
        cube.var[cube.var>1e20] = 0
    if (cube.var<0).any():              # There should be none anymore
        print "WARNING: discarding negative variances in extract_spec"
        cube.var[cube.var<0] = 0

    # The PSF parameters are only the shape parameters. We set the intensity
    # of each slice to 1.
    param = N.concatenate((psf_param,N.ones(cube.nslice)))

    # Linear least-squares fit: I*PSF + sky [ + a*x + b*y + ...]

    spxSize = psf_ctes[0]               # Spaxel size [arcsec]
    cube.x  = cube.i - 7                # x in spaxel
    cube.y  = cube.j - 7                # y in spaxel
    model = psf_fn(psf_ctes, cube)
    psf   = model.comp(param, normed=True) # nslice,nlens

    npar_sky = (skyDeg+1)*(skyDeg+2)/2  # Nb param. in polynomial bkgnd
    Z = N.zeros((cube.nslice,cube.nlens,npar_sky+1),'d')
    Z[:,:,0] = psf                      # Intensity
    if npar_sky:                        # =0 when no background (skyDeg<=-1)
        Z[:,:,1] = 1                    # Constant background
        n = 2
        for d in xrange(1,skyDeg+1):
            for j in xrange(d+1):
                Z[:,:,n] = cube.x**(d-j) * cube.y**j # Bkgnd polynomials
                n += 1                  # Finally: n = npar_sky + 1

    # Chi2 (weight=1/var) vs. Least-square (weight=1) fit
    if chi2fit:
        weight = N.where(cube.var>0, 1/N.sqrt(cube.var), 0) # nslice,nlens
    else:
        weight = N.where(cube.var>0, 1, 0) # nslice,nlens
    X = (Z.T * weight.T).T              # nslice,nlens,npar+1
    b = weight*cube.data                # nslice,nlens

    # The linear least-squares fit could be done directly using
    # spec = N.array([ N.linalg.lstsq(xx,bb)[0] for xx,bb in zip(X,b) ])
    # but A is needed anyway to compute covariance matrix C=1/A.
    # Furthermore, linear resolution
    # [ N.linalg.solve(aa,bb) for aa,bb in zip(A,B) ]
    # can be replace by faster (~x10) matrix product
    # [ N.dot(cc,bb) for cc,bb in zip(C,B) ]
    # since C=1/A is readily available.
    # See Numerical Recipes (2nd ed.), sect.15.4

    # OTOH, "Solving Ax = b: inverse vs cholesky factorization" thread
    # (http://thread.gmane.org/gmane.comp.python.numeric.general/41365)
    # advocates to never invert a matrix directly.

    A = N.array([ N.dot(xx.T, xx) for xx in X ]) # nslice,npar+1,npar+1
    B = N.array([ N.dot(xx.T, bb) for xx,bb in zip(X,b) ]) # nslice,npar+1
    try:
        C = N.array([ N.linalg.pinv(aa) for aa in A ]) # nslice,npar+1,npar+1
    except N.linalg.LinAlgError:
        raise N.linalg.LinAlgError("Singular matrix during spectrum extraction")
    # spec & var = nslice x Star,Sky,[slope_x...]
    spec = N.array([ N.dot(cc,bb) for cc,bb in zip(C,B) ]) # nslice,npar+1
    var  = N.array([ N.diag(cc) for cc in C ]) # nslice,npar+1

    # Compute the least-square variance using the chi2-case method
    if not chi2fit:
        A = N.array([ N.dot(xx.T, xx) for xx in
                      (Z.T * N.where(cube.var>0, 1/N.sqrt(cube.var), 0).T).T ])
        try:
            C = N.array([ N.linalg.pinv(aa) for aa in A ])
        except N.linalg.LinAlgError:
            raise N.linalg.LinAlgError("Singular matrix during variance extraction")
        var = N.array([ N.diag(cc) for cc in C ])

    # Now, what about negative sky? The pb arises for short-exposures,
    # where there's probably no sky whatsoever (except if taken during
    # twilight), and where a (significantly) negative sky is actually
    # a shortcoming of the PSF. For long exposures, one expects "some"
    # negative sky values, where sky is compatible to 0.

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
        negSky = spec[:,1]<0            # Test for presence of negative sky
        if negSky.any(): # and 'long' not in psf_fn.name.lower():
            print "WARNING: %d slices w/ sky<0 in extract_spec" % \
                  (len(negSky.nonzero()[0]))
            print_msg(str(cube.lbda[negSky]), 2, verbosity)
        #if 'short' in psf_fn.name:
            # For slices w/ sky<0, fit only PSF without background
            A = N.array([ N.dot(xx,xx) for xx in X[negSky,:,0] ])
            B = N.array([ N.dot(xx,bb)
                          for xx,bb in zip(X[negSky,:,0],b[negSky]) ])
            C = 1/A
            spec[negSky,0] = C*B        # Linear fit without sky
            spec[negSky,1] = 0          # Set sky to null
            var[negSky,0] = C
            var[negSky,1] = 0

    if method == 'psf':                 # Nothing else to be done
        return cube.lbda,spec,var

    # Reconstruct background and subtract it from cube
    bkgnd     = N.zeros_like(cube.data)
    var_bkgnd = N.zeros_like(cube.var)
    if npar_sky:
        for d in xrange(1,npar_sky+1):      # Loop over sky components
            bkgnd     += (Z[:,:,d].T * spec[:,d]).T
            var_bkgnd += (Z[:,:,d].T**2 * var[:,d]).T
    subData = cube.data - bkgnd         # Bkgnd subtraction (nslice,nlens)
    subVar = cube.var.copy()
    good = cube.var>0
    subVar[good] += var_bkgnd[good]     # Variance of bkgnd-sub. signal

    # Replace invalid data (var=0) by model PSF = Intensity*PSF
    if not good.all():
        print_msg("Replacing %d vx with modeled signal" % \
                  len((~good).nonzero()[0]), 1, verbosity)
        subData[~good] = (spec[:,0]*psf.T).T[~good]

    # Plain summation over aperture

    # Aperture center [spx] (nslice)
    xc = psf_param[2] + psf_param[0]*model.ADR_coeff[:,0]*N.cos(psf_param[1])
    yc = psf_param[3] + psf_param[0]*model.ADR_coeff[:,0]*N.sin(psf_param[1])
    # Aperture radius in spaxels
    aperRad = radius / spxSize
    print_msg("Aperture radius: %.2f arcsec = %.2f spx" % (radius,aperRad), 1, verbosity)

    # Radius [spx] (nslice,nlens)
    r = N.hypot((model.x.T - xc).T, (model.y.T - yc).T)
    # Circular aperture (nslice,nlens)
    # Use r<aperRad[:,N.newaxis] if radius is a (nslice,) vec.
    frac = (r < aperRad).astype('float')

    if method == 'subaperture':
        # fractions accounting for subspaxels (a bit slow)
        newfrac = subaperture(xc, yc, aperRad, 4)
        # remove bad spaxels since subaperture returns the full spaxel grid
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
        print_msg("  Extend FoV by %d spx: nlens=%d -> %d" % \
                  (ns, model.nlens, extnlens), 1, verbosity)

        # Compute PSF on extended range (nslice,extnlens)
        extModel = psf_fn(psf_ctes, cube, coords=(extx,exty)) # Extended model
        extPsf   = extModel.comp(param, normed=True) # nslice,extnlens

        # Embed background-subtracted data in extended model PSF
        origData = subData.copy()
        origVar  = subVar.copy()
        subData  = (spec[:,0]*extPsf.T).T   # Extended model, nslice,extnlens
        subVar   = N.zeros((extModel.nslice,extModel.nlens))
        for i in xrange(model.nlens):
            # Embeb original spx i in extended model array by finding
            # corresponding index j in new array
            j, = ((extModel.x[0]==model.x[0,i]) & \
                  (extModel.y[0]==model.y[0,i])).nonzero()
            subData[:,j[0]] = origData[:,i]
            subVar[:,j[0]]  = origVar[:,i]

        r = N.hypot((extModel.x.T - xc).T, (extModel.y.T - yc).T)
        frac = (r < aperRad).astype('float')

    if method.endswith('aperture'):
        # Replace signal and variance estimates from plain summation
        spec[:,0] = (frac    * subData).sum(axis=1)
        var[:,0]  = (frac**2 * subVar).sum(axis=1)
        return cube.lbda,spec,var

    if method == 'optimal':
        # Model signal = Intensity*PSF + bkgnd
        modsig = (spec[:,0]*psf.T).T + bkgnd # nslice,nlens

        # One has to have a model of the variance. This can be estimated from
        # a simple 'photon noise + RoN' model on each slice: signal ~ alpha*N
        # (alpha = 1/flat-field coeff and N = photon counts) and variance ~ (N
        # + RoN**2) * alpha**2 = (signal/alpha + RoN**2) * alpha**2 =
        # alpha*signal + beta. This model disregards spatial component of
        # flat-field, which is supposed to be constant of FoV.

        # Model variance = alpha*Signal + beta
        coeffs = N.array([ polyfit_clip(modsig[s], cube.var[s], 1, clip=5)
                           for s in xrange(cube.nslice) ])
        coeffs = F.median_filter(coeffs, (5,1)) # A bit of smoothing...
        modvar = N.array([ N.polyval(coeffs[s], modsig[s])
                           for s in xrange(cube.nslice) ]) # nslice,nlens

        # Optimal weighting
        norm = (frac * psf).sum(axis=1) # PSF norm, nslice
        npsf = (psf.T / norm).T         # nslice,nlens
        weight = frac * npsf / modvar   # Unormalized weights, nslice,nlens
        norm = (weight * npsf).sum(axis=1) # Weight norm, nslice
        weight = (weight.T / norm).T    # Normalized weights, nslice,nlens

        # Replace signal and variance estimates from optimal summation
        spec[:,0] = (weight    * subData).sum(axis=1)
        var[:,0]  = (weight**2 * subVar).sum(axis=1)
        return cube.lbda,spec,var

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
    # resample spaxel center positions
    # originally [-7:7]
    f = 2.**f
    epsilon = 1. / f / 2.
    border = nspaxel / 2.
    r = N.linspace(-border + epsilon, border - epsilon, nspaxel*f)

    # (x,y) positions of resampled array
    x,y = N.meshgrid(r, r)
    # spaxel fraction
    frac = N.ones(x.shape) / f**2

    try:
        xc.shape[0]
    except (AttributeError, IndexError):
        xc = isinstance(xc, (tuple, list)) and N.array(xc) or N.array([xc])

    try:
        yc.shape[0]
    except (AttributeError, IndexError):
        yc = isinstance(yc, (tuple, list)) and N.array(yc) or N.array([yc])

    try:
        rc.shape[0]
    except (AttributeError, IndexError):
        rc = isinstance(rc, (tuple, list)) and N.array(rc) or N.array([rc])

    assert xc.shape == yc.shape
    # one single radius?
    if len(rc) == 1:
        rc = N.repeat(rc, xc.shape)

    out = []
    # this loop could possibly be achieved with some higher order matrix
    for i,j,k in zip(xc, yc, rc):
        fr = frac.copy()
        # subspaxels outside circle
        fr[N.hypot(x-i, y-j) > k] = 0.
        # resample back to original size and sum
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
            hdr.update('PRESSURE',MK_pressure,"Default MK pressure [mbar]")
        pressure = MK_pressure

    temp = hdr.get('TEMP', N.nan)
    if not -20 < temp < 20:             # Non-std temperature
        print "WARNING: non-std temperature (%.0f C) updated to %.0f C" % \
              (temp, MK_temp)
        if isinstance(hdr, dict):       # pySNIFS.SNIFS_cube.e3d_data_header
            hdr['TEMP'] = MK_temp
        else:                           # True pyfits header, add comment
            hdr.update('TEMP', MK_temp, "Default MK temperature [C]")
        temp = MK_temp

    return pressure,temp

def read_psf_name(hdr):
    """Return PSF function name read (or guessed) from header."""

    assert hdr['ES_METH']=='psf', \
        "PSF reconstruction only works for PSF spectro-photometry"

    try:
        psfname = hdr['ES_PSF']
    except KeyError:
        efftime = hdr['EFFTIME']
        print "WARNING: cannot read 'ES_PSF' keyword, " \
            "guessing from EFFTIME=%.0fs" % efftime
        # Assert it's an old correlation PSF (i.e. 'long' or 'short')
        psfname = (efftime > 12.) and 'long' or 'short'

    # Convert PSF name (e.g. 'short red') to PSF function name
    # ('ShortRed_ExposurePSF')
    fnname = ''.join(map(str.capitalize,psfname.split())) + '_ExposurePSF'
    print "PSF name: %s [%s]" % (psfname, fnname)

    return eval(fnname)


def read_psf_ctes(hdr, lrange=()):
    """Read PSF constants [lbda_ref,alphaDeg,ellDeg] from header."""

    assert ('ES_LMIN' in hdr and 'ES_LMAX' in hdr) or lrange,\
       'ES_LMIN/ES_LMAX not found and lrange not set'

    # this reproduces exactly the PSF parameters used by extract_spec(full_cube...)
    if lrange:
        lmin, lmax = lrange
    else:
        lmin = hdr['ES_LMIN']
        lmax = hdr['ES_LMAX']
    lref = (lmin+lmax)/2

    # this can be put back as soon as we are sure that
    # everything is understood in what concerns the PSF
    # reconstruction from the spectra  headers
#    lref = hdr['ES_LREF']       # Reference wavelength [A]

    # Count up alpha/ell coefficients (ES_Ann/ES_Enn) to get the
    # polynomial degrees
    adeg = len([ k for k in hdr.keys()
                 if re.match('ES_A\d+$',k) is not None ]) - 1
    edeg = len([ k for k in hdr.keys()
                 if re.match('ES_E\d+$',k) is not None ]) - 1
    print "PSF constants: lref=%.2fA, alphaDeg=%d, ellDeg=%d" % (lref,adeg,edeg)

    return [lref,adeg,edeg]


def read_psf_param(hdr, lrange=()):
    """Read (7+ellDeg+alphaDeg) PSF parameters from header:
    delta,theta,x0,y0,PA,e0,...en,a0,...an."""

    airmass = hdr['ES_AIRM']    # Effective airmass
    parang = hdr['ES_PARAN']    # Effective parallactic angle [deg]
    delta = N.tan(N.arccos(1/airmass)) # ADR intensity
    theta = parang/57.295779513082323  # Parallactic angle [rad]

    # Polynomial coeffs in lr~ = lambda/LbdaRef - 1
    c_ell = [ v for k,v in hdr.items() if re.match('ES_E\d+$',k) is not None ]
    c_alp = [ v for k,v in hdr.items() if re.match('ES_A\d+$',k) is not None ]

    assert ('ES_LMIN' in hdr and 'ES_LMAX' in hdr) or lrange,\
       'ES_LMIN/ES_LMAX not found and lrange not set'

    # this can be put back as soon as we are sure that
    # everything is understood in what concerns the PSF
    # reconstruction from the spectra  headers
#    lstp = hdr['CDELT1']                                # Step
#    lmin = hdr['CRVAL1'] - (hdr.get('CRPIX1',1)-1)*lstp # Start
#    lmax = lmin + (hdr['NAXIS1']-1)*lstp                # End

    if lrange:
        lmin, lmax = lrange
    else:
        lmin = hdr['ES_LMIN']
        lmax = hdr['ES_LMAX']
    lref = hdr['ES_LREF']       # Reference wavelength [A]

    # Convert polynomial coeffs from lr~ = lambda/LbdaRef - 1 = a+b*lr
    # back to lr = (2*lambda - (lmin+lmax))/(lmax-lmin)
    a = (lmin+lmax) / (2*lref) - 1
    b = (lmax-lmin) / (2*lref)
    ecoeffs = polyConvert(c_ell, trans=(a,b), backward=True).tolist()
    acoeffs = polyConvert(c_alp, trans=(a,b), backward=True).tolist()

    x0 = hdr['ES_XC']           # Reference position [spx]
    y0 = hdr['ES_YC']
    pa = hdr['ES_XY']           # (Nearly) position angle

    # this reproduces exactly the PSF parameters used by extract_spec(full_cube...)
    # we should NOT need this if everything was properly computed at 5000.A by ES
    # which seems not to be the case for <pa>
    pressure,temp = read_PT(hdr)
    adr = A.ADR(pressure, temp, lref=(lmin+lmax)/2,
                airmass=airmass, parangle=parang)
    x0,y0 = adr.refract(x0, y0, lref, unit=0.43, backward=True)

    print "PSF parameters: airmass=%.3f, parangle=%.1fdeg, " \
        "refpos=%.2fx%.2f spx @%.2fA" % (airmass,parang,x0,y0,(lmin+lmax)/2)

    return [delta,theta,x0,y0,pa] + ecoeffs + acoeffs

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

    import scipy.misc
    a,b = trans
    m = N.zeros((n,n), dtype='d')
    for r in range(n):
        for c in range(r,n):
            m[r,c] = S.misc.comb(c,r) * b**r * a**(c-r)
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
            raise ValueError("polyfit_clip: Not enough points left (%d) " \
                             "for degree %d" % (y[good].size,deg))
    return coeffs

def chebNorm(x, xmin, xmax):
    """Normalization [xmin,xmax] to [-1,1]"""

    return ( 2*x - (xmax+xmin) ) / (xmax-xmin)

def chebEval(pars, nx, chebpolys=[]):
    """Orthogonal Chebychev polynomial expansion, x should be already
    normalized in [-1,1]."""

    if len(chebpolys)<len(pars):
        print "Initializing Chebychev polynomials up to order %d" % len(pars)
        chebpolys[:] = [ S.special.chebyu(i) for i in range(len(pars)) ]

    return N.sum( [ par*cheb(nx) for par,cheb in zip(pars,chebpolys) ], axis=0)

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
    pa = phi*A.RAD2DEG                  # From rad to deg

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
        self.lbda_ref = psf_ctes[1]     # Reference wavelength [AA]
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
        self.n_ref = A.atmosphericIndex(self.lbda_ref, P=pressure, T=temp)
        self.ADR_coeff = ( self.n_ref - 
                           A.atmosphericIndex(self.l, P=pressure, T=temp) ) * \
                           A.RAD2ARC / self.spxSize # l > l_ref <=> coeff > 0

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
        x0 = xc + delta*self.ADR_coeff*N.sin(theta) # nslice,nlens
        y0 = yc - delta*self.ADR_coeff*N.cos(theta)

        # Other params
        PA          = self.param[4]
        ellCoeffs   = self.param[5:6+self.ellDeg]
        alphaCoeffs = self.param[6+self.ellDeg:self.npar_cor]

        ell = polyEval(ellCoeffs, self.lrel) # nslice,nlens
        alpha = polyEval(alphaCoeffs, self.lrel)

        # Correlated params
        if self.correlations=='new':
            lcheb = chebNorm(self.l, *self.chebRange)
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

        # Gaussian + Moffat
        dx = self.x - x0
        dy = self.y - y0
        # CAUTION: ell & PA are not the true ellipticity and position angle!
        r2 = dx**2 + ell*dy**2 + 2*PA*dx*dy
        gaussian = N.exp(-r2/2/sigma**2)
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
        x0 = xc + delta*self.ADR_coeff*sintheta
        y0 = yc - delta*self.ADR_coeff*costheta

        # Other params
        PA  = self.param[4]
        ellCoeffs   = self.param[5:6+self.ellDeg]
        alphaCoeffs = self.param[6+self.ellDeg:self.npar_cor]

        ell = polyEval(ellCoeffs, self.lrel)
        alpha = polyEval(alphaCoeffs, self.lrel)

        # Correlated params
        if self.correlations=='new':
            lcheb = chebNorm(self.l, *self.chebRange)
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

        # Gaussian + Moffat
        dx = self.x - x0
        dy = self.y - y0
        dy2 = dy**2
        r2 = dx**2 + ell*dy2 + 2*PA*dx*dy
        gaussian = N.exp(-r2/2/sigma**2)
        ea = 1 + r2/alpha**2
        moffat = ea**(-beta)
        j1 = eta/sigma**2
        j2 = 2*beta/ea/alpha**2
        da0 = gaussian * ( e1 + s1*r2*j1/sigma ) + \
              moffat * ( -b1*N.log(ea) + r2*j2/alpha )

        # Derivatives
        grad = N.zeros((self.npar_cor+self.npar_ind,)+self.x.shape,'d')
        tmp = gaussian*j1 + moffat*j2
        grad[2] = tmp*(    dx + PA*dy)  # dPSF/dx0
        grad[3] = tmp*(ell*dy + PA*dx)  # dPSF/dy0
        grad[0] =       self.ADR_coeff*(sintheta*grad[2] - costheta*grad[3])
        grad[1] = delta*self.ADR_coeff*(sintheta*grad[3] + costheta*grad[2])
        grad[4] = -tmp   * dx*dy        # dPSF/dPA
        for i in xrange(self.ellDeg + 1):
            grad[5+i] = -tmp/2 * dy2 * self.lrel**i
        for i in xrange(self.alphaDeg + 1):
            grad[6+self.ellDeg+i] = da0 * self.lrel**i
        grad[:self.npar_cor] *= self.param[N.newaxis,self.npar_cor:,N.newaxis]
        grad[self.npar_cor] = moffat + eta*gaussian # dPSF/dI

        return grad

    def _HWHM_fn(self, r, alphaCoeffs, lbda):
        """Half-width at half maximum function (=0 at HWHM)."""

        lrel = chebNorm(lbda, self.lmin, self.lmax) # Norm to [-1,1]
        alpha = polyEval(alphaCoeffs, lrel)
        if self.correlations=='new':
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
        gaussian = N.exp(-r**2/2/sigma**2)
        moffat = (1 + r**2/alpha**2)**(-beta)

        # PSF=moffat + eta*gaussian, maximum is 1+eta
        return moffat + eta*gaussian - (eta + 1)/2

    def FWHM(self, param, lbda):
        """Estimate FWHM of PSF at wavelength lbda."""

        alphaCoeffs = param[6+self.ellDeg:self.npar_cor]
        # Compute FWHM from radial profile
        fwhm = 2*S.optimize.fsolve(func=self._HWHM_fn, x0=1.,
                                   args=(alphaCoeffs,lbda))

        # Beware: scipy-0.8.0 fsolve returns a size 1 array
        return N.squeeze(fwhm)  # In spaxels

# Old PSF parameters description without chromaticity for long and
# short exposures.

class Long_ExposurePSF(ExposurePSF):

    name = 'long'
    correlations = 'old'

    beta0  = 1.685
    beta1  = 0.345
    sigma0 = 0.545
    sigma1 = 0.215
    eta0   = 1.04
    eta1   = 0.00

class Short_ExposurePSF(ExposurePSF):

    name = 'short'
    correlations = 'old'

    beta0  = 1.395
    beta1  = 0.415
    sigma0 = 0.56
    sigma1 = 0.2
    eta0   = 0.6
    eta1   = 0.16

# New PSF parameters description using 2nd order chebychev polynomial
# for long and short exposures and blue and red channels.

class LongBlue_ExposurePSF(ExposurePSF):

    name = 'long blue'
    correlations = 'new'
    chebRange = (3399.,5100.)      # Domain of validity of Chebychev expansion

    beta0  = [ 1.220, 0.016,-0.056] # b00,b01,b02
    beta1  = [ 0.590, 0.004, 0.014] # b10,b11,b12
    sigma0 = [ 0.710,-0.024, 0.016] # s00,s01,s02
    sigma1 = [ 0.119, 0.001,-0.004] # s10,s11,s12
    eta0   = [ 0.544,-0.090, 0.039] # e00,e01,e02
    eta1   = [ 0.223, 0.060,-0.020] # e10,e11,e12

class LongRed_ExposurePSF(ExposurePSF):

    name = 'long red'
    correlations = 'new'
    chebRange = (5318.,9508.)      # Domain of validity of Chebychev expansion

    beta0  = [ 1.205,-0.100,-0.031] # b00,b01,b02
    beta1  = [ 0.578, 0.062, 0.028] # b10,b11,b12
    sigma0 = [ 0.596, 0.044, 0.011] # s00,s01,s02
    sigma1 = [ 0.173,-0.035,-0.008] # s10,s11,s12
    eta0   = [ 1.366,-0.184,-0.126] # e00,e01,e02
    eta1   = [-0.134, 0.121, 0.054] # e10,e11,e12

class ShortBlue_ExposurePSF(ExposurePSF):

    name = 'short blue'
    correlations = 'new'
    chebRange = (3399.,5100.)      # Domain of validity of Chebychev expansion

    beta0  = [ 1.355, 0.023,-0.042] # b00,b01,b02
    beta1  = [ 0.524,-0.012, 0.020] # b10,b11,b12
    sigma0 = [ 0.492,-0.037, 0.000] # s00,s01,s02
    sigma1 = [ 0.176, 0.016, 0.000] # s10,s11,s12
    eta0   = [ 0.499, 0.080, 0.061] # e00,e01,e02
    eta1   = [ 0.316,-0.015,-0.050] # e10,e11,e12

class ShortRed_ExposurePSF(ExposurePSF):

    name = 'short red'
    correlations = 'new'
    chebRange = (5318.,9508.)      # Domain of validity of Chebychev expansion

    beta0  = [ 1.350,-0.030,-0.012] # b00,b01,b02
    beta1  = [ 0.496, 0.032, 0.020] # b10,b11,b12
    sigma0 = [ 0.405,-0.003, 0.000] # s00,s01,s02
    sigma1 = [ 0.212,-0.017, 0.000] # s10,s11,s12
    eta0   = [ 0.704,-0.060, 0.044] # e00,e01,e02
    eta1   = [ 0.343, 0.113,-0.045] # e10,e11,e12


# Deprecated functions and classes ==============================

class ADR_model:

    def __init__(self, pressure=616., temp=2., **kwargs):
        """ADR_model(pressure, temp,
        [lref=, delta=, theta=, airmass=, parangle=])."""

        import warnings
        warnings.warn("Replaced by ToolBox.Atmosphere", DeprecationWarning)

        if not 550<pressure<650 and not not -20<temp<20:
            raise ValueError("ADR_model: Non-std pressure (%.0f mbar) or"
                             "temperature (%.0f C)" % (pressure, temp))
        self.P = pressure
        self.T = temp
        if 'lref' in kwargs:
            self.set_ref(lref=kwargs['lref'])
        else:
            self.set_ref()
        if 'airmass' in kwargs and 'parangle' in kwargs:
            self.set_param(kwargs['airmass'], kwargs['parangle'], obs=True)
        elif 'delta' in kwargs and 'theta' in kwargs:
            self.set_param(kwargs['delta'],kwargs['theta'])

    def __str__(self):

        s = "ADR [ref:%.0fA]: P=%.0f mbar, T=%.0fC" % \
            (self.lref,self.P,self.T)
        if hasattr(self, 'delta') and hasattr(self, 'theta'):
            s += ", airmass=%.2f, parangle=%.1f deg" % \
                 (self.get_airmass(),self.get_parangle())

        return s

    def set_ref(self, lref=5000.):

        self.lref = lref                # [Angstrom]
        self.nref = A.atmosphericIndex(self.lref, P=self.P, T=self.T)

    def set_param(self, p1, p2, obs=False):

        if obs:                         # p1 = airmass, p2 = parangle [deg]
            self.delta = N.tan(N.arccos(1./p1))
            self.theta = p2/A.RAD2DEG
        else:                           # p1 = delta, p2 = theta [rad]
            self.delta = p1
            self.theta = p2

    def refract(self, x, y, lbda, backward=False, unit=1.):
        """Return refracted position at wavelength lbda from reference
        position x,y (in units of unit in arcsec)."""

        if not hasattr(self, 'delta'):
            raise AttributeError("ADR parameters 'delta' and 'theta' "
                                 "are not set.")

        x0 = N.atleast_1d(x)            # (npos,)
        y0 = N.atleast_1d(y)
        npos = len(x0)
        assert len(x0)==len(y0), "Incompatible x and y vectors."
        lbda = N.atleast_1d(lbda)       # (nlbda,)
        nlbda = len(lbda)

        dz = (self.nref - A.atmosphericIndex(lbda, P=self.P, T=self.T)) * \
             A.RAD2ARC / unit           # (nlbda,)
        dz *= self.delta

        if backward:
            assert npos==nlbda, "Incompatible x,y and lbda vectors."
            x = x0 - dz*N.sin(self.theta)
            y = y0 + dz*N.cos(self.theta) # (nlbda=npos,)
            out = N.vstack((x,y))         # (2,npos)
        else:
            dz = dz[:,N.newaxis]          # (nlbda,1)
            x = x0 + dz*N.sin(self.theta) # (nlbda,npos)
            y = y0 - dz*N.cos(self.theta) # (nlbda,npos)
            out = N.dstack((x.T,y.T)).T   # (2,nlbda,npos)
            assert out.shape == (2,nlbda,npos), "ARGH"

        return N.squeeze(out)

    def get_airmass(self):
        return 1/N.cos(N.arctan(self.delta))

    def get_parangle(self):
        return self.theta*A.RAD2DEG # Parangle in deg.
