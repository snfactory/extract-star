#!/usr/bin/env python
##############################################################################
## Filename:      extract_star.py
## Version:       $Revision$
## Description:   Standard star spectrum extraction
## Author:        $Author$
## $Id$
##############################################################################

"""Primarily based on E. Pecontal's point source extractor (extract_star.py).
This version replaces the double gaussian PSF profile by an ad-hoc PSF profile
(correlated Gaussian + Moffat)."""

__author__ = "C. Buton, Y. Copin, E. Pecontal"
__version__ = '$Id$'

import os
import optparse

import pyfits                           # getheader
import pySNIFS
import pySNIFS_fit

import scipy as S
from scipy import stats
from scipy import linalg as L
from scipy import interpolate as I 
from scipy.ndimage import filters as F

SpaxelSize = 0.43                       # Spaxel size in arcsec
MK_pressure = 616.                      # Default pressure [mbar]
MK_temp = 2.                            # Default temperature [C]

class ADR_model:

    def __init__(self, pressure=616., temp=2., **kwargs):
        """ADR_model(pressure, temp, [lref=, delta=, theta=])."""

        if not 550<pressure<650 and not not -20<temp<20:
            raise ValueError("ADR_model: Non-std pressure (%.0f mbar) or"
                             "temperature (%.0f C)" % (pressure, temp))        
        self.P = pressure
        self.T = temp
        if 'lref' in kwargs:
            self.set_ref(lref=kwargs['lref'])
        else:
            self.set_ref()
        if 'delta' in kwargs and 'theta' in kwargs:
            self.set_param(delta=kwargs['delta'],theta=kwargs['theta'])

    def __str__(self):

        s = "ADR: P=%.0f mbar, T=%.0fC" % (self.P,self.T)
        if hasattr(self, 'lref'):
            s += ", ref.lambda=%.0fA" % self.lref
        if hasattr(self, 'delta') and hasattr(self, 'theta'):
            s += ", delta=%.2f, theta=%.1f deg" % \
                 (self.delta,self.theta/S.pi*180)

        return s

    def set_ref(self, lref=5000.):

        self.lref = lref
        self.nref = atmosphericIndex(self.lref, P=self.P, T=self.T)

    def set_param(self, delta, theta):

        self.delta = delta
        self.theta = theta

    def refract(self, x, y, lbda, backward=False):

        if not hasattr(self, 'delta'):
            raise AttributeError("ADR parameters 'delta' and 'theta' are not set.")

        x0 = S.atleast_1d(x)            # (npos,)
        y0 = S.atleast_1d(y)
        assert len(x)==len(y), "Incompatible x and y vectors."
        lbda = S.atleast_1d(lbda)       # (nlbda,)
        npos = len(x)
        nlbda = len(lbda)

        dz = (self.nref - atmosphericIndex(lbda, P=self.P, T=self.T)) * \
             206265. / SpaxelSize       # (nlbda,)
        dz *= self.delta

        if backward:
            assert npos==nlbda, "Incompatible x,y and lbda vectors."
            x = x0 - dz*S.sin(self.theta)
            y = y0 + dz*S.cos(self.theta) # (nlbda=npos,)
            out = S.vstack((x,y))       # (2,npos)
        else:
            dz = dz[:,S.newaxis]        # (nlbda,1)
            x = x0 + dz*S.sin(self.theta) # (nlbda,npos)
            y = y0 - dz*S.cos(self.theta) # (nlbda,npos)
            out = S.dstack((x.T,y.T)).T # (2,nlbda,npos)
            assert out.shape == (2,nlbda,npos), "ARGH"

        return S.squeeze(out)


# Definitions ================================================================

def print_msg(str, limit):
    """Print message 'str' if verbosity level (opts.verbosity) >= limit."""

    if opts.verbosity >= limit:
        print str


def atmosphericIndex(lbda, P=616., T=2.):
    """Compute atmospheric refractive index: lbda in angstrom, P
    in mbar, T in C. Relative humidity effect is neglected.

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
        ( 1 + (1.049 - 0.0157*T)*1e-6*P ) / ( 720.883*(1 + 0.003661*T) )
    
    return n


def read_PT(hdr, update=False):
    """Read pressure and temperature from hdr, and check value consistency."""

    pressure = hdr.get('PRESSURE', S.nan)
    temp = hdr.get('TEMP', S.nan)

    if not 550<pressure<650:        # Non-std pressure
        if update:
            print "WARNING: non-std pressure (%.0f mbar) updated to %.0f mbar" % \
                  (pressure, MK_pressure)
            hdr.update('PRESSURE',MK_pressure,"Default MaunaKea pressure [mbar]")
        pressure = MK_pressure
    if not -20<temp<20:             # Non-std temperature
        if update:
            print "WARNING: non-std temperature (%.0f C) updated to %.0f C" % \
                  (temp, MK_temp)
            hdr.update('TEMP', MK_temp, "Default MaunaKea temperature [C]")
        temp = MK_temp

    return pressure,temp


def estimate_parangle(hdr):
    """Estimate parallactic angle [degree] from header keywords."""

    from math import sin,cos,pi,sqrt,acos,atan2

    d2r = pi/180.                       # Degree to Radians
    # DTCS latitude is probably not the most precise one (see fit_ADR.py)
    phi = hdr['LATITUDE']*d2r           # Latitude [rad]
    sinphi = sin(phi)
    cosphi = cos(phi)
    try:
        ha = hdr['HAMID']               # Hour angle (format: 04:04:52.72)
    except:
        ha = hdr['HA']                  # Hour angle (format: 04:04:52.72)
    try:
        dec = hdr['TELDEC']             # Declination (format 08:23:19.20)
    except:
        dec = hdr['DEC']                # Declination (format 08:23:19.20)
    # We neglect position offset (see
    # https://projects.lbl.gov/mantis/view.php?id=280 note 773) since offset
    # keywords are not universal...

    def dec_deg(dec):
        """Convert DEC string (DD:MM:SS.SS) to degrees."""
        l = [ float(x) for x in dec.split(':') ]
        return l[0] + l[1]/60. + l[2]/3600.

    ha  = dec_deg(ha)*15*d2r            # Hour angle [rad]
    dec = dec_deg(dec)*d2r              # Declination [rad]
    sinha = sin(ha)
    cosha = cos(ha)
    sindec = sin(dec)
    cosdec = cos(dec)

    # Zenithal angle (to be compared to dec_deg(hdr['ZD']))
    cosdz = sindec*sinphi + cosphi*cosdec*cosha
    sindz = sqrt(1. - cosdz**2)

    # Parallactic angle (to be compared to hdr['PARANG'])
    sineta = sinha*cosphi / sindz
    coseta = ( cosdec*sinphi - sindec*cosphi*cosha ) / sindz
    eta = atan2(sineta,coseta)          # [rad]

    return eta/d2r                      # [deg]


def eval_poly(coeffs, x):
    """Evaluate polynom sum_i ci*x**i on x. It uses 'natural' convention for
    polynomial coeffs: [c0,c1...,cn] (opposite to S.polyfit).."""

    if S.isscalar(x):
        y = 0                           # Faster on scalar
        for i,c in enumerate(coeffs):
            # Incremental computation of x**i is only slightly faster
            y += c * x**i
    else:                               # Faster on arrays
        y = S.polyval(coeffs[::-1], x)  # Beware coeffs order!
        
    return y


def laplace_filtering(cube, eps=1e-4):

    lapl = F.laplace(cube.data/cube.data.mean())
    fdata = F.median_filter(cube.data, size=[1, 3])
    hist = pySNIFS.histogram(S.ravel(S.absolute(lapl)), nbin=100,
                             Max=100, cumul=True)
    threshold = hist.x[S.argmax(S.where(hist.data<(1-eps), 0, 1))]
    print_msg("Laplace filter threshold [eps=%g]: %.2f" % (eps,threshold), 2)
    filt = (S.absolute(lapl) <= threshold)

    return filt


def polyfit_clip(x, y, deg, clip=3, nitermax=10):
    """Least squares polynomial fit with sigma-clipping (if clip>0). Returns
    polynomial coeffs w/ same convention as S.polyfit: [cn,...,c1,c0]."""
    
    good = S.ones(y.shape, dtype='bool')
    niter = 0
    while True:
        niter += 1
        coeffs = S.polyfit(x[good], y[good], deg)
        old = good
        if clip:
            dy = S.polyval(coeffs, x) - y
            good = S.absolute(dy) < clip*S.std(dy)
        if (good==old).all(): break     # No more changes, stop there
        if niter > nitermax:            # Max. # of iter, stop there
            print_msg("polyfit_clip reached max. # of iterations: " \
                      "deg=%d, clip=%.2f x %f, %d px removed" % \
                      (deg, clip, S.std(dy), len((~old).nonzero()[0])), 2)
            break
        if y[good].size <= deg+1:
            raise ValueError("polyfit_clip: Not enough points left (%d) " \
                             "for degree %d" % (y[good].size,deg))

    return coeffs


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

    # Rejection of bad points (YC: need some clarifications...)
    filt = laplace_filtering(cube)
    if (~filt).any():
        print "WARNING: %d/%d vx in %d/%d slices filtered out in extract_spec" % \
              (len((~filt).nonzero()[0]), cube.data.size, 
               len(S.nonzero([ f.any() for f in ~filt ])[0]), cube.nslice)
    cube.var *= filt                    # Discard non-selected voxels

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
    bkgnd = 0
    var_bkgnd = 0
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
        coeffs = S.array([ polyfit_clip(modsig[s], cube.var[s], 1, clip=5)
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
    
    print_msg("  Adjusted parameters: [delta],[theta],xc,yc,PA,ell,alpha,I,"
              "%d bkgndCoeffs" % (skyDeg and npar_sky or 0),2)

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
        hess = pySNIFS_fit.approx_deriv(model_star.objgrad, model_star.fitpar,
                                        order=2)

        if model_star.fitpar[5]>0 and \
               model_star.fitpar[6]>0 and model_star.fitpar[7]>0: 
            cov = S.linalg.inv(hess[2:,2:]) # Discard 1st 2 lines (unfitted)
            diag = cov.diagonal()
            if (diag>0).all():
                errorpar = S.concatenate(([0.,0.], S.sqrt(diag)))
                # Shall we *= model_star.khi2, see
                # http://www.asu.edu/sas/sasdoc/sashtml/stat/chap45/sect24.htm
            else:                       # Some negative diagonal elements!
                print "WARNING: negative covariance diag. elements in metaslice %d" % (i+1)
                model_star.khi2 *= -1   # To be discarded
                errorpar = S.zeros(len(error_mat.T))
        else:
            # Set error to 0 if alpha, intens. or ellipticity is 0.
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
        print_msg("  Fit result [DoF=%d chi2=%f]: %s" % \
                  (model_star.dof,model_star.khi2,model_star.fitpar), 2)

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

    logfile.write('# lbda  %s        chi2\n' % \
                  '  '.join('%8s +/- d%-8s' % (n,n)
                            for n in ['delta','theta','xc','yc','PA',
                                      'ell','alpha','I'] + \
                            ['sky%d' % d for d in xrange(npar_sky)] ))
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
                       cube,cube_fit,fitpar,khi3D,errorpar,lbda_ref):

    logfile = open(filename,'w')
    logfile.write('# cube    : %s   \n' % os.path.basename(opts.input))
    logfile.write('# object  : %s   \n' % object)
    logfile.write('# airmass : %.2f \n' % airmass)
    logfile.write('# efftime : %.2f \n' % efftime)

    # Global parameters
    logfile.write('# lref  %s        chi2\n' % \
                  '  '.join('%8s +/- d%-8s' % (n,n)
                            for n in ['delta','theta','xc','yc','PA'] + \
                            ['ell%d' % d for d in xrange(ellDeg+1)] +
                            ['alpha%d' % d for d in xrange(alphaDeg+1)]))
    fmt = '%6.0f  ' + \
          '  '.join(["%10.4g"]*((5+(ellDeg+1)+(alphaDeg+1))*2+1)) + \
          '\n'
    list3D = [lbda_ref,
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

    logfile.write('# lbda  %s        chi2\n' % \
                  '  '.join('%8s +/- d%-8s' % (n,n)
                            for n in ['I'] + \
                            ['sky%d' % d for d in xrange(npar_sky)] ))
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


def fill_header(hdr, param, lbda_ref, opts, khi2, seeing, tflux, sflux):
    """Fill header hdr with fit-related keywords."""
    
    hdr.update('ES_VERS', __version__)
    hdr.update('ES_CUBE', opts.input, 'Input cube')
    hdr.update('ES_LREF', lbda_ref,   'Lambda ref. [A]')
    hdr.update('ES_SDEG', opts.skyDeg,'Polynomial bkgnd degree')
    hdr.update('ES_KHI2', khi2,       'Chi2 of 3D fit')
    hdr.update('ES_AIRM', 1/S.cos(S.arctan(param[0])), 'Effective airmass')
    hdr.update('ES_PARAN',param[1]/S.pi*180, 'Effective parangle [deg]')
    hdr.update('ES_XC',   param[2],   'xc @lbdaRef [spx]')
    hdr.update('ES_YC',   param[3],   'yc @lbdaRef [spx]')
    hdr.update('ES_PA'   ,param[4],   'Position angle')
    for i in xrange(opts.ellDeg + 1):    
        hdr.update('ES_E%i' % i, param[5+i], 'Ellipticity coeff. e%d' % i)
    for i in xrange(opts.alphaDeg + 1):
        hdr.update('ES_A%i' % i, param[6+opts.ellDeg+i], 'Alpha coeff. a%d' % i)
    if opts.supernova:
        hdr.update('ES_SNMOD',opts.supernova,'Supernova mode')
    hdr.update('ES_METH', opts.method, 'Extraction method')
    if method != 'psf':
        hdr.update('ES_APRAD', opts.radius, 'Aperture radius [sigma]')
    hdr.update('ES_TFLUX',tflux,      'Sum of the spectrum flux')
    if opts.skyDeg >= 0:
        hdr.update('ES_SFLUX',sflux,  'Sum of the sky flux')
    hdr.update('SEEING', seeing, 'Seeing [arcsec] (extract_star)')


# PSF classes ================================================================

class ExposurePSF:
    """
    Empirical PSF 3D function used by the L{model} class.
    """
    
    def __init__(self, psf_ctes, cube, coords=None):
        """Initiating the class.
        @param psf_ctes: Internal parameters (pixel size in cube spatial unit,
                       reference wavelength and polynomial degree of alpha). A
                       list of three numbers.
        @param cube: Input cube. This is a L{SNIFS_cube} object.
        @param coords: if not None, should be (x,y).
        """
        self.spxSize  = psf_ctes[0]     # Spaxel size [arcsec]
        self.lbda_ref = psf_ctes[1]     # Reference wavelength [AA]
        self.alphaDeg = int(psf_ctes[2]) # Alpha polynomial degree
        self.ellDeg   = int(psf_ctes[3]) # Ellip polynomial degree
        
        self.npar_cor = 7 + self.alphaDeg + self.ellDeg # PSF parameters
        self.npar_ind = 1               # Intensity parameters per slice
        self.nslice = cube.nslice
        self.npar = self.npar_cor + self.npar_ind*self.nslice

        if coords is None:
            self.nlens = cube.nlens
            self.x = S.resize(cube.x, (self.nslice,self.nlens)) # nslice,nlens
            self.y = S.resize(cube.y, (self.nslice,self.nlens))
        else:
            x = coords[0].ravel()
            y = coords[1].ravel()
            assert len(x)==len(y), \
                   "Incompatible coordinates (%d/%d)" % (len(x),len(y))
            self.nlens = len(x)
            self.x = S.resize(x, (self.nslice,self.nlens)) # nslice,nlens
            self.y = S.resize(y, (self.nslice,self.nlens))
        self.l = S.resize(cube.lbda, (self.nlens,self.nslice)).T # nslice,nlens

        # ADR in spaxels (nslice,nlens)
        if hasattr(cube,'e3d_data_header'): # Read from cube if possible
            pressure,temp = read_PT(cube.e3d_data_header)
        else:
            pressure,temp = MK_pressure,MK_temp # Default values for P and T
        self.n_ref = atmosphericIndex(self.lbda_ref, P=pressure, T=temp)
        self.ADR_coeff = ( self.n_ref - \
                           atmosphericIndex(self.l, P=pressure, T=temp) ) * \
                           206265 / self.spxSize # l > l_ref <=> coeff > 0

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

        self.param = S.asarray(param)

        # ADR params
        delta = self.param[0]
        theta = self.param[1]
        xc    = self.param[2]
        yc    = self.param[3]
        x0 = xc + delta*self.ADR_coeff*S.sin(theta) # nslice,nlens
        y0 = yc - delta*self.ADR_coeff*S.cos(theta)

        # Other params
        PA          = self.param[4]
        ellCoeffs   = self.param[5:6+self.ellDeg]        
        alphaCoeffs = self.param[6+self.ellDeg:self.npar_cor]

        lbda_rel = self.l / self.lbda_ref - 1 # nslice,nlens
        ell = eval_poly(ellCoeffs, lbda_rel)
        alpha = eval_poly(alphaCoeffs, lbda_rel)

        # Correlated params
        s1,s0,b1,b0,e1,e0 = self.corrCoeffs
        sigma = s0 + s1*alpha
        beta  = b0 + b1*alpha
        eta   = e0 + e1*alpha
        
        # Gaussian + Moffat
        dx = self.x - x0
        dy = self.y - y0
        r2 = dx**2 + ell*dy**2 + 2*PA*dx*dy
        gaussian = S.exp(-r2/2/sigma**2)
        moffat = (1 + r2/alpha**2)**(-beta)

        # Function
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
        x0 = xc + delta*self.ADR_coeff*sintheta
        y0 = yc - delta*self.ADR_coeff*costheta
        
        # Other params
        PA  = self.param[4]
        ellCoeffs   = self.param[5:6+self.ellDeg]                
        alphaCoeffs = self.param[6+self.ellDeg:self.npar_cor]

        lbda_rel = self.l / self.lbda_ref - 1
        ell = eval_poly(ellCoeffs, lbda_rel)
        alpha = eval_poly(alphaCoeffs, lbda_rel)

        # Correlated params
        s1,s0,b1,b0,e1,e0 = self.corrCoeffs
        sigma = s0 + s1*alpha
        beta  = b0 + b1*alpha
        eta   = e0 + e1*alpha

        # Gaussian + Moffat
        dx = self.x - x0
        dy = self.y - y0
        dy2 = dy**2
        r2 = dx**2 + ell*dy2 + 2*PA*dx*dy
        gaussian = S.exp(-r2/2/sigma**2)
        ea = 1 + r2/alpha**2
        moffat = ea**(-beta)
        j1 = eta/sigma**2
        j2 = 2*beta/ea/alpha**2
        da0 = gaussian * ( e1 + s1*r2*j1/sigma ) + \
              moffat * ( -b1*S.log(ea) + r2*j2/alpha )

        # Derivatives
        grad = S.zeros((self.npar_cor+self.npar_ind,)+self.x.shape,'d')
        tmp = gaussian*j1 + moffat*j2
        grad[2] = tmp*(    dx + PA*dy)  # dPSF/dx0
        grad[3] = tmp*(ell*dy + PA*dx)  # dPSF/dy0
        grad[0] =       self.ADR_coeff*(sintheta*grad[2] - costheta*grad[3])
        grad[1] = delta*self.ADR_coeff*(sintheta*grad[3] + costheta*grad[2])
        grad[4] = -tmp   * dx*dy        # dPSF/dPA
        for i in xrange(self.ellDeg + 1):
            grad[5+i] = -tmp/2 * dy2 * lbda_rel**i
        for i in xrange(self.alphaDeg + 1):
            grad[6+self.ellDeg+i] = da0 * lbda_rel**i
        grad[:self.npar_cor] *= self.param[S.newaxis,self.npar_cor:,S.newaxis]
        grad[self.npar_cor] = moffat + eta*gaussian # dPSF/dI

        return grad

    def _HWHM_fn(self, r, alphaCoeffs, lbda):
        """Half-width at half maximum function (=0 at HWHM)."""

        lbda_rel = lbda/self.lbda_ref - 1
        alpha = eval_poly(alphaCoeffs, lbda_rel)
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

        alphaCoeffs = param[6+self.ellDeg:self.npar_cor]
        # Compute FWHM from radial profile
        fwhm = 2*S.optimize.fsolve(func=self._HWHM_fn, x0=1.,
                                   args=(alphaCoeffs,lbda))

        return fwhm                     # In spaxels


class Long_ExposurePSF(ExposurePSF): 

    name = 'long'
    corrCoeffs = [0.215,0.545,0.345,1.685,0.0,1.04] # long exposures


class Short_ExposurePSF(ExposurePSF):

    name = 'short'
    corrCoeffs = [0.2,0.56,0.415,1.395,0.16,0.6] # short exposures


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
                      "[%default sigma]", default=5.)
    parser.add_option("-p", "--plot", action='store_true',
                      help="Plot flag (syn. '--graph=png')")
    parser.add_option("-g", "--graph", type="string",
                      help="Graphic output format ('eps', 'png' or 'pylab')")
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

    opts,args = parser.parse_args()
    if not opts.input:
        parser.error("No input datacube specified.")

    if opts.graph:
        opts.plot = True
    elif opts.plot:
        opts.graph = 'png'

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
    ise3d = pyfits.getheader(opts.input,1)['NAXIS']!=3
    if ise3d:
        full_cube = pySNIFS.SNIFS_cube(e3d_file=opts.input)
    else:
        full_cube = pySNIFS.SNIFS_cube(fits3d_file=opts.input)
    step = full_cube.lstep
    print_msg("Cube %s: %d slices [%.2f-%.2f], %d spaxels" % \
              (os.path.basename(opts.input), full_cube.nslice,
               full_cube.lbda[0], full_cube.lbda[-1], full_cube.nlens), 1)

    # The full_cube.e3d_data_header dictionary is not enough for later updates
    # in fill_hdr, which requires a *true* pyfits header.
    inhdr = pyfits.getheader(opts.input, 1) # 1st extension
    obj = inhdr.get('OBJECT', 'Unknown')
    efftime = inhdr['EFFTIME']
    airmass = inhdr['AIRMASS']
    parangle = inhdr.get('PARANG', S.nan)
    if S.isnan(parangle):
        print "WARNING: cannot read PARANG keyword, estimate it from header"
        parangle = estimate_parangle(inhdr)
    channel = inhdr['CHANNEL'][0].upper()
    pressure,temp = read_PT(inhdr, update=True)

    ellDeg   = opts.ellDeg
    alphaDeg = opts.alphaDeg
    npar_psf = 7 + ellDeg + alphaDeg

    skyDeg   = opts.skyDeg
    npar_sky = int((skyDeg+1)*(skyDeg+2)/2)

    # Select the PSF (short or long)
    psfFn = (efftime > 12.) and Long_ExposurePSF or Short_ExposurePSF

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

    if ise3d:
        cube   = pySNIFS.SNIFS_cube(e3d_file=opts.input, slices=slices)
    else:
        cube   = pySNIFS.SNIFS_cube(fits3d_file=opts.input, slices=slices)
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
    lbda_ref = 5000.               # Use constant lbda_ref for easy comparison
    nslice = cube.nslice
    lbda_rel = cube.lbda / lbda_ref - 1
    
    # 1) ADR parameters (from keywords)
    delta = S.tan(S.arccos(1./airmass)) # ADR power
    theta = parangle/180.*S.pi          # ADR angle [rad]

    # 2) Reference position
    # Convert meta-slice centroids to position at ref. lbda, and clip around
    # median position
    adr = ADR_model(pressure, temp, lref=lbda_ref, delta=delta, theta=theta)
    print_msg(str(adr), 1)
    xref,yref = adr.refract(xc_vec,yc_vec, cube.lbda, backward=True)
    valid = khi2_vec > 0                # Discard unfitted slices
    x0,y0 = S.median(xref[valid]),S.median(yref[valid]) # Robust to outliers
    r = S.hypot(xref - x0, yref - y0)
    rmax = 5*S.median(r[valid])         # Robust to outliers
    good = valid & (r <= rmax)          # Valid fit and reasonable position
    bad = valid & (r > rmax)            # Valid fit but discarded position
    if (valid & bad).any():
        print "WARNING: %d metaslices discarded after ADR selection" % \
              (len(S.nonzero(valid & bad)))
    print_msg("%d/%d centroids found withing %.2f spx of (%.2f,%.2f)" % \
              (len(xref[good]),len(xref),rmax,x0,y0), 1)
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
    print_msg("  Reference position guess [%.2fA]: %.2f x %.2f spx" % \
              (lbda_ref,xc,yc), 1)
    print_msg("  ADR guess: delta=%.2f, theta=%.1f deg" % \
              (delta, theta/S.pi*180), 1)

    # 3) Other parameters
    PA       = S.median(PA_vec[good])
    polEll   = pySNIFS.fit_poly(ell_vec[good],3,ellDeg,lbda_rel[good])
    polAlpha = pySNIFS.fit_poly(alpha_vec[good],3,alphaDeg,lbda_rel[good])

    # Filling in the guess parameter arrays (px) and bounds arrays (bx)
    p1     = [None]*(npar_psf+nslice)
    p1[:5] = [delta, theta, xc, yc, PA]
    p1[5:6+ellDeg]        = polEll.coeffs[::-1]
    p1[6+ellDeg:npar_psf] = polAlpha.coeffs[::-1]
    p1[npar_psf:npar_psf+nslice] = int_vec.tolist()

    if opts.supernova:                  # Fix all parameters but intensities
        print "WARNING: supernova-mode, no 3D PSF-fit"
        # This mode completely discards 3D fit. In pratice, a 3D-fit is still
        # performed on intensities, just to be coherent w/ the remaining of
        # the code.
        b1 = [[delta, delta],           # delta 
              [theta, theta],           # theta 
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

    func = [ '%s;%f,%f,%f,%f' % \
             (psfFn.name,SpaxelSize,lbda_ref,alphaDeg,ellDeg) ] # PSF
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
               skyDeg>=0 and (npar_sky*nslice) or 0), 3)
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
    khi2 = data_model.khi2              # Reduced khi2 of meta-fit
    khi2 *= data_model.dof              # Restore real chi2
    cov = data_model.param_error(fitpar) # Covariance matrix
    errorpar = S.sqrt(cov.diagonal())

    print_msg("  Fit result: DoF: %d, chi2=%f" % (data_model.dof, khi2), 2)
    print_msg("  Fit result [PSF param]: %s" % fitpar[:npar_psf], 2)
    print_msg("  Fit result [Intensities]: %s" % \
              fitpar[npar_psf:npar_psf+nslice], 3)
    if skyDeg >= 0:
        print_msg("  Fit result [Background]: %s" % \
                  fitpar[npar_psf+nslice:], 3)

    print_msg("  Reference position fit [%.2fA]: %.2f x %.2f spx" % \
              (lbda_ref,fitpar[2],fitpar[3]), 1)
    print_msg("  ADR fit: delta=%.2f, theta=%.1f deg" % \
              (fitpar[0], fitpar[1]/S.pi*180), 1)

    # Compute seeing (FWHM in arcsec)
    seeing = data_model.func[0].FWHM(fitpar[:npar_psf], lbda_ref) * SpaxelSize
    print "  Seeing estimate: %.2f'' FWHM" % seeing
    print "  Effective airmass: %.2f" % (1/S.cos(S.arctan(fitpar[0])))

    # Test positivity of alpha and ellipticity. At some point, maybe it would
    # be necessary to force positivity in the fit (e.g. fmin_cobyla).
    fit_alpha = eval_poly(fitpar[6+ellDeg:npar_psf], lbda_rel)
    if fit_alpha.min() < 0:
        raise ValueError("Alpha is negative (%.2f) at %.0fA" % \
                         (fit_alpha.min(), cube.lbda[fit_alpha.argmin()]))
    fit_ell = eval_poly(fitpar[5:6+ellDeg], lbda_rel)
    if fit_ell.min() < 0:
        raise ValueError("Ellipticity is negative (%.2f) at %.0fA" % \
                         (fit_ell.min(), cube.lbda[fit_ell.argmin()]))

    # Computing final spectra for object and background ======================

    # Compute aperture radius
    if opts.method == 'psf':
        radius = None
        method = 'psf'
    else:
        radius = opts.radius * seeing/2.355 # Aperture radius [arcsec]
        method = "%s r=%.1f sigma=%.2f''" % \
                 (opts.method, opts.radius, radius)
    print "Extracting the spectrum [method=%s]..." % method
    if skyDeg < 0:
        print "WARNING: no background adjusted"

    psfCtes = [SpaxelSize,lbda_ref,alphaDeg,ellDeg]
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
    # Do not use directly data_model.evalfit() since we want to keep
    # psf and bkg separated. But in the end, psf+bkg = data_model.evalfit()
    cube_fit = pySNIFS.SNIFS_cube(lbda=cube.lbda)
    cube_fit.x = cube_fit.i - 7     # x in spaxel 
    cube_fit.y = cube_fit.j - 7     # y in spaxel
    
    psf_model = psfFn(psfCtes, cube=cube_fit)
    #psf_model = data_model.func[0]
    psf = psf_model.comp(fitpar[:psf_model.npar])
    cube_fit.data = psf

    if skyDeg >= 0:
        bkg_model = pySNIFS_fit.poly2D(skyDeg, cube_fit)
        #bkg_model = data_model.func[1]
        bkg = bkg_model.comp(fitpar[psf_model.npar: \
                                    psf_model.npar+bkg_model.npar])
        cube_fit.data += bkg

    # Update header ==========================================================
    
    tflux = spec[:,0].sum()             # Total flux of extracted spectrum
    if skyDeg >= 0:
        sflux = bkg_spec.data.sum()     # Total flux of sky
    else:
        sflux = 0                       # No stored anyway
    
    fill_header(inhdr,fitpar[:npar_psf],lbda_ref,opts,khi2,seeing,tflux,sflux)

    # Save star spectrum =====================================================

    if not opts.out:
        opts.out = 'spec_%s.fits' % (channel)
        print "WARNING: saving output source spectrum to %s" % opts.out

    print lbda[0],step
    star_spec = pySNIFS.spectrum(data=spec[:,0],start=lbda[0],step=step)
    print star_spec.start,star_spec.step
    star_spec.WR_fits_file(opts.out,header_list=inhdr.items())
    star_var = pySNIFS.spectrum(data=var[:,0],start=lbda[0],step=step)
    star_var.WR_fits_file('var_'+opts.out,header_list=inhdr.items())

    # Save sky spectrum/spectra ==============================================

    if skyDeg >= 0:
        if not opts.sky:
            opts.sky = 'sky_%s.fits' % (channel)
            print "WARNING: saving output sky spectrum to %s" % opts.sky

        sky_spec = pySNIFS.spectrum(data=bkg_spec.data,start=lbda[0],step=step)
        sky_spec.WR_fits_file(opts.sky,header_list=inhdr.items())
        sky_var = pySNIFS.spectrum(data=bkg_spec.var,start=lbda[0],step=step)
        sky_var.WR_fits_file('var_'+opts.sky,header_list=inhdr.items())

    # Save 3D adjusted parameter file ========================================
    
    if opts.File:
        print "Producing 3D adjusted parameter file '%s'..." % opts.File
        create_3D_log_file(opts.File,obj,airmass,efftime,
                           cube,cube_fit,fitpar,khi2,errorpar,lbda_ref)
    
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
        axS.set_title("Point-source spectrum [%s, %s]" % (obj,method))
        axS.set_xlim(star_spec.x[0],star_spec.x[-1])
        axS.set_xticklabels([])
        axS.text(0.95,0.8, os.path.basename(opts.input), fontsize='smaller',
                 horizontalalignment='right', transform=axS.transAxes)

        axN.plot(star_spec.x, S.sqrt(star_var.data), 'b')

        if skyDeg >= 0:
            axB.plot(bkg_spec.x, bkg_spec.data, 'g')
            axB.set_xlim(bkg_spec.x[0],bkg_spec.x[-1])
            axB.set_title("Background spectrum (per sq. arcsec)")
            axB.set_xticklabels([])

            axN.plot(bkg_spec.x, S.sqrt(bkg_spec.var), 'g')
            axN.set_title("Error spectra")

        axN.set_title("Error spectrum")
        axN.semilogy()
        axN.set_xlim(star_spec.x[0],star_spec.x[-1])
        axN.set_xlabel("Wavelength [A]")

        # Plot of the fit on each slice --------------------------------------

        print_msg("Producing slice fit plot %s..." % plot2, 1)

        ncol = S.floor(S.sqrt(nslice))
        nrow = S.ceil(nslice/float(ncol))

        fig2 = pylab.figure()
        fig2.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97)
        mod = data_model.evalfit()
        for i in xrange(nslice):        # Loop over meta-slices
            ax = fig2.add_subplot(nrow, ncol, i+1)
            data = cube.data[i,:]
            fit = mod[i,:]
            fmin = min(data.min(), fit.min()) - 2e-2
            ax.plot(data - fmin)        # Signal
            ax.plot(fit - fmin)         # Fit
            ax.semilogy()
            ax.set_xlim(0,len(data))
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.1,0.8, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='left', transform=ax.transAxes)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Spaxel ID", fontsize=8)
                ax.set_ylabel("Flux + cte", fontsize=8)

        # Plot of the fit on rows and columns sum ----------------------------

        print_msg("Producing profile plot %s..." % plot3, 1)

        fig3 = pylab.figure()
        fig3.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97)
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
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.1,0.8, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='left', transform=ax.transAxes)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I (blue) or J (red)", fontsize=8)
                ax.set_ylabel("Flux", fontsize=8)

        # Plot of the star center of gravity and adjusted center -------------

        print_msg("Producing ADR plot %s..." % plot4, 1)

        xguess = xc + delta*psf_model.ADR_coeff[:,0]*S.sin(theta)
        yguess = yc - delta*psf_model.ADR_coeff[:,0]*S.cos(theta)
        xfit = fitpar[2] + fitpar[0]*psf_model.ADR_coeff[:,0]*S.sin(fitpar[1])
        yfit = fitpar[3] - fitpar[0]*psf_model.ADR_coeff[:,0]*S.cos(fitpar[1])

        fig4 = pylab.figure()
        ax4a = fig4.add_subplot(2, 2, 1)
        ax4b = fig4.add_subplot(2, 2, 2)
        ax4c = fig4.add_subplot(2, 1, 2, aspect='equal', adjustable='datalim')

        ax4a.errorbar(cube.lbda[good], xc_vec[good], yerr=error_mat[good,2],
                      fmt='b.',ecolor='b',label="Fit 2D")
        if bad.any():
            ax4a.plot(cube.lbda[bad],xc_vec[bad],'r.', label='_nolegend_')
        ax4a.plot(cube.lbda, xguess, 'k--', label="Guess 3D")
        ax4a.plot(cube.lbda, xfit, 'g', label="Fit 3D")
        ax4a.set_xlabel("Wavelength [A]")
        ax4a.set_ylabel("X center [spaxels]")
        pylab.setp(ax4a.get_xticklabels()+ax4a.get_yticklabels(), fontsize=8)
        leg = ax4a.legend(loc='best')
        pylab.setp(leg.get_texts(), fontsize='smaller')

        ax4b.errorbar(cube.lbda[good], yc_vec[good], yerr=error_mat[good,3],
                      fmt='b.',ecolor='b')
        if bad.any():
            ax4b.plot(cube.lbda[bad],yc_vec[bad],'r.')
        ax4b.plot(cube.lbda, yfit, 'g')
        ax4b.plot(cube.lbda, yguess, 'k--')
        ax4b.set_xlabel("Wavelength [A]")
        ax4b.set_ylabel("Y center [spaxels]")
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
        ax4c.plot((x0,xc),(y0,yc),'k-')
        ax4c.plot(xguess, yguess, 'k--') # Guess ADR
        ax4c.plot(xfit, yfit, 'g')       # Adjusted ADR
        ax4c.set_autoscale_on(False)
        ax4c.plot((xc,),(yc,),'k+')
        ax4c.add_patch(matplotlib.patches.Circle((x0,y0),radius=rmax,
                                                 ec='0.8',fc=None))
        ax4c.add_patch(matplotlib.patches.Rectangle((-7.5,-7.5),15,15,
                                                 ec='0.8',lw=2,fc=None)) # FoV
        ax4c.text(0.03, 0.85,
                  'Guess: x0,y0=%4.2f,%4.2f  airmass=%.2f parangle=%.1fdeg' % \
                  (xc, yc, airmass, parangle),
                  transform=ax4c.transAxes, fontsize='smaller')
        ax4c.text(0.03, 0.75,
                  'Fit: x0,y0=%4.2f,%4.2f  airmass=%.2f parangle=%.1fdeg' % \
                  (fitpar[2], fitpar[3],
                   1/S.cos(S.arctan(fitpar[0])), fitpar[1]/S.pi*180),
                  transform=ax4c.transAxes, fontsize='smaller')
        ax4c.set_xlabel("X center [spaxels]")
        ax4c.set_ylabel("Y center [spaxels]")
        fig4.text(0.5, 0.93, "ADR plot [%s, airmass=%.2f]" % (obj, airmass),
                  horizontalalignment='center', size='large')

        # Plot of the other model parameters ---------------------------------

        print_msg("Producing model parameter plot %s..." % plot6, 1)

        guess_ell = eval_poly(polEll.coeffs[::-1], lbda_rel)
        guess_alpha = eval_poly(polAlpha.coeffs[::-1], lbda_rel)

        def estimate_error(x, cov, idx):
            return S.sqrt(eval_poly(cov.diagonal()[idx], x))
        def plot_conf_interval(ax, x, y, dy):
            ax.plot(x, y, 'g', label="Fit 3D")
            ax.plot(x, y+dy, 'g:', label='_nolegend_')
            ax.plot(x, y-dy, 'g:', label='_nolegend_')

        err_ell   = estimate_error(lbda_rel,cov,range(5,6+ellDeg))
        err_alpha = estimate_error(lbda_rel,cov,range(6+ellDeg,npar_psf))
        err_PA = S.sqrt(cov.diagonal()[4])

        fig6 = pylab.figure()
        ax6a = fig6.add_subplot(2, 1, 1)
        ax6b = fig6.add_subplot(4, 1, 3)
        ax6c = fig6.add_subplot(4, 1, 4)

        ax6a.errorbar(cube.lbda[good], alpha_vec[good], error_mat[good,6],
                      fmt='b.', ecolor='b', label="Fit 2D")
        if bad.any():
            ax6a.plot(cube.lbda[bad],alpha_vec[bad],'r.', label="_nolegend_")
        ax6a.plot(cube.lbda, guess_alpha, 'k--', label="Guess 3D")
        plot_conf_interval(ax6a, cube.lbda, fit_alpha, err_alpha)
        ax6a.text(0.03, 0.15,
                  'Guess: %s' % \
                  (', '.join([ 'a%d=%.2f' % (i,a) for i,a in
                              enumerate(polAlpha.coeffs[::-1]) ]) ),
                  transform=ax6a.transAxes, fontsize='smaller')
        ax6a.text(0.03, 0.05,
                  'Fit: %s' % \
                  (', '.join(['a%d=%.2f' % (i,a) for i,a in
                             enumerate(fitpar[6+ellDeg:npar_psf])])),
                  transform=ax6a.transAxes, fontsize='smaller')
        leg = ax6a.legend(loc='best')
        pylab.setp(leg.get_texts(), fontsize='smaller')
        ax6a.set_ylabel('Alpha [spx]')
        ax6a.set_xticklabels([])
        ax6a.set_title("Model parameters [%s, seeing %.2f'' FWHM]" % \
                       (obj,seeing))

        ax6b.errorbar(cube.lbda[good], ell_vec[good], error_mat[good,5],
                      fmt='b.',ecolor='blue')
        if bad.any():
            ax6b.plot(cube.lbda[bad],ell_vec[bad],'r.')
        ax6b.plot(cube.lbda, guess_ell, 'k--')
        plot_conf_interval(ax6b, cube.lbda, fit_ell, err_ell)
        ax6b.text(0.03, 0.3,
                  'Guess: %s' % \
                  (', '.join([ 'e%d=%.2f' % (i,e)
                              for i,e in enumerate(polEll.coeffs[::-1]) ]) ),
                  transform=ax6b.transAxes, fontsize='smaller')
        ax6b.text(0.03, 0.1,
                  'Fit: %s' % \
                  (', '.join([ 'e%d=%.2f' % (i,e)
                              for i,e in enumerate(fitpar[5:6+ellDeg]) ])),
                  transform=ax6b.transAxes, fontsize='smaller')
        ax6b.set_ylabel('1/q^2')
        ax6b.set_xticklabels([])        

        ax6c.errorbar(cube.lbda[good], PA_vec[good]/S.pi*180,
                      error_mat[good,4]/S.pi*180, fmt='b.', ecolor='b')
        if bad.any():
            ax6c.plot(cube.lbda[bad],PA_vec[bad]/S.pi*180,'r.')
        ax6c.plot([cube.lbda[0],cube.lbda[-1]], [PA/S.pi*180]*2, 'k--')
        plot_conf_interval(ax6c, S.asarray([cube.lbda[0],cube.lbda[-1]]),
                           S.asarray([fitpar[4]/S.pi*180]*2),
                           S.asarray([err_PA/S.pi*180]*2))
        ax6c.set_ylabel('PA [deg]')
        ax6c.text(0.03, 0.1,
                  'Guess: PA=%4.2f  Fit: PA=%4.2f' % \
                  (PA/S.pi*180,fitpar[4]/S.pi*180),
                  transform=ax6c.transAxes, fontsize='smaller')
        ax6c.set_xlabel("Wavelength [A]")

        # Plot of the radial profile -----------------------------------------

        print_msg("Producing radial profile plot %s..." % plot7, 1)

        fig7 = pylab.figure()
        fig7.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97)

        def ellRadius(x,y, x0,y0, ell, q):
            dx = x - x0
            dy = y - y0
            return S.sqrt(dx**2 + ell*dy**2 + 2*q*dx*dy)
        
        for i in xrange(nslice):        # Loop over slices
            ax = fig7.add_subplot(nrow, ncol, i+1)
            # Use adjusted elliptical radius instead of plain radius
            #r = S.hypot(cube.x-xfit[i],cube.y-yfit[i])
            #rfit = S.hypot(cube_fit.x-xfit[i],cube_fit.y-yfit[i])
            r = ellRadius(cube.x,cube.y, xfit[i],yfit[i], fit_ell[i], fitpar[4])
            rfit = ellRadius(cube_fit.x,cube_fit.y, xfit[i],yfit[i],
                             fit_ell[i], fitpar[4])
            ax.plot(r, cube.data[i], 'b.')
            ax.plot(rfit, cube_fit.data[i], 'r,')
            ax.plot(rfit, psf[i], 'g,')
            if skyDeg >= 0:
                ax.plot(rfit, bkg[i],'c,')
            ax.semilogy()
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
            ax.text(0.9,0.8, "%.0f" % cube.lbda[i], fontsize=8,
                    horizontalalignment='right', transform=ax.transAxes)
            if method!='psf':
                ax.axvline(radius/SpaxelSize, color='y', lw=2)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Elliptical radius [spaxels]", fontsize=8)
                ax.set_ylabel("Flux", fontsize=8)
            ax.axis([0, rfit.max()*1.1, 
                     cube.data[i][cube.data[i]>0].min()/1.2,
                     cube.data[i].max()*1.2])

        # Radial Chi2 plot (not activated by default)
        if False:
            print_msg("Producing radial chi2 plot...", 1)
            fig = pylab.figure()
            fig.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97)
            for i in xrange(nslice):        # Loop over slices
                ax = fig.add_subplot(nrow, ncol, i+1)
                rfit = ellRadius(cube_fit.x,cube_fit.y, xfit[i],yfit[i],
                                 fit_ell[i], fitpar[4])
                chi2 = (cube.slice2d(i,coord='p') - \
                        cube_fit.slice2d(i,coord='p'))**2 / \
                        cube.slice2d(i,coord='p',var=True)
                ax.plot(rfit, chi2.flatten(), 'b.')
                ax.semilogy()
                pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=6)
                ax.text(0.9,0.8, "%.0f" % cube.lbda[i], fontsize=8,
                        horizontalalignment='right', transform=ax.transAxes)
                if method!='psf':
                    ax.axvline(radius/SpaxelSize, color='y', lw=2)
                if ax.is_last_row() and ax.is_first_col():
                    ax.set_xlabel("Elliptical radius [spaxels]", fontsize=8)
                    ax.set_ylabel("chi2", fontsize=8)

        # Contour plot of each slice -----------------------------------------

        print_msg("Producing PSF contour plot %s..." % plot8, 1)

        fig8 = pylab.figure()
        fig8.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97,
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
                                                       fc=None, ec='y', lw=2))
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
        fig5.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97,
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
