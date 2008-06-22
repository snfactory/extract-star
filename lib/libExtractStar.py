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

import scipy as S

MK_pressure = 616.                      # Default pressure [mbar]
MK_temp = 2.                            # Default temperature [C]

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

    def refract(self, x, y, lbda, backward=False, unit=1.):

        if not hasattr(self, 'delta'):
            raise AttributeError("ADR parameters 'delta' and 'theta' are not set.")

        x0 = S.atleast_1d(x)            # (npos,)
        y0 = S.atleast_1d(y)
        assert len(x)==len(y), "Incompatible x and y vectors."
        lbda = S.atleast_1d(lbda)       # (nlbda,)
        npos = len(x)
        nlbda = len(lbda)

        dz = (self.nref - atmosphericIndex(lbda, P=self.P, T=self.T)) * \
             206265. / unit             # (nlbda,)
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

# PSF classes ================================================================

def eval_poly(coeffs, x):
    """Evaluate polynom sum_i ci*x**i on x. It uses 'natural' convention for
    polynomial coeffs: [c0,c1...,cn] (opposite to S.polyfit)."""

    if S.isscalar(x):
        y = 0                           # Faster on scalar
        for i,c in enumerate(coeffs):
            # Incremental computation of x**i is only slightly faster
            y += c * x**i
    else:                               # Faster on arrays
        y = S.polyval(coeffs[::-1], x)  # Beware coeffs order!
        
    return y

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



