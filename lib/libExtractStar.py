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

DEG2RAD = S.pi/180

# Chebychev coefficients for 2nd order polynomial
chebs = [ S.special.chebyu(i) for i in range(3) ]

def atmosphericIndex(lbda, P=616., T=2.):
    """Compute atmospheric refractive index: lbda in angstrom, P in mbar, T in
    C. Relative humidity effect is neglected. From Fillipenko, 1982 (from
    Edlen 1953).

    Cohen & Cromer 1988 (PASP, 100, 1582) give P=456 mmHg=608 mbar and T=2C
    for Mauna Kea. However, SNIFS observations show an average recorded
    pression of 616 mbar and temperature of 2C (see
    https://projects.lbl.gov/mantis/view.php?id=1341).

    The difference between sea-level and MK refractive index is ~1e-4. The
    index fluctuations with typical P,T MK-variations (+/-5mbar, +/-5C) is +/-
    ~3e-6. Water vapor contribution is around ~1e-10.

    See also Peck & Reeder (1972, J.Opt.Soc.Am., 62, 958) and 
    http://emtoolbox.nist.gov/Wavelength/Documentation.asp"""

    # Sea-level (P=760 mmHg, T=15C)
    iml2 = 1/(lbda*1e-4)**2             # lambda in microns
    n = 1 + 1e-6*(64.328 + 29498.1/(146-iml2) + 255.4/(41-iml2))

    if P is not None and T is not None:
        # (P,T) correction
        P *= 0.75006168                 # Convert P to mmHg: *= 760./1013.25
        n = 1 + (n-1) * P * \
            ( 1 + (1.049 - 0.0157*T)*1e-6*P ) / ( 720.883*(1 + 0.003661*T) )
    
    return n


def read_PT(hdr):
    """Read pressure and temperature from hdr, and check value consistency."""

    pressure = hdr.get('PRESSURE', S.nan)
    if not 550<pressure<650:        # Non-std pressure
        print "WARNING: non-std pressure (%.0f mbar) updated to %.0f mbar" % \
              (pressure, MK_pressure)
        if isinstance(hdr, dict):       # pySNIFS.SNIFS_cube.e3d_data_header
            hdr['PRESSURE'] = MK_pressure
        else:                           # True pyfits header, add comment
            hdr.update('PRESSURE',MK_pressure,"Default MK pressure [mbar]")
        pressure = MK_pressure

    temp = hdr.get('TEMP', S.nan)
    if not -20<temp<20:             # Non-std temperature
        print "WARNING: non-std temperature (%.0f C) updated to %.0f C" % \
              (temp, MK_temp)
        if isinstance(hdr, dict):       # pySNIFS.SNIFS_cube.e3d_data_header
            hdr['TEMP'] = MK_temp
        else:                           # True pyfits header, add comment
            hdr.update('TEMP', MK_temp, "Default MK temperature [C]")
        temp = MK_temp

    return pressure,temp


def read_parangle(hdr):
    """Read or estimate parallactic angle [degree] from header keywords."""

    parang = hdr.get('PARANG', S.nan)
    if S.isfinite(parang):              # PARANG keyword is available
        return parang

    # PARANG keyword is absent, estimate it from LATITUDE,HA,DEC
    print "WARNING: cannot read PARANG keyword, estimate it from header"
    
    from math import sin,cos,pi,sqrt,atan2

    d2r = pi/180.                       # Degree to Radians
    # DTCS latitude is probably not the most precise one (see fit_ADR.py)
    phi = hdr['LATITUDE']*d2r           # Latitude [rad]
    sinphi = sin(phi)
    cosphi = cos(phi)
    try:
        ha = hdr['HAMID']               # Hour angle (format: 04:04:52.72)
    except KeyError:
        ha = hdr['HA']                  # Hour angle (format: 04:04:52.72)
    try:
        dec = hdr['TELDEC']             # Declination (format 08:23:19.20)
    except KeyError:
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

    parang = eta/d2r                    # [deg]
    print "  Estimated parallactic angle: %.2f deg" % parang
    if isinstance(hdr, dict):           # pySNIFS.SNIFS_cube.e3d_data_header
        hdr['PARANG'] = parang
    else:                               # True pyfits header, add comment
        hdr.update('PARANG',parang,"Parallactic angle [deg]")

    return parang


class ADR_model:

    def __init__(self, pressure=616., temp=2., **kwargs):
        """ADR_model(pressure, temp,
        [lref=, delta=, theta=, airmass=, parangle=])."""

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
        self.nref = atmosphericIndex(self.lref, P=self.P, T=self.T)

    def set_param(self, p1, p2, obs=False):

        if obs:                         # p1 = airmass, p2 = parangle [deg]
            self.delta = S.tan(S.arccos(1./p1))
            self.theta = p2*DEG2RAD
        else:                           # p1 = delta, p2 = theta [rad]
            self.delta = p1
            self.theta = p2

    def refract(self, x, y, lbda, backward=False, unit=1.):
        """Return refracted position at wavelength lbda from reference
        position x,y (in units of unit in arcsec)."""

        if not hasattr(self, 'delta'):
            raise AttributeError("ADR parameters 'delta' and 'theta' "
                                 "are not set.")

        x0 = S.atleast_1d(x)            # (npos,)
        y0 = S.atleast_1d(y)
        npos = len(x0)
        assert len(x0)==len(y0), "Incompatible x and y vectors." 
        lbda = S.atleast_1d(lbda)       # (nlbda,)
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

    def get_airmass(self):
        return 1/S.cos(S.arctan(self.delta))

    def get_parangle(self):
        return self.theta/DEG2RAD       # Parangle in deg.


# Polynomial utilities ======================================================

def polyEval(coeffs, x):
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

def polyConvMatrix(n, trans=(0,1)):
    """Return the upper triangular matrix (i,k) * b**k * a**(i-k), that
    converts polynomial coeffs for x~:=a+b*x (P~ = a0~ + a1~*x~ + a2~*x~**2 +
    ...) in polynomial coeffs for x (P = a0 + a1*x + a2*x**2 +
    ...). Therefore, (a,b)=(0,1) gives identity."""

    a,b = trans
    m = S.zeros((n,n), dtype='d')
    for r in range(n):
        for c in range(r,n):
            m[r,c] = S.comb(c,r) * b**r * a**(c-r)
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
    return S.dot(polyConvMatrix(len(coeffs), (a,b)),coeffs)

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

def chebychev(pars, x, xmin, xmax, chebs):
    """Orthogonal Chebychev polynomial"""

    # Normalization -> [-1,1] 
    nx = ( 2*x - (xmax+xmin) ) / (xmax-xmin)

    return S.sum( [ p * ch(nx) for p,ch in zip(pars,chebs) ], axis=0)

# Ellipse utilities ==============================

def quadEllipse(a,b,c,d,f,g):
    """Ellipse elements (center, semi-axes and PA) from the general
    quadratic curve a*x2 + 2*b*x*y + c*y2 + 2*d*x + 2*f*y + g = 0.

    http://mathworld.wolfram.com/Ellipse.html"""

    D = S.linalg.det([[a,b,d],[b,c,f],[d,f,g]])
    J = S.linalg.det([[a,b],[b,c]])
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
    ap = S.sqrt( 2*(a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g) /
                 (b2mac * (S.sqrt((a-c)**2 + 4*b**2) - (a+c))) )
    bp = S.sqrt( 2*(a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g) /
                 (b2mac * (-S.sqrt((a-c)**2 + 4*b**2) - (a+c))) )
    # Position angle
    if b==0:
        phi = 0
    else:
        phi = S.tan((a-c)/(2*b))/2
    if a>c:
        phi += S.pi/2

    return x0,y0,ap,bp,phi

def flatAndPA(cy2, c2xy):
    """Return flattening q=b/a and position angle PA [deg] for ellipse defined
    by x**2 + cy2*y**2 + 2*c2xy*x*y = 1.
    """

    x0,y0,a,b,phi = quadEllipse(1,c2xy,cy2,0,0,-1)
    assert a>0 and b>0, "Input equation does not correspond to an ellipse"
    q = b/a                             # Flattening
    pa = phi/DEG2RAD                    # From rad to deg

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
        if self.nslice > 1:
            self.lmin = cube.lbda[0]
            self.lmax = cube.lbda[-1]
            lrel = ( 2*cube.lbda - (self.lmin + self.lmax) ) / \
                   ( self.lmax - self.lmin ) # From -1 to +1
            self.lrel = S.resize(lrel, (self.nlens,self.nslice)).T
        else:
            self.lmin,self.lmax = -1,+1
            self.lrel = self.l

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

        ell = polyEval(ellCoeffs, self.lrel) # nslice,nlens
        alpha = polyEval(alphaCoeffs, self.lrel)

        # Correlated params
        b0 = chebychev(self.beta0, self.l, self.lbda_min, self.lbda_max, chebs)
        b1 = chebychev(self.beta1, self.l, self.lbda_min, self.lbda_max, chebs)
        s0 = chebychev(self.sigma0, self.l, self.lbda_min, self.lbda_max, chebs)
        s1 = chebychev(self.sigma1, self.l, self.lbda_min, self.lbda_max, chebs)
        e0 = chebychev(self.eta0, self.l, self.lbda_min, self.lbda_max, chebs)
        e1 = chebychev(self.eta1, self.l, self.lbda_min, self.lbda_max, chebs)                 
        sigma = s0 + s1*alpha
        beta  = b0 + b1*alpha
        eta   = e0 + e1*alpha
        
        # Gaussian + Moffat
        dx = self.x - x0
        dy = self.y - y0
        # CAUTION: ell & PA are not the true ellipticity and position angle!
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

        ell = polyEval(ellCoeffs, self.lrel)
        alpha = polyEval(alphaCoeffs, self.lrel)

        # Correlated params
        b0 = chebychev(self.beta0, self.l, self.lbda_min, self.lbda_max, chebs)
        b1 = chebychev(self.beta1, self.l, self.lbda_min, self.lbda_max, chebs)
        s0 = chebychev(self.sigma0, self.l, self.lbda_min, self.lbda_max, chebs)
        s1 = chebychev(self.sigma1, self.l, self.lbda_min, self.lbda_max, chebs)
        e0 = chebychev(self.eta0, self.l, self.lbda_min, self.lbda_max, chebs)
        e1 = chebychev(self.eta1, self.l, self.lbda_min, self.lbda_max, chebs)          
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
            grad[5+i] = -tmp/2 * dy2 * self.lrel**i
        for i in xrange(self.alphaDeg + 1):
            grad[6+self.ellDeg+i] = da0 * self.lrel**i
        grad[:self.npar_cor] *= self.param[S.newaxis,self.npar_cor:,S.newaxis]
        grad[self.npar_cor] = moffat + eta*gaussian # dPSF/dI

        return grad

    def _HWHM_fn(self, r, alphaCoeffs, lbda):
        """Half-width at half maximum function (=0 at HWHM)."""

        lrel = ( 2*lbda - (self.lmin+self.lmax) ) / ( self.lmax-self.lmin )
        alpha = polyEval(alphaCoeffs, lrel)
        b0 = chebychev(self.beta0, lbda, self.lbda_min, self.lbda_max, chebs)
        b1 = chebychev(self.beta1, lbda, self.lbda_min, self.lbda_max, chebs)
        s0 = chebychev(self.sigma0, lbda, self.lbda_min, self.lbda_max, chebs)
        s1 = chebychev(self.sigma1, lbda, self.lbda_min, self.lbda_max, chebs)
        e0 = chebychev(self.eta0, lbda, self.lbda_min, self.lbda_max, chebs)
        e1 = chebychev(self.eta1, lbda, self.lbda_min, self.lbda_max, chebs)          
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

class long_blue_exposure_psf(ExposurePSF): 

    name = 'long blue'
    lbda_min = 3399.0
    lbda_max = 5100.0

    beta0  = [ 1.318,-0.022, 0.031] # b00,b01,b02
    beta1  = [ 0.519, 0.017, 0.011] # b10,b11,b12
    sigma0 = [ 0.565, 0.007, 0.024] # s00,s01,s02
    sigma1 = [ 0.234,-0.009,-0.029] # s10,s11,s12
    eta0   = [ 1.106,-0.036,-0.109] # e00,e01,e02
    eta1   = [-0.184, 0.019, 0.056] # e10,e11,e12    

class long_red_exposure_psf(ExposurePSF):         

    name = 'long red'
    lbda_min = 5318.0
    lbda_max = 9508.0
    
    beta0  = [ 1.294,-0.049, 0.021] # b00,b01,b02
    beta1  = [ 0.506, 0.024, 0.020] # b10,b11,b12
    sigma0 = [ 0.536, 0.031, 0.024] # s00,s01,s02
    sigma1 = [ 0.248,-0.022,-0.029] # s10,s11,s12
    eta0   = [ 1.251,-0.063,-0.091] # e00,e01,e02   
    eta1   = [-0.229, 0.042, 0.057] # e10,e11,e12
    
class short_blue_exposure_psf(ExposurePSF):

    name = 'short blue'
    lbda_min = 3399.0
    lbda_max = 5100.0    
    
    beta0  = [ 1.433, 0.030, 0.024] # b00,b01,b02
    beta1  = [ 0.470,-0.013, 0.029] # b10,b11,b12
    sigma0 = [ 0.515,-0.019, 0.022] # s00,s01,s02
    sigma1 = [ 0.234, 0.003,-0.042] # s10,s11,s12
    eta0   = [ 0.525,-0.009,-0.116] # e00,e01,e02   
    eta1   = [ 0.016, 0.017, 0.055] # e10,e11,e12
    
class short_red_exposure_psf(ExposurePSF):

    name = 'short red'
    lbda_min = 5318.0
    lbda_max = 9508.0
    
    beta0  = [1.407,-0.012,-0.009] # b00,b01,b02
    beta1  = [0.476, 0.019, 0.022] # b10,b11,b12
    sigma0 = [0.535,-0.012, 0.007] # s00,s01,s02
    sigma1 = [0.166,-0.009,-0.009] # s10,s11,s12
    eta0   = [0.013,-0.082, 0.000] # e00,e01,e02   
    eta1   = [0.521, 0.101, 0.000] # e10,e11,e12
