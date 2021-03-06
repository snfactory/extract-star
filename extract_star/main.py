# -*- coding: utf-8 -*-

"""
3D PSF-based point-source extractor. The PSF is a constrained
Gaussian+Moffat with elliptical distortion.

Todo:

* one could use Aitchison (or 'additive log-ratio') transform to
  enforce the normalization constraint on alphas (see
  http://thread.gmane.org/gmane.comp.python.scientific.user/16180/focus=16187)
  or Multinomial logit (see
  http://en.wikipedia.org/wiki/Multinomial_logit and
  http://thread.gmane.org/gmane.comp.python.scientific.user/20318/focus=20320)

Polynomial approximation
========================

To minimize parameter correlations, the polynomial approximations for
alpha and ellipticity are adjusted internally as function of lr :=
(2*lambda - (lmin+lmax))/(lmax-lmin) where lmin and lmax being the
wavelength of first and last meta-slices. The polynomial
approximations for alpha and ellipticity stored in the log files
(options '-F' and '-f') are expressed in this internal coordinate.

However, in order to have channel-independent expressions, the coeffs
stored in header keywords are for polynoms of lr := lambda/lref - 1
(see below).

PSF parameter keywords
======================

- ES_VERS: code version
- ES_CUBE: input cube
- ES_LREF: reference wavenlength (arbitrary set to 5000 A)
- ES_SDEG: polynomial background degree (-1: no background, 0: constant, etc.)
- ES_CHI2: total chi2 of 3D-fit (or Residual Sum of Squares for lsq-fit)
- ES_AIRM: Effective airmass, from ADR fit
- ES_PARAN: Effective parallactic angle [deg], from ADR fit
- ES_[X|Y]C: Point-source position [spx] at reference wavelength
- ES_XY: XY coefficient (position angle)
- ES_L[MIN|MAX]: wavelength of first/last meta-slice
- ES_Exx: Y2 coefficients (flattening)
- ES_Axx: alpha coefficient
- ES_METH: extraction method (psf|optimal|[sub]aperture)
- ES_PSF: PSF name and model ('[short|long], classic[-powerlaw]' or
  '[long|short] [blue|red], chromatic')
- ES_SUB: PSF sub-sampling factor
- ES_APRAD: aperture radius (>0: arcsec, <0: seeing sigma)
- ES_TFLUX: integrated flux of extracted point-source spectrum
- ES_SFLUX: integrated flux of sky spectrum [per square-arcsec]
- SEEING: estimated seeing FWHM [\"] at reference wavelength
- ES_PRIOR: PSF prior hyper-scale (aka 'supernova mode')
- ES_PRISE: Seeing prior [arcsec]
- ES_PRIXY: 'x,y' or 'DDT' or 'cubefit' priors on position
- ES_BNDx: constraints on 3D-PSF parameters

The chromatic evolution of Y2-coefficient can be computed from ES_Exx
coefficients of polynom function of relative wavelength
lr:=lbda/lref-1:

Y2-coeff(lbda) = e0 + e1*lr + e2*lr**2 + ...

With a polynomial (i.e. non-'powerlaw') PSF, the chromatic evolution
of alpha is computed similarly from ES_Axx coefficients:

alpha(lbda) = a0 + a1*lr + a2*lr**2 + ...

hence alpha(lref) = a0.

For a 'powerlaw' PSF, the appropriate expression is:

alpha(lbda) = a[-1] * (lbda/lref)**( a[-2] + a[-3]*(lbda/lref - 1) + ...)

hence alpha(lref) = a[-1].
"""

from __future__ import print_function

import os
import sys
import warnings

from astropy.io import fits as F
import numpy as N

from . import pySNIFS
from . import pySNIFS_fit
from . import libExtractStar as libES
from .extern import Atmosphere as TA
from .extern.Arrays import metaslice
from .extern.Misc import make_method, warning2stdout
from .version import __version__

warnings.showwarning = warning2stdout   # Redirect warnings to stdout
warnings.filterwarnings("ignore", "Overwriting existing file")

# Numpy setup
N.set_printoptions(linewidth=999)       # X-wide lines

LbdaRef = libES.LbdaRef

# MLA tilt: MLA vertical is rotated by ~5° wrt. north: theta_MLA =
# theta_DTCS + 5°
DTHETA = {'B': 2.6, 'R': 5.0}             # [deg]

MAX_POSITION_PRIOR_OFFSET = 1    # Max position offset wrt. prior [spx]
MAX_SEEING_PRIOR_OFFSET = 40     # Max seeing offset wrt. prior [%]
MAX_AIRMASS_PRIOR_OFFSET = 20    # Max airmass offset wrt. prior [%]
MAX_PARANG_PRIOR_OFFSET = 20     # Max parangle offset wrt. prior [deg]
MAX_POSITION = 6                 # Max position wrt. FoV center [spx]

MIN_SEEING = 0.3                 # Min reasonable seeing ['']
MAX_SEEING = 4.0                 # Max reasonable seeing ['']
MAX_AIRMASS = 4.                 # Max reasonable airmass
MIN_ELLIPTICITY = 0.2            # Min reasonable ellipticity
MAX_ELLIPTICITY = 5.0            # Max reasonable ellipticity

# Non-default colors (from Colorbrewer2.org, qualitative)
class Colors(object):
    blue   = '#377EB8'
    red    = '#E41A1C'
    green  = '#4DAF4A'
    orange = '#FF7F00'
    purple = '#984EA3'
    yellow = '#FFFF33'
    brown  = '#A65628'
    pink   = '#F781BF'
    gray   = '#999999'

# Definitions ================================================================


def print_msg(str, limit):
    """Print message 'str' if verbosity level (opts.verbosity) >= limit."""

    libES.print_msg(str, limit, verb=opts.verbosity)


def spec_covariance(cube, psf, skyDeg, covpar):
    """
    Compute point-source spectrum full covariance from parameter
    covariance.
    """

    psfFn, psfCtes, fitpar = psf

    # Function fitpar to point-source spectrum (nslice,)
    func = lambda fitpar: libES.extract_specs(cube,
                                              (psfFn, psfCtes, fitpar),
                                              skyDeg=skyDeg)[1][:, 0]
    # Associated jacobian (numerical evaluation) (npar,nslice)
    jac = pySNIFS_fit.approx_deriv(func, fitpar)

    # Covariance propagation
    return N.dot(N.dot(jac.T, covpar), jac)  # (nslice,nslice)


@make_method(pySNIFS.spectrum)
def write_fits(self, filename=None, header=None):
    """
    Overrides pySNIFS_fit.spectrum.WR_fits_file. Allows full header
    propagation (including comments) and covariance matrix storage.
    """

    assert not (self.start is None or self.step is None or self.data is None)

    # Primary HDU: signal
    hdusig = F.PrimaryHDU(self.data, header=header)
    for key in ['EXTNAME', 'CTYPES', 'CRVALS', 'CDELTS', 'CRPIXS']:
        if key in hdusig.header:
            del(hdusig.header[key])  # Remove technical keys from E3D cube
    hdusig.header.set('CRVAL1', self.start, after='NAXIS1')
    hdusig.header.set('CDELT1', self.step, after='CRVAL1')

    hduList = F.HDUList([hdusig])

    # 1st extension 'VARIANCE': variance
    if self.has_var:
        hduvar = F.ImageHDU(self.var, name='VARIANCE')
        hduvar.header['CRVAL1'] = self.start
        hduvar.header['CDELT1'] = self.step

        hduList.append(hduvar)

    # 2nd (compressed) extension 'COVAR': covariance (lower triangle)
    if hasattr(self, 'cov'):
        hducov = F.ImageHDU(N.tril(self.cov), name='COVAR')
        hducov.header['CRVAL1'] = self.start
        hducov.header['CDELT1'] = self.step
        hducov.header['CRVAL2'] = self.start
        hducov.header['CDELT2'] = self.step

        hduList.append(hducov)

    if filename:                        # Save hduList to disk
        hduList.writeto(filename, output_verify='silentfix', clobber=True)

    return hduList                      # For further handling if needed


@make_method(pySNIFS.SNIFS_cube)
def flag_nans(cube, varflag=0, name='cube'):
    """
    Flag non-finite values (NaN or Inf) in `cube.data|var` in with
    `varflag` in `cube.var`.
    """

    # bad = (cube.data == 0) & (cube.var != 0)
    # if bad.any():
    #     print "WARNING: %s data contains %d strictly null values." % \
    #         (name, len(cube.data[bad]))
    #     cube.var[bad] = varflag

    for label, arr in (('data', cube.data), ('variance', cube.var)):
        bad = ~N.isfinite(arr)
        if bad.any():
            print("WARNING: %s %s contains %d NaNs/%d Inf values." % \
                  (name, label, len(arr[N.isnan(arr)]), len(arr[N.isinf(arr)])))
        cube.var[bad] = varflag

    # Look for suspicious variances
    medvar = N.median(cube.var[cube.var > 0])
    bad = cube.var / medvar > 1e6
    if bad.any():
        print("WARNING: %s contains %d suspicious variance values." % \
              (name, len(cube.var[bad])))
        if len(cube.var[bad]) > len(cube.var):
            raise ValueError("%s is too heavily corrupted, aborting" % name)

    cube.var[bad] = varflag


def create_2Dlog(opts, cube, params, dparams, chi2):
    """Dump an informative text log about the PSF (metaslice) 2D-fit."""

    logfile = file(opts.log2D, 'w')

    logfile.write('# cube    : %s   \n' % os.path.basename(opts.input))
    logfile.write('# object  : %s   \n' % cube.e3d_data_header["OBJECT"])
    logfile.write('# airmass : %.3f \n' % cube.e3d_data_header["AIRMASS"])
    logfile.write('# efftime : %.3f \n' % cube.e3d_data_header["EFFTIME"])

    if opts.skyDeg == -2:       # Step background
        npar_sky = 2
    else:                       # Polynomial background (or none)
        npar_sky = (opts.skyDeg + 1) * (opts.skyDeg + 2) / 2

    delta, theta = params[:2]
    xc, yc = params[2:4]
    xy, ell, alpha = params[4:7]
    intensity = params[-npar_sky - 1]
    sky = params[-npar_sky:]

    names = ['delta', 'theta', 'x0', 'y0', 'xy', 'ell', 'alpha', 'I'] + \
            ['sky%d' % d for d in xrange(npar_sky)]
    labels = '# lbda  ' + '  '.join('%8s +/- d%-8s' % (n, n) for n in names)
    if cube.var is None:        # Least-square fit: compute Res. Sum of Squares
        labels += '        RSS\n'
    else:                       # Chi2 fit: compute chi2 per slice
        labels += '        chi2\n'
    logfile.write(labels)
    fmt = '%6.0f  ' + '  '.join(["%10.4g"] * ((8 + npar_sky) * 2 + 1)) + '\n'

    for n in xrange(cube.nslice):
        list2D = [cube.lbda[n],
                  delta[n], dparams[n][0],
                  theta[n], dparams[n][1],
                  xc[n], dparams[n][2],
                  yc[n], dparams[n][3],
                  xy[n], dparams[n][4],
                  ell[n], dparams[n][5],
                  alpha[n], dparams[n][6],
                  intensity[n], dparams[n][-npar_sky - 1]]
        if npar_sky:
            tmp = N.transpose((sky[:, n], dparams[n][-npar_sky:]))
            list2D += tmp.flatten().tolist()
        list2D += [chi2[n]]
        logfile.write(fmt % tuple(list2D))

    logfile.close()


def create_3Dlog(opts, cube, cube_fit, fitpar, dfitpar, chi2):
    """Dump an informative text log about the PSF (full-cube) 3D-fit."""

    logfile = file(opts.log3D, 'w')

    logfile.write('# cube    : %s   \n' % os.path.basename(opts.input))
    logfile.write('# object  : %s   \n' % cube.e3d_data_header["OBJECT"])
    logfile.write('# airmass : %.3f \n' % cube.e3d_data_header["AIRMASS"])
    logfile.write('# efftime : %.3f \n' % cube.e3d_data_header["EFFTIME"])

    # Global parameters
    # lmin  lmax  delta +/- ddelta  ...  alphaN +/- dalphaN chi2|RSS
    names = ['delta', 'theta', 'xc', 'yc', 'xy'] + \
            ['ell%d' % d for d in xrange(ellDeg + 1)] + \
            ['alpha%d' % d for d in xrange(alphaDeg + 1)]
    labels = '# lmin  lmax' + \
        '  '.join('%8s +/- d%-8s' % (n, n) for n in names)
    if cube.var is None:        # Least-square fit: Residual Sum of Squares
        labels += '        RSS\n'
    else:                       # Chi2 fit: true chi2
        labels += '        chi2\n'
    logfile.write(labels)
    fmt = '%6.0f  %6.0f  ' + \
          '  '.join(
              ["%10.4g"] * ((5 + (ellDeg + 1) + (alphaDeg + 1)) * 2 + 1)) + '\n'
    list3D = [cube.lstart, cube.lend,
              fitpar[0], dfitpar[0],
              fitpar[1], dfitpar[1],
              fitpar[2], dfitpar[2],
              fitpar[3], dfitpar[3],
              fitpar[4], dfitpar[4]]
    for i in xrange(ellDeg + 1):   # Ellipticity coefficiens
        list3D += [fitpar[5 + i], dfitpar[5 + i]]
    for i in xrange(alphaDeg + 1):  # Alpha coefficients
        list3D += [fitpar[6 + ellDeg + i], dfitpar[6 + ellDeg + i]]
    list3D += [chi2]             # chi2|RSS
    logfile.write(fmt % tuple(list3D))

    # Metaslice parameters
    # lbda  I -/- dI  sky0 +/- dsky0  sky1 +/- dsky1  ...  chi2|RSS
    npar_psf = 7 + ellDeg + alphaDeg
    if opts.skyDeg == -2:       # Step background
        npar_sky = 2
    else:                       # Polynomial background (or none)
        npar_sky = (opts.skyDeg + 1) * (opts.skyDeg + 2) / 2

    names = ['I'] + ['sky%d' % d for d in range(npar_sky)]
    labels = '# lbda  ' + '  '.join('%8s +/- d%-8s' % (n, n) for n in names)
    if cube.var is None:        # Least-square fit: compute Res. Sum of Squares
        labels += '        RSS\n'
    else:                       # Chi2 fit: compute chi2 per slice
        labels += '        chi2\n'
    logfile.write(labels)
    fmt = '%6.0f  ' + '  '.join(["%10.4g"] * ((1 + npar_sky) * 2 + 1)) + '\n'
    for n in xrange(cube.nslice):       # Loop over metaslices
        # Wavelength, intensity and error on intensity
        list2D = [cube.lbda[n], fitpar[npar_psf + n], dfitpar[npar_psf + n]]
        for i in xrange(npar_sky):  # Add background parameters
            list2D.extend([fitpar[npar_psf + cube.nslice + n * npar_sky + i],
                           dfitpar[npar_psf + cube.nslice + n * npar_sky + i]])
        # Compute chi2|RSS
        chi2 = N.nan_to_num((cube.slice2d(n, coord='p') -
                             cube_fit.slice2d(n, coord='p')) ** 2)
        if cube.var is not None:    # chi2: divide by variance
            chi2 /= cube.slice2d(n, coord='p', var=True)
        list2D += [chi2.sum()]      # Slice chi2|RSS
        logfile.write(fmt % tuple(list2D))

    logfile.close()


def fill_header(hdr, psf, param, adr, cube, opts, chi2,
                seeing, posprior, fluxes):
    """Fill header *hdr* with PSF fit-related keywords."""

    # Convert reference position from lmid = (lmin+lmax)/2 to LbdaRef
    lmin, lmax = cube.lstart, cube.lend   # 1st and last meta-slice wavelength
    xref, yref = adr.refract(param[2], param[3], LbdaRef, unit=cube.spxSize)
    print_msg("Ref. position [%.0f A]: %+.2f x %+.2f spx" %
              (LbdaRef, xref, yref), 0)

    # "[short|long], classic[-powerlaw]" or "[long|short] [blue|red],
    # chromatic"
    psfname = ', '.join((psf.name, psf.model))

    # Convert polynomial coeffs from lr = (2*lbda - (lmin+lmax))/(lmax-lmin)
    # to lr~ = lbda/LbdaRef - 1 = a + b*lr
    a = (lmin + lmax) / (2. * LbdaRef) - 1
    b = (lmax - lmin) / (2. * LbdaRef)
    if opts.ellDeg:                     # Beyond a simple constant
        c_ell = libES.polyConvert(param[5:6 + opts.ellDeg], trans=(a, b))
    else:
        c_ell = param[5:6]              # Simple constant
    if opts.alphaDeg and not psfname.endswith('powerlaw'):
        c_alp = libES.polyConvert(
            param[6 + opts.ellDeg:7 + opts.ellDeg + opts.alphaDeg],
            trans=(a, b))
    else:
        c_alp = param[6 + opts.ellDeg:7 + opts.ellDeg + opts.alphaDeg]

    hdr['ES_VERS'] = __version__
    hdr['ES_CUBE'] = (opts.input, 'Input cube')
    hdr['ES_LREF'] = (LbdaRef, 'Lambda ref. [A]')
    hdr['ES_SDEG'] = (opts.skyDeg, 'Polynomial bkgnd degree')
    hdr['ES_CHI2'] = (chi2, 'Chi2|RSS of 3D-fit')
    hdr['ES_AIRM'] = (adr.get_airmass(), 'Effective airmass')
    hdr['ES_PARAN'] = (adr.get_parangle(), 'Effective parangle [deg]')
    hdr['ES_XC'] = (xref, 'xc @lbdaRef [spx]')
    hdr['ES_YC'] = (yref, 'yc @lbdaRef [spx]')
    hdr['ES_XY'] = (param[4], 'XY coeff.')
    hdr['ES_LMIN'] = (lmin, 'Meta-slices minimum lambda')
    hdr['ES_LMAX'] = (lmax, 'Meta-slices maximum lambda')

    for i in xrange(opts.ellDeg + 1):
        hdr['ES_E%i' % i] = (c_ell[i], 'Y2 coeff. e%d' % i)
    for i in xrange(opts.alphaDeg + 1):
        hdr['ES_A%i' % i] = (c_alp[i], 'Alpha coeff. a%d' % i)

    hdr['ES_METH'] = (opts.method, 'Extraction method')
    hdr['ES_PSF'] = (psfname, 'PSF model name')
    hdr['ES_SUB'] = (psf.subsampling, 'PSF subsampling')
    if opts.method.endswith('aperture'):
        hdr['ES_APRAD'] = (opts.radius, 'Aperture radius [" or sigma]')

    tflux, sflux = fluxes       # Total point-source and sky fluxes
    hdr['ES_TFLUX'] = (tflux, 'Total point-source flux')
    if sflux:
        hdr['ES_SFLUX'] = (sflux, 'Total sky flux/arcsec^2')

    hdr['SEEING'] = (seeing, 'Estimated seeing @lbdaRef ["] (extract_star)')

    if opts.usePriors:
        hdr['ES_PRIOR'] = (opts.usePriors, 'PSF prior hyper-scale')
        if opts.seeingPrior is not None:
            hdr['ES_PRISE'] = (opts.seeingPrior, 'Seeing prior [arcsec]')
        if posprior is not None:
            if opts.positionPrior.lower() == 'cubefit':  # refer to cubefit positions
                hdr['ES_PRIXY'] = ("cubefit",
                                   "Use CBFT_SN[X|Y] as priors on position")
            elif opts.positionPrior.upper() == 'DDT':  # refer to DDT positions
                hdr['ES_PRIXY'] = ("DDT",
                                   "Use DDT[X|Y]P as priors on position")
            else:               # Use explicit position prior
                hdr['ES_PRIXY'] = ("%+.2f,%+.2f" % tuple(posprior),
                                   "Priors on position")
    if opts.psf3Dconstraints:
        for i, constraint in enumerate(opts.psf3Dconstraints):
            hdr['ES_BND%d' % (i + 1)] = (constraint, "Constraint on 3D-PSF")


def setPSF3Dconstraints(psfConstraints, params, bounds):
    """
    Decipher psf3Dconstraints=[constraint] option and set initial
    guess params and/or bounds accordingly. Each constraint is a
    string 'n:val' (strict constraint) or 'n:val1,val2' (loose
    constraint), for n=0 (delta), 1 (theta), 2,3 (position), 4 (xy),
    5...6+ellDeg (ellipticity polynomial coefficients) and 7+ellDeg
    ... 8+ellDeg+alphaDeg (alpha polynomial coefficients).
    """

    for psfConstraint in psfConstraints:
        try:
            n, constraintStr = psfConstraint.split(':')
            n = int(n)
            vals = map(float, constraintStr.split(','))
            assert len(vals) in (1, 2)
        except (ValueError, AssertionError):
            print("WARNING: Cannot decipher constraint '%s', discarded" %
                  psfConstraints)
            continue
        else:
            if len(vals) == 1:  # Strict constraint: param = val
                val = vals[0]
                params[n] = val
                bounds[n] = [val, val]
                print("WARNING: Forcing PSF param[%d] to %f" % (n, val))

            else:               # Loose constraint: vmin <= param <= vmax
                vmin, vmax = sorted(vals)
                params[n] = min(max(params[n], vmin), vmax)
                bounds[n] = [vmin, vmax]
                print("WARNING: Constraining PSF param[%d] in %f,%f" %
                      (n, vmin, vmax))

# ########## MAIN ##############################

def extract_star():

    import optparse

    # Options ================================================================

    usage = "[%prog] [options] incube.fits"

    parser = optparse.OptionParser(usage, version=__version__)

    parser.add_option("-i", "--in", type=str, dest="input",
                      help="Input datacube (or use argument)")
    parser.add_option("-o", "--out", type=str,
                      help="Output point source spectrum")
    parser.add_option("-s", "--sky", type=str,
                      help="Output sky spectrum")

    # PSF parameters
    parser.add_option("-S", "--skyDeg", type=int,
                      help="Sky polynomial background degree "
                      "(-1: none, -2: step) [%default]",
                      default=0)
    parser.add_option("-A", "--alphaDeg", type=int,
                      help="Alpha polynomial degree [%default]",
                      default=2)
    parser.add_option("-E", "--ellDeg", type=int,
                      help="Ellipticity polynomial degree [%default]",
                      default=0)

    # PSF model
    parser.add_option("--psf",
                      choices=('classic', 'classic-powerlaw', 'chromatic'),
                      help="PSF model "
                      "(classic[-powerlaw]|chromatic) [%default]",
                      default='classic-powerlaw')

    # Extraction method and parameters
    parser.add_option("-N", "--nmeta", type=int,
                      help="Number of chromatic meta-slices [%default]",
                      default=12)
    parser.add_option("--subsampling", type=int,
                      help="Spaxel subsampling [%default]",
                      default=3)

    parser.add_option("-m", "--method",
                      choices=('psf', 'optimal', 'aperture', 'subaperture'),
                      help="Extraction method "
                      "(psf|optimal|[sub]aperture) [%default]",
                      default="psf")
    parser.add_option("-r", "--radius", type=float,
                      help="Aperture radius for non-PSF extraction "
                           "(>0: in \", <0: in seeing sigma) [%default]",
                      default=-5.)
    parser.add_option("-L", "--leastSquares",
                      dest="chi2fit", action="store_false",
                      help="Least-square fit [default is a chi2 fit]",
                      default=True)

    # Plotting
    parser.add_option("-g", "--graph",
                      choices=('png', 'eps', 'pdf', 'svg', 'pylab'),
                      help="Graphic output format (png,eps,pdf,svg,pylab)")
    parser.add_option("-p", "--plot", action='store_true',
                      help="Plot flag (='-g pylab')")

    # Covariance management
    parser.add_option("-V", "--covariance", action='store_true',
                      help="Compute and store covariance matrix in extension")

    # Priors
    parser.add_option("--usePriors", type=float,
                      help="PSF prior hyper-scale, or 0 for none "
                      "(req. powerlaw-PSF) [%default]",
                      default=0.)
    parser.add_option("--seeingPrior", type=float,
                      help="Seeing prior (from Exposure.Seeing) [\"]")
    parser.add_option("--positionPrior", type=str,
                      help="Position prior ('lref,x,y'|'DDT'|'cubefit')")

    # Expert options
    parser.add_option("--no3Dfit", action='store_true',
                      help="Do not perform final 3D-fit")
    parser.add_option("--keepmodel", action='store_true',
                      help="Store meta-slices and adjusted model in 3D cubes")
    parser.add_option("--psf3Dconstraints", type=str, action='append',
                      help="Constraints on PSF parameters (n:val,[val])")

    # Debug options
    parser.add_option("-v", "--verbosity", type=int,
                      help="Verbosity level (<0: quiet) [%default]",
                      default=0)
    parser.add_option("-f", "--file", type=str, dest="log2D",
                      help="2D adjustment logfile name")
    parser.add_option("-F", "--File", type=str, dest="log3D",
                      help="3D adjustment logfile name")
    parser.add_option("--ignorePertinenceTests", action='store_true',
                      # help=optparse.SUPPRESS_HELP
                      help="Ignore tests on PSF pertinence (but DON'T!)")

    # Production options
    parser.add_option("--accountant",
                      help="Accountant output YAML file")

    opts, args = parser.parse_args()
    if not opts.input:
        if args:
            opts.input = args[0]
        else:
            parser.error("No input datacube specified.")

    if opts.graph:
        opts.plot = True
    elif opts.plot:
        opts.graph = 'pylab'

    if opts.skyDeg < -2:
        opts.skyDeg = -1        # No sky background
        if opts.sky:
            print("WARNING: Cannot extract sky spectrum in no-sky mode.")

    if opts.verbosity <= 0:
        N.seterr(all='ignore')

    if opts.usePriors:
        if opts.usePriors < 0:
            parser.error("Prior scale (--usePriors) must be positive.")
        if not opts.psf.endswith('powerlaw'):
            parser.error("Priors implemented for 'powerlaw' PSF only.")
        if opts.alphaDeg != 2 or opts.ellDeg != 0:
            parser.error("Priors mode requires '--alphaDeg 2 --ellDeg 0'.")

    if opts.seeingPrior and not opts.usePriors:
        parser.error("Seeing prior requires prior usage (--usePriors > 0).")
    if opts.positionPrior and not opts.usePriors:
        parser.error("Position prior requires prior usage (--usePriors > 0).")

    # Input datacube ==========================================================

    print("Opening datacube %s" % opts.input)

    # The pySNIFS e3d_data_header dictionary is not enough for later
    # updates in fill_header, which requires a *true* pyfits header.

    try:
        try:                                    # Try to read a Euro3D cube
            inhdr = F.getheader(opts.input, 1)  # 1st extension
            full_cube = pySNIFS.SNIFS_cube(e3d_file=opts.input)
            isE3D = True
        except ValueError:                      # Try to read a 3D FITS cube
            inhdr = F.getheader(opts.input, 0)  # Primary extension
            full_cube = pySNIFS.SNIFS_cube(fits3d_file=opts.input)
            isE3D = False
    except IOError:
        parser.error("Cannot access file '%s'" % opts.input)
    full_cube.flag_nans(name='input cube')
    step = full_cube.lstep

    print_msg("Cube %s [%s]: %d slices [%.2f-%.2f], %d spaxels" %
              (os.path.basename(opts.input), 'E3D' if isE3D else '3D',
               full_cube.nslice,
               full_cube.lbda[0], full_cube.lbda[-1], full_cube.nlens), 1)

    objname = inhdr.get('OBJECT', 'Unknown')
    efftime = inhdr['EFFTIME']            # [s]
    airmass = inhdr['AIRMASS']
    try:
        parangle = inhdr['PARANG']        # Sky parallactic angle [deg]
    except KeyError:                      # Not in original headers
        print("WARNING: Computing PARANG "
              "from header ALTITUDE, AZIMUTH and LATITUDE.")
        _, inhdr['PARANG'] = libES.estimate_zdpar(inhdr)  # [deg]

    channel = inhdr['CHANNEL'][0].upper()  # 'B' or 'R'
    # Include validity tests and defaults
    pressure, temp = libES.read_PT(inhdr)

    ellDeg = opts.ellDeg
    alphaDeg = opts.alphaDeg
    npar_psf = 7 + ellDeg + alphaDeg

    skyDeg = opts.skyDeg
    if skyDeg == -2:            # Step background
        npar_sky = 2
    else:                       # Polynomial background (or none)
        npar_sky = (skyDeg + 1) * (skyDeg + 2) / 2
    hasSky = skyDeg != -1       # Sky component

    # Test channel and set default output name
    if channel not in ('B', 'R'):
        parser.error(
            "Input datacube %s has no valid CHANNEL keyword (%s)" %
            (opts.input, channel))
    if not opts.out:                    # Default output
        opts.out = 'spec_%s.fits' % (channel)

    # Select the PSF
    if opts.psf == 'chromatic':         # Includes chromatic correlations
        # Chromatic parameter description (short or long, red or blue)
        if (efftime > 12.) and (channel == 'B'):
            psfFn = libES.LongBlue_ExposurePSF
        elif (efftime > 12.) and (channel == 'R'):
            psfFn = libES.LongRed_ExposurePSF
        elif (efftime <= 12.) and (channel == 'B'):
            psfFn = libES.ShortBlue_ExposurePSF
        elif (efftime <= 12.) and (channel == 'R'):
            psfFn = libES.ShortRed_ExposurePSF
    elif opts.psf.startswith('classic'):  # Achromatic correlations
        # Classical parameter description (short or long)
        psfFn = libES.Long_ExposurePSF if (efftime > 12.) \
            else libES.Short_ExposurePSF
        if opts.psf.endswith('powerlaw'):
            psfFn.model += '-powerlaw'
    else:
        parser.error("Invalid PSF model '%s'" % opts.psf)

    # Sub-sampling
    psfFn.subsampling = opts.subsampling

    print("  Object: %s, Efftime: %.1fs, Airmass: %.2f" %
          (objname, efftime, airmass))
    print("  PSF: '%s', sub-sampled x%d" %
          (', '.join((psfFn.model, psfFn.name)), psfFn.subsampling))
    if opts.skyDeg > 0:
        print("  Sky: polynomial, degree %d" % opts.skyDeg)
    elif opts.skyDeg == 0:
        print("  Sky: uniform")
    elif opts.skyDeg == -1:
        print("  Sky: none")
    elif opts.skyDeg == -2:
        print("  Sky: uniform + step in J at %s" % (libES.STEPJ,))
    else:
        parser.error("Invalid sky degree '%d'" % opts.skyDeg)

    # Accounting
    if opts.accountant:
        try:
            from libRecord import Accountant
        except ImportError:
            print("WARNING: libRecord is not accessible, accounting disabled")
        else:
            import atexit

            accountant = Accountant(opts.accountant, opts.out)
            print(accountant)
            atexit.register(accountant.finalize)
    else:
        accountant = None

    # 2D-model fitting ========================================================

    # Meta-slice definition (min,max,step [px]) ------------------------------

    slices = metaslice(full_cube.nslice, opts.nmeta, trim=10)
    print("  Channel: '%s', extracting slices: %s" % (channel, slices))

    if isE3D:
        meta_cube = pySNIFS.SNIFS_cube(e3d_file=opts.input, slices=slices)
    else:
        meta_cube = pySNIFS.SNIFS_cube(fits3d_file=opts.input, slices=slices)
    meta_cube.flag_nans(name='meta-cube')
    meta_cube.x = meta_cube.i - 7       # From I,J to spx coords
    meta_cube.y = meta_cube.j - 7
    spxSize = meta_cube.spxSize
    nmeta = meta_cube.nslice

    print_msg("  Meta-slices before selection: %d from %.2f to %.2f by %.2f A" %
              (nmeta, meta_cube.lstart, meta_cube.lend, meta_cube.lstep), 0)

    # Normalisation of the signal and variance in order to avoid numerical
    # problems with too small numbers
    norm = ( N.abs(meta_cube.data.mean(axis=None)) +
             N.abs(meta_cube.data.mean(axis=-1)) ).reshape(-1, 1)  # (nmeta, 1)
    print_msg("  Meta-slice normalization (|mean|+|mean_slice|): %s" %
              (norm.squeeze()), 1)
    meta_cube.data /= norm                           # (nmeta,nspx)
    meta_cube.var /= norm ** 2

    if opts.keepmodel:                  # Store meta-slices in 3D-cube
        path, name = os.path.split(opts.out)
        outpsf = os.path.join(path, 'meta_' + name)
        print("Saving meta-slices in 3D-fits cube '%s'..." % outpsf)
        meta_cube.WR_3d_fits(outpsf)

    # 2D-fit priors ------------------------------

    # Mean wavelength
    lmid = (meta_cube.lstart + meta_cube.lend) / 2

    # Initial ADR parameters
    if opts.usePriors:
        delta, theta = libES.Hyper_PSF3D_PL.predict_adr_params(inhdr)
        adr = TA.ADR(pressure, temp, lref=lmid, delta=delta, theta=theta)
    else:
        adr = TA.ADR(pressure, temp, lref=lmid,
                     airmass=airmass, parangle=parangle + DTHETA[channel])
    print_msg('  ' + str(adr), 1)

    # Priors on position
    posPrior = None
    if opts.positionPrior:
        if opts.positionPrior.lower() == 'cubefit':
            cflxy = libES.read_cubefit_pos(inhdr)  # lref, x, y
            print_msg(
                "  Cubefit-predicted position [%.0f A]: %+.2f x %+.2f spx" %
                cflxy, 0)
            posPrior = adr.refract(   # Back-propagate to ref. wavelength lmid
                cflxy[1], cflxy[2], cflxy[0],
                backward=True, unit=spxSize)  # x,y
        elif opts.positionPrior.upper() == 'DDT':
            ddtlxy = libES.read_DDTpos(inhdr)  # lref, x, y
            print_msg(
                "  DDT-predicted position [%.0f A]: %+.2f x %+.2f spx" %
                ddtlxy, 0)
            posPrior = adr.refract(   # Back-propagate to ref. wavelength lmid
                ddtlxy[1], ddtlxy[2], ddtlxy[0],
                backward=True, unit=spxSize)  # x,y
        else:
            try:
                lxy = _, _, _ = tuple(
                    float(val) for val in opts.positionPrior.split(',') )
            except ValueError as err:
                parser.error(
                    "Cannot parse position prior '%s'" % opts.positionPrior)
            else:
                print_msg(
                    "  Predicted position [%.0f A]: %+.2f x %+.2f spx" % lxy, 1)
                posPrior = adr.refract(   # Back-propagate to ref. wavelength lmid
                    lxy[1], lxy[2], lxy[0],
                    backward=True, unit=spxSize)  # x,y
        print_msg("  Prior on position [%.0f A]: %+.2f x %+.2f spx" %
                  (lmid, posPrior[0], posPrior[1]), 0)

    # 2D-fit ------------------------------

    print("Meta-slice 2D-fitting (%s)..." %
          ('chi2' if opts.chi2fit else 'least-squares'))
    params, chi2s, dparams = libES.fit_metaslices(
        meta_cube, psfFn, skyDeg=skyDeg, chi2fit=opts.chi2fit,
        scalePriors=opts.usePriors,
        seeingPrior=opts.seeingPrior if opts.usePriors else None,
        posPrior=posPrior if opts.usePriors else None,
        airmass=airmass, verbosity=opts.verbosity)
    print_msg("", 1)

    params = params.T                         # (nparam,nslice)
    xc_vec, yc_vec = params[2:4]              # PSF centroid position
    xy_vec, ell_vec, alpha_vec = params[4:7]  # PSF shape parameters
    int_vec = params[7]                       # PSF intensity
    if hasSky:
        sky_vec = params[8:]                  # Background parameters

    # Save 2D adjusted parameter file ------------------------------

    if opts.log2D:
        print("Producing 2D adjusted parameter logfile %s..." % opts.log2D)
        create_2Dlog(opts, meta_cube, params, dparams, chi2s)

    # 3D-model fitting ========================================================

    print("Datacube 3D-fitting (%s)..." %
          ('chi2' if opts.chi2fit else 'least-squares'))

    # Initial guesses ------------------------------

    # 0) ADR parameters
    delta0 = adr.delta           # ADR power = tan(zenithal distance)
    theta0 = adr.theta           # ADR angle = parallactic angle [rad]
    print_msg("  ADR guess: delta=%.2f (airmass=%.2f), theta=%.1f deg" %
              (delta0, adr.get_airmass(), theta0 * TA.RAD2DEG), 1)

    # 1) Reference position
    # Convert meta-slice centroids to position at ref. lbda, and clip around
    # median position, using effective parangle including MLA tilt
    xmids, ymids = adr.refract(   # Back-propagate positions to lmid wavelength
        xc_vec, yc_vec, meta_cube.lbda, backward=True, unit=spxSize)
    xmids = N.atleast_1d(xmids)  # Some dim could be squeezed in adr.refract
    ymids = N.atleast_1d(ymids)
    valid = chi2s > 0                   # Discard unfitted slices
    xmid = N.median(xmids[valid])       # Robust to outliers
    ymid = N.median(ymids[valid])
    r = N.hypot(xmids - xmid, ymids - ymid)
    rmax = 4.4478 * N.median(r[valid])  # Robust to outliers 3*1.4826
    good = valid & (r <= rmax)          # Valid fit and reasonable position
    bad = valid & (r > rmax)            # Valid fit but discarded position
    if bad.any():
        print("WARNING: %d metaslices discarded after ADR selection" %
              (len(N.nonzero(bad))))

    if opts.positionPrior:              # Use prior on position
        xc, yc = posPrior
    elif good.any():
        print_msg("%d/%d centroids found within %.2f spx of (%.2f,%.2f)" %
                  (len(xmids[good]), len(xmids), rmax, xmid, ymid), 1)
        xc, yc = xmids[good].mean(), ymids[good].mean()  # Position at lmid
    else:
        raise ValueError('No position initial guess')

    if not good.all():                  # Invalid slices + discarded centroids
        print("%d/%d centroid positions discarded for initial guess" %
              (len(xc_vec[~good]), nmeta))
        if len(xc_vec[good]) <= ellDeg + 1 and not opts.usePriors:
            raise ValueError('Not enough points for ellipticity initial guess')
        if len(xc_vec[good]) <= alphaDeg + 1 and not opts.usePriors:
            raise ValueError('Not enough points for alpha initial guess')

    print_msg("  Ref. position guess [%.0f A]: %+.2f x %+.2f spx" %
              (lmid, xc, yc), 1)

    # 2) Other parameters
    lbda_rel = libES.chebNorm(
        meta_cube.lbda, meta_cube.lstart, meta_cube.lend)  # in [-1,1]

    if opts.usePriors:
        # Use priors: predict shape parameters
        xy = 0.                         # Safe bet
        guessEllCoeffs = [libES.Hyper_PSF3D_PL.predict_y2_param(inhdr)]
    else:
        xy = N.median(xy_vec[good])
        # Polynomial-fit with 3-MAD clipping
        polEll = pySNIFS.fit_poly(ell_vec[good], 3, ellDeg, lbda_rel[good])
        guessEllCoeffs = polEll.coeffs[::-1]

    if not opts.psf.endswith('powerlaw'):  # Polynomial expansion
        # Polynomial-fit with 3-MAD clipping
        polAlpha = pySNIFS.fit_poly(
            alpha_vec[good], 3, alphaDeg, lbda_rel[good])
        guessAlphaCoeffs = polAlpha.coeffs[::-1]
    else:                                 # Power-law expansion
        if opts.seeingPrior:
            # Use PSF priors: predict parameters from seeing prior
            guessAlphaCoeffs = libES.Hyper_PSF3D_PL.predict_alpha_coeffs(
                opts.seeingPrior, channel)
        else:
            guessAlphaCoeffs = libES.powerLawFit(
                meta_cube.lbda[good] / LbdaRef, alpha_vec[good], alphaDeg)

    # Filling in the guess parameter arrays (px) and bounds arrays (bx)
    p1 = [None] * (npar_psf + nmeta)
    p1[:5] = [delta0, theta0, xc, yc, xy]
    p1[5:6 + ellDeg] = guessEllCoeffs
    p1[6 + ellDeg:npar_psf] = guessAlphaCoeffs
    p1[npar_psf:npar_psf + nmeta] = int_vec.tolist()  # Intensities

    # Bounds ------------------------------

    if opts.no3Dfit:              # Fix all parameters but intensities
        print("WARNING: no 3D PSF-fit.")
        # This mode completely discards 3D fit. In pratice, a 3D-fit
        # is still performed on intensities, just to be coherent w/
        # the remaining of the code.
        b1 = [[delta0, delta0],         # delta
              [theta0, theta0],         # theta [rad]
              [xc, xc],                 # xc (position at lmid)
              [yc, yc],                 # yc
              [xy, xy]]                 # xy
        for coeff in p1[5:6 + ellDeg] + p1[6 + ellDeg:npar_psf]:
            b1 += [[coeff, coeff]]      # ell and alpha coeff.
    else:
        b1 = [[None, None],             # delta
              [None, None],             # theta
              [None, None],             # xc
              [None, None],             # yc
              [None, None]]             # xy
        b1 += [[0, None]] + [[None, None]] * ellDeg  # ell0 > 0
        if not opts.psf.endswith('powerlaw'):
            b1 += [[0, None]] + [[None, None]] * alphaDeg  # a[0] > 0
        else:
            b1 += [[None, None]] * alphaDeg + [[0, None]]  # a[-1] > 0
    b1 += [[0, None]] * nmeta            # Intensities

    if opts.psf3Dconstraints:           # Read and set constraints from option
        setPSF3Dconstraints(opts.psf3Dconstraints, p1, b1)

    psfCtes = [spxSize, lmid, alphaDeg, ellDeg]
    func = ['%s;%s' %
            (psfFn.name, ','.join(str(p) for p in psfCtes))]  # PSF
    param = [p1]
    bounds = [b1]

    myfunc = {psfFn.name: psfFn}

    if hasSky:
        p2 = N.ravel(sky_vec.T)
        if skyDeg == -2:        # Step along J-direction
            b2 = ([[0, None], [None, None]]) * nmeta  # Mean > 0 and diff.
            func += ['stepJ;%d,%d' % libES.STEPJ]
            myfunc.update(((libES.StepJ.name, libES.StepJ),))
        else:                   # Add polynomial background
            b2 = ([[0, None]] + [[None, None]] * (npar_sky - 1)) * nmeta
            func += ['poly2D;%d' % skyDeg]
        param += [p2]
        bounds += [b2]

    print_msg("  Adjusted parameters: delta,theta,xc,yc,xy,"
              "%d ellCoeffs,%d alphaCoeffs,%d intensities, %d bkgndCoeffs" %
              (ellDeg + 1, alphaDeg + 1, nmeta, npar_sky * nmeta), 2)
    print_msg("  Initial guess [PSF]: %s" % p1[:npar_psf], 2)
    print_msg("  Initial guess [Intensities]: %s" %
              p1[npar_psf:npar_psf + nmeta], 3)
    if hasSky:
        print_msg("  Initial guess [Bkgnd]: %s" % p2, 3)

    # Actual fit ------------------------------

    # Chi2 vs. Least-square fit
    if not opts.chi2fit:
        meta_cube.var = None    # Will be handled by pySNIFS_fit.model

    # Hyper-term
    hyper = {}
    if opts.usePriors:
        hterm = libES.Hyper_PSF3D_PL(psfCtes, inhdr,
                                     seeing=opts.seeingPrior, position=posPrior,
                                     scale=opts.usePriors)
        print_msg(str(hterm), 1)
        hyper = {psfFn.name: hterm}     # Hyper dict {fname:hyper}

    # Parameter names
    parnames = ['delta', 'theta', 'x0', 'y0', 'xy'] + \
               [ 'E%02d' % i for i in range(ellDeg + 1) ] + \
               [ 'A%02d' % i for i in range(alphaDeg + 1) ] + \
               [ 'I%02d' % (i + 1) for i in range(nmeta) ] + \
               ['B%02d_%02d' % (i + 1, j)
                for i in range(nmeta) for j in range(npar_sky)]

    # Instantiate the model and perform the 3D-fit (fmin_tnc)
    data_model = pySNIFS_fit.model(data=meta_cube, func=func,
                                   param=param, bounds=bounds,
                                   myfunc=myfunc,
                                   hyper=hyper)

    if opts.verbosity >= 4:
        print("Gradient checks:")        # Include hyper-terms if any
        data_model.check_grad()

    # Minimization: default method is 'TNC'
    data_model.minimize(verbose=(opts.verbosity >= 2), tol=1e-6,
                        options={'maxiter': 400}, method='TNC')
    if not data_model.success:  # Try with 'L-BFGS-B'
        print("WARNING: 3D-PSF fit did not converge w/ TNC minimizer " \
              "(status=%d: %s), trying again with L-BFGS-B minimizer" % \
              (data_model.status, data_model.res.message))
        data_model.minimize(verbose=(opts.verbosity >= 2), tol=1e-6,
                            options={'maxiter': 400}, method='L-BFGS-B')

    # Print out fit facts
    print(data_model.facts(params=opts.verbosity >= 1, names=parnames))

    if not data_model.success:
        raise ValueError('3D-PSF fit did not converge (status=%d: %s)' %
                         (data_model.status, data_model.res.message))

    # Store guess and fit parameters ------------------------------

    fitpar = data_model.fitpar          # Adjusted parameters
    data_model.khi2 *= data_model.dof   # Restore real chi2 (or RSS)
    chi2 = data_model.khi2              # Total chi2 of 3D-fit
    covpar = data_model.param_cov(fitpar)  # Parameter covariance matrix
    dfitpar = N.sqrt(covpar.diagonal())  # Diag. errors on adjusted parameters

    print_msg("  Fit result: chi2/dof=%.2f/%d" % (chi2, data_model.dof), 1)
    print_msg("  Fit result [PSF param]: %s" % fitpar[:npar_psf], 2)
    print_msg("  Fit result [Intensities]: %s" %
              fitpar[npar_psf:npar_psf + nmeta], 3)
    if hasSky:
        print_msg("  Fit result [Background]: %s" %
                  fitpar[npar_psf + nmeta:], 3)
    if opts.usePriors:
        print_msg("  Hyper-term: h=%f" % hterm.comp(fitpar[:npar_psf]), 1)

    print_msg("  Ref. position fit @%.0f A: %+.2f±%.2f × %+.2f±%.2f spx" %
              (lmid, fitpar[2], dfitpar[2], fitpar[3], dfitpar[3]), 1)
    # Update ADR params
    print_msg("  ADR fit: delta=%.2f±%.2f, theta=%.1f±%.1f deg" %
              (fitpar[0], dfitpar[0],
               fitpar[1] * TA.RAD2DEG, dfitpar[1] * TA.RAD2DEG), 1)
    adr.set_param(delta=fitpar[0], theta=fitpar[1])
    print("  Effective airmass: %.2f" % adr.get_airmass())
    # Estimated seeing (FWHM in arcsec)
    seeing = data_model.func[0].FWHM(fitpar[:npar_psf], LbdaRef) * spxSize
    print('  Seeing estimate @%.0f A: %.2f" FWHM' % (LbdaRef, seeing))

    # Estimated chromatic profiles
    if not opts.psf.endswith('powerlaw'):
        fit_alpha = libES.polyEval(fitpar[6 + ellDeg:npar_psf], lbda_rel)
    else:
        fit_alpha = libES.powerLawEval(
            fitpar[6 + ellDeg:npar_psf], meta_cube.lbda / LbdaRef)
    fit_ell = libES.polyEval(fitpar[5:6 + ellDeg], lbda_rel)

    # Check fit pertinence ------------------------------

    if opts.usePriors:      # Test against priors
        # These are warnings only, since these tests can eventually be
        # performed a posteriori. Note that it would be preferable to perform
        # tests on hyper-term contributions rather than on absolute
        # comparisons.

        # Test position of point-source
        if opts.positionPrior:
            dprior = N.hypot(fitpar[2] - posPrior[0], fitpar[3] - posPrior[1])
            if dprior > MAX_POSITION_PRIOR_OFFSET:
                print("WARNING: " \
                      "Point-source %.2fx%.2f is %.2f spx away " \
                      "from position prior %.2fx%.2f" % \
                      (fitpar[2], fitpar[3], dprior, posPrior[0], posPrior[1]))
                if accountant:
                    accountant.add_warning("ES_PRIOR_POSITION")
        # Tests on seeing
        if opts.seeingPrior:
            fac = (seeing / opts.seeingPrior - 1) * 1e2
            if abs(fac) > MAX_SEEING_PRIOR_OFFSET:
                print("WARNING: " \
                      "Seeing %.2f\" is %+.0f%% away from predicted %.2f\"" % \
                      (seeing, fac, opts.seeingPrior))
                if accountant:
                    accountant.add_warning("ES_PRIOR_SEEING")
        # Tests on ADR parameters
        fac = (adr.get_airmass() / adr.get_airmass(delta0) - 1) * 1e2
        if abs(fac) > MAX_AIRMASS_PRIOR_OFFSET:
            print("WARNING: " \
                  "Airmass %.2f is %+.0f%% away from predicted %.2f" % \
                  (adr.get_airmass(), fac, adr.get_airmass(delta0)))
            if accountant:
                accountant.add_warning("ES_PRIOR_AIRMASS")
        # Rewrap angle difference [rad]
        rewrap = lambda dtheta: (dtheta + N.pi) % (2 * N.pi) - N.pi
        err = rewrap(adr.theta - theta0) * TA.RAD2DEG
        if abs(err) > MAX_PARANG_PRIOR_OFFSET:
            print("WARNING: " \
                  "Parangle %.0fdeg is %+.0fdeg away from predicted %.0fdeg" % \
                  (adr.get_parangle(), err, theta0 * TA.RAD2DEG))
            if accountant:
                accountant.add_warning("ES_PRIOR_PARANGLE")

    if not (abs(fitpar[2]) < MAX_POSITION and abs(fitpar[3]) < MAX_POSITION):
        print("WARNING: Point-source %+.2f x %+.2f mis-centered" % \
              (fitpar[2], fitpar[3]))
        if accountant:
            accountant.add_warning("ES_MIS-CENTERED")

    try:
        # Tests on seeing and airmass
        if not MIN_SEEING < seeing < MAX_SEEING:
            raise ValueError("Unphysical seeing (%.2f\")" % seeing)
        if not 1. <= adr.get_airmass() < MAX_AIRMASS:
            raise ValueError(
                "Unphysical airmass (%.2f)" % adr.get_airmass())
        # Test positivity of alpha and ellipticity
        if fit_alpha.min() < 0:
            raise ValueError(
                "Alpha is negative (%.2f) at %.0f A" %
                (fit_alpha.min(), meta_cube.lbda[fit_alpha.argmin()]))
        if fit_ell.min() < MIN_ELLIPTICITY:
            raise ValueError(
                "Unphysical ellipticity (%.2f) at %.0f A" %
                (fit_ell.min(), meta_cube.lbda[fit_ell.argmin()]))
        if fit_ell.max() > MAX_ELLIPTICITY:
            raise ValueError(
                "Unphysical ellipticity (%.2f) at %.0f A" %
                (fit_ell.max(), meta_cube.lbda[fit_ell.argmax()]))
    except ValueError as nonPertinentException:
        if opts.ignorePertinenceTests:
            sys.stderr.write("ERROR: %s\n" % str(nonPertinentException))
        else:
            raise               # Will reraise input exception

    # Compute point-source and background spectra =============================

    # Compute aperture radius
    if opts.method == 'psf':
        radius = None
        method = 'psf, %s' % ('chi2' if opts.chi2fit else 'least-squares')
    else:
        if opts.radius < 0:     # Aperture radius [sigma]
            radius = -opts.radius * seeing / 2.355  # [arcsec]
            method = '%s r=%.1f sigma=%.2f"' % \
                     (opts.method, -opts.radius, radius)
        else:                   # Aperture radius [arcsec]
            radius = opts.radius        # [arcsec]
            method = '%s r=%.2f"' % (opts.method, radius)
    print("Extracting the point-source spectrum (method=%s)..." % method)
    if not hasSky:
        print("WARNING: No background adjusted.")

    # Spectrum extraction (point-source, sky, etc.)
    lbda, sigspecs, varspecs = libES.extract_specs(
        full_cube, (psfFn, psfCtes, fitpar[:npar_psf]),
        skyDeg=skyDeg, method=opts.method,
        radius=radius, chi2fit=opts.chi2fit,
        verbosity=opts.verbosity)

    if hasSky:        # Convert (mean) sky spectrum to "per arcsec**2"
        sigspecs[:, 1] /= spxSize ** 2
        varspecs[:, 1] /= spxSize ** 4
        if skyDeg == -2:        # (lower - upper) differential sky spectrum
            sigspecs[:, 2] /= 0.5 * spxSize ** 2
            varspecs[:, 2] /= 0.5 * spxSize ** 4

    # Full covariance matrix of point-source spectrum
    if opts.covariance:
        print("Computing point-source spectrum covariance...")
        covspec = spec_covariance(full_cube,
                                  (psfFn, psfCtes, fitpar[:npar_psf]), skyDeg,
                                  covpar[:npar_psf, :npar_psf])

        # Add diagonal contribution from signal noise
        covspec += N.diag(varspecs[:, 0])
        # Extract diagonal term
        varspecs[:, 0] = covspec.diagonal()

    # Creating a standard SNIFS cube with the adjusted data
    # We cannot directly use data_model.evalfit() because 1. we want
    # to keep psf and bkg separated; 2. cube_fit will always have 225
    # spx, data_model.evalfit() might have less.  But in the end,
    # psf+bkg ~= data_model.evalfit()
    cube_fit = pySNIFS.SNIFS_cube(lbda=meta_cube.lbda)  # Always 225 spx
    cube_fit.x = cube_fit.i - 7                        # x in spaxel
    cube_fit.y = cube_fit.j - 7                        # y in spaxel

    psf_model = psfFn(psfCtes, cube=cube_fit)
    psf = psf_model.comp(fitpar[:psf_model.npar])
    cube_fit.data = psf.copy()

    if skyDeg >= 0:             # Polynomial background
        bkg_model = pySNIFS_fit.poly2D(skyDeg, cube_fit)
    elif skyDeg == -2:          # Step background
        bkg_model = libES.StepJ(libES.STEPJ, cube_fit)
    bkg = bkg_model.comp(
        fitpar[psf_model.npar:psf_model.npar + bkg_model.npar])
    cube_fit.data += bkg

    # Update header ------------------------------

    tflux = sigspecs[:, 0].sum()      # Total point-source flux
    if hasSky:
        sflux = sigspecs[:, 1].sum()  # Total sky flux (per arcsec**2)
    else:
        sflux = 0                     # Not stored

    fill_header(inhdr, psfFn, fitpar[:npar_psf], adr, meta_cube, opts,
                chi2, seeing, posPrior, (tflux, sflux))

    # Save point-source spectrum ------------------------------

    print("Saving output point-source spectrum to '%s'" % opts.out)

    # Store variance as extension to signal
    star_spec = pySNIFS.spectrum(data=sigspecs[:, 0], var=varspecs[:, 0],
                                 start=lbda[0], step=step)
    if opts.covariance:  # Append covariance directly to pySNIFS.spectrum
        star_spec.cov = covspec
    star_spec.write_fits(opts.out, inhdr)

    # Save background spectrum ------------------------------

    if hasSky:
        if not opts.sky:        # Use default sky spectrum name
            opts.sky = 'sky_%s.fits' % (channel)
        print("Saving output sky spectrum to '%s'" % opts.sky)
        # Store variance as extension to signal
        sky_spec = pySNIFS.spectrum(data=sigspecs[:, 1], var=varspecs[:, 1],
                                    start=lbda[0], step=step)
        sky_spec.write_fits(opts.sky, inhdr)
        # Save differential background spectrum
        if skyDeg == -2 and opts.verbosity >= 1:
            print("Saving differential background spectrum to 'step_%s'" % opts.sky)
            step_spec = pySNIFS.spectrum(data=sigspecs[:, 2], var=varspecs[:, 2],
                                         start=lbda[0], step=step)
            step_spec.write_fits('step_' + opts.sky, inhdr)

    # Save 3D adjusted parameter file ------------------------------

    if opts.log3D:
        print("Producing 3D adjusted parameter logfile %s..." % opts.log3D)
        create_3Dlog(opts, meta_cube, cube_fit, fitpar, dfitpar, chi2)

    # Save adjusted PSF ------------------------------

    if opts.keepmodel:
        path, name = os.path.split(opts.out)
        outpsf = os.path.join(path, 'psf_' + name)
        print("Saving adjusted meta-slice PSF in 3D-fits cube '%s'..." % outpsf)
        cube_fit.WR_3d_fits(outpsf, header=[])  # No header in cube_fit

    # Create output graphics =================================================

    if opts.plot:
        print("Producing output figures [%s]..." % opts.graph)

        import matplotlib as M
        backends = {'png': 'Agg', 'eps': 'PS', 'pdf': 'PDF', 'svg': 'SVG'}
        try:
            M.use(backends[opts.graph.lower()])
            basename = os.path.splitext(opts.out)[0]
            plot1 = os.path.extsep.join((basename + "_plt", opts.graph))
            plot2 = os.path.extsep.join((basename + "_fit1", opts.graph))
            plot3 = os.path.extsep.join((basename + "_fit2", opts.graph))
            plot4 = os.path.extsep.join((basename + "_fit3", opts.graph))
            plot5 = os.path.extsep.join((basename + "_fit4", opts.graph))
            plot6 = os.path.extsep.join((basename + "_fit5", opts.graph))
            plot7 = os.path.extsep.join((basename + "_fit6", opts.graph))
            plot8 = os.path.extsep.join((basename + "_fit7", opts.graph))
        except KeyError:
            opts.graph = 'pylab'
            plot1 = plot2 = plot3 = plot4 = plot5 = plot6 = plot7 = plot8 = ''
        import matplotlib.pyplot as P

        # Non-default colors
        blue = Colors.blue
        red = Colors.red
        green = Colors.green
        orange = Colors.orange
        purple = Colors.purple

        # Plot of the star and sky spectra -----------------------------------

        print_msg("Producing spectra plot %s..." % plot1, 1)

        fig1 = P.figure()

        if hasSky and sky_spec.data.any():
            axS = fig1.add_subplot(3, 1, 1)  # Point-source
            axB = fig1.add_subplot(3, 1, 2)  # Sky
            axN = fig1.add_subplot(3, 1, 3)  # S/N
        else:
            axS = fig1.add_subplot(2, 1, 1)  # Point-source
            axN = fig1.add_subplot(2, 1, 2)  # S/N

        axS.text(0.95, 0.8, os.path.basename(opts.input),
                 fontsize='small', ha='right', transform=axS.transAxes)

        axS.plot(star_spec.x, star_spec.data, blue)
        axS.errorband(star_spec.x, star_spec.data, N.sqrt(star_spec.var),
                      color=blue)
        axN.plot(star_spec.x, star_spec.data / N.sqrt(star_spec.var), blue)

        if hasSky and sky_spec.data.any():
            axB.plot(sky_spec.x, sky_spec.data, green)
            axB.errorband(sky_spec.x, sky_spec.data, N.sqrt(sky_spec.var),
                          color=green)
            axB.set(title=u"Background spectrum (per arcsec²)",
                    xlim=(sky_spec.x[0], sky_spec.x[-1]),
                    xticklabels=[])
            if skyDeg == -2:
                axB.plot(sky_spec.x, sigspecs[:, 2] * 10, red,
                         label=u"Differential ×10")
                axB.errorband(
                    sky_spec.x, sigspecs[:, 2] * 10, N.sqrt(varspecs[:, 2]) * 10,
                    color=red)
                axB.legend(loc='upper right', fontsize='small')
            # Sky S/N
            axN.plot(sky_spec.x, sky_spec.data / N.sqrt(sky_spec.var), green)

        axS.set(title="Point-source spectrum [%s, %s]" % (objname, method),
                xlim=(star_spec.x[0], star_spec.x[-1]), xticklabels=[])
        axN.set(title="Signal/Noise", xlabel=u"Wavelength [Å]",
                xlim=(star_spec.x[0], star_spec.x[-1]))

        fig1.tight_layout()
        if plot1:
            fig1.savefig(plot1)

        # Plot of the fit on each slice --------------------------------------

        print_msg("Producing slice fit plot %s..." % plot2, 1)

        ncol = int(N.floor(N.sqrt(nmeta)))
        nrow = int(N.ceil(nmeta / float(ncol)))

        fig2 = P.figure()
        fig2.suptitle(
            "Slice plots [%s, airmass=%.2f]" % (objname, airmass),
            fontsize='large')

        mod = data_model.evalfit()      # Total model (same nb of spx as cube)

        # Compute PSF & bkgnd models on incomplete cube
        sno = N.sort(meta_cube.no)
        psf2 = psfFn(psfCtes, cube=meta_cube).comp(fitpar[:psf_model.npar])
        if hasSky and sky_spec.data.any():
            if skyDeg == -2:    # Step background
                bkg2 = libES.StepJ(libES.STEPJ, meta_cube).comp(
                    fitpar[psf_model.npar:psf_model.npar + bkg_model.npar])
            else:               # Polynomial background
                bkg2 = pySNIFS_fit.poly2D(skyDeg, meta_cube).comp(
                    fitpar[psf_model.npar:psf_model.npar + bkg_model.npar])

        for i in xrange(nmeta):        # Loop over meta-slices
            data = meta_cube.data[i, :]
            fit = mod[i, :]
            ax = fig2.add_subplot(nrow, ncol, i + 1)
            ax.plot(sno, data, color=blue, ls='-')  # Signal
            if meta_cube.var is not None:
                ax.errorband(
                    sno, data, N.sqrt(meta_cube.var[i, :]), color=blue)
            ax.plot(sno, fit, color=red, ls='-')   # Model
            if hasSky and sky_spec.data.any():
                ax.plot(sno, psf2[i, :], color=green, ls='-')   # PSF alone
                ax.plot(sno, bkg2[i, :], color=orange, ls='-')  # Background
            P.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                   fontsize='xx-small')
            ax.text(0.05, 0.85, u"%.0f Å" % meta_cube.lbda[i],
                    fontsize='x-small', transform=ax.transAxes)

            ax.set_ylim(data.min() / 1.2, data.max() * 1.2)
            ax.set_xlim(-1, 226)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Spaxel #", fontsize='small')
                ax.set_ylabel("Flux", fontsize='small')

        fig2.subplots_adjust(left=0.07, right=0.96, bottom=0.06, top=0.94)
        if plot2:
            fig2.savefig(plot2)

        # Plot of the fit on rows and columns sum ----------------------------

        print_msg("Producing profile plot %s..." % plot3, 1)

        if not opts.covariance:     # Plot fit on rows and columns sum

            fig3 = P.figure()
            fig3.suptitle(
                "Rows and columns [%s, airmass=%.2f]" % (objname, airmass),
                fontsize='large')

            for i in xrange(nmeta):        # Loop over slices
                ax = fig3.add_subplot(nrow, ncol, i + 1)

                # Signal
                sigSlice = meta_cube.slice2d(i, coord='p', NAN=False)
                prof_I = sigSlice.sum(axis=0)  # Sum along rows
                prof_J = sigSlice.sum(axis=1)  # Sum along columns
                # Errors
                if opts.chi2fit:              # Chi2 fit: plot errorbars
                    varSlice = meta_cube.slice2d(
                        i, coord='p', var=True, NAN=False)
                    err_I = N.sqrt(varSlice.sum(axis=0))
                    err_J = N.sqrt(varSlice.sum(axis=1))
                    ax.errorbar(range(len(prof_I)), prof_I, err_I,
                                fmt='o', c=blue, ecolor=blue, ms=3)
                    ax.errorbar(range(len(prof_J)), prof_J, err_J,
                                fmt='^', c=red, ecolor=red, ms=3)
                else:            # Least-square fit
                    ax.plot(range(len(prof_I)), prof_I,
                            marker='o', c=blue, ms=3, ls='None')
                    ax.plot(range(len(prof_J)), prof_J,
                            marker='^', c=red, ms=3, ls='None')
                # Model
                modSlice = cube_fit.slice2d(i, coord='p')
                mod_I = modSlice.sum(axis=0)
                mod_J = modSlice.sum(axis=1)
                ax.plot(mod_I, ls='-', color=blue)
                ax.plot(mod_J, ls='-', color=red)

                P.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                       fontsize='xx-small')
                ax.text(0.05, 0.85, u"%.0f Å" % meta_cube.lbda[i],
                        fontsize='x-small', transform=ax.transAxes)
                if ax.is_last_row() and ax.is_first_col():
                    ax.set_xlabel("I (blue) or J (red)", fontsize='small')
                    ax.set_ylabel("Flux", fontsize='small')

                fig3.subplots_adjust(left=0.06, right=0.96,
                                     bottom=0.06, top=0.95)
        else:                           # Plot correlation matrices

            # Parameter correlation matrix
            corrpar = covpar / N.outer(dfitpar, dfitpar)
            parnames = data_model.func[0].parnames  # PSF param names
            if skyDeg >= 0:    # Add polynomial background param names
                coeffnames = ["00"] + \
                             [ "%d%d" % (d - j, j)
                               for d in range(1, skyDeg + 1)
                               for j in range(d + 1) ]
                parnames += [ "b%02d_%s" % (s + 1, c)
                              for c in coeffnames for s in range(nmeta) ]
            elif skyDeg == -2:
                coeffnames = ["mean", "diff"]
                parnames += [ "b%02d_%s" % (s + 1, c)
                              for c in coeffnames for s in range(nmeta) ]

            assert len(parnames) == corrpar.shape[0]
            # Remove some of the names for clarity
            parnames[npar_psf + 1::2] = [''] * len(parnames[npar_psf + 1::2])

            fig3 = P.figure(figsize=(7, 6))
            ax3 = fig3.add_subplot(1, 1, 1,
                                   title="Parameter correlation matrix")
            im3 = ax3.imshow(N.absolute(corrpar),
                             vmin=1e-3, vmax=1,
                             norm=P.matplotlib.colors.LogNorm(),
                             aspect='equal', origin='upper',
                             interpolation='nearest')
            ax3.set_xticks(range(len(parnames)))
            ax3.set_xticklabels(parnames,
                                va='top', fontsize='x-small', rotation=90)
            ax3.set_yticks(range(len(parnames)))
            ax3.set_yticklabels(parnames,
                                ha='right', fontsize='x-small')

            cb3 = fig3.colorbar(im3, ax=ax3, orientation='vertical')
            cb3.set_label("|Correlation|")

            fig3.tight_layout()

        if plot3:
            fig3.savefig(plot3)

        # Plot of the star center of gravity and adjusted center -------------

        print_msg("Producing ADR plot %s..." % plot4, 1)
        # Guessed and adjusted position at current wavelength
        xguess = xc + delta0 * N.sin(theta0) * psf_model.ADRscale[:, 0]
        yguess = yc - delta0 * N.cos(theta0) * psf_model.ADRscale[:, 0]
        xfit = (fitpar[2] +
                fitpar[0] * N.sin(fitpar[1]) * psf_model.ADRscale[:, 0])
        yfit = (fitpar[3] -
                fitpar[0] * N.cos(fitpar[1]) * psf_model.ADRscale[:, 0])

        fig4 = P.figure()

        ax4c = fig4.add_subplot(2, 1, 1,
                                aspect='equal', adjustable='datalim',
                                xlabel="X center [spx]",
                                ylabel="Y center [spx]",
                                title="ADR plot [%s, airmass=%.2f]" %
                                (objname, airmass))
        ax4a = fig4.add_subplot(2, 2, 3,
                                xlabel=u"Wavelength [Å]",
                                ylabel="X center [spx]")
        ax4b = fig4.add_subplot(2, 2, 4,
                                xlabel=u"Wavelength [Å]",
                                ylabel="Y center [spx]")

        if good.any():
            ax4a.errorbar(meta_cube.lbda[good], xc_vec[good],
                          yerr=dparams[good, 2], fmt=None, ecolor=green)
            ax4a.scatter(meta_cube.lbda[good], xc_vec[good],
                         edgecolors='none', c=meta_cube.lbda[good],
                         cmap=M.cm.jet, zorder=3, label="Fit 2D")
        if bad.any():
            ax4a.plot(meta_cube.lbda[bad], xc_vec[bad],
                      mfc=red, mec=red, marker='.', ls='None', label='_')
        ax4a.plot(meta_cube.lbda, xguess, 'k--',
                  label="Guess 3D" if not opts.positionPrior else "Prior 3D")
        ax4a.plot(meta_cube.lbda, xfit, green, label="Fit 3D")
        P.setp(ax4a.get_xticklabels() + ax4a.get_yticklabels(),
               fontsize='xx-small')
        ax4a.legend(loc='best', fontsize='small', frameon=False)

        if good.any():
            ax4b.errorbar(meta_cube.lbda[good], yc_vec[good],
                          yerr=dparams[good, 3], fmt=None, ecolor=green)
            ax4b.scatter(meta_cube.lbda[good], yc_vec[good], edgecolors='none',
                         c=meta_cube.lbda[good], cmap=M.cm.jet, zorder=3)
        if bad.any():
            ax4b.plot(meta_cube.lbda[bad], yc_vec[bad],
                      marker='.', mfc=red, mec=red, ls='None')
        ax4b.plot(meta_cube.lbda, yfit, green)
        ax4b.plot(meta_cube.lbda, yguess, 'k--')
        P.setp(ax4b.get_xticklabels() + ax4b.get_yticklabels(),
               fontsize='xx-small')

        if valid.any():
            ax4c.errorbar(xc_vec[valid], yc_vec[valid],
                          xerr=dparams[valid, 2], yerr=dparams[valid, 3],
                          fmt=None, ecolor=green)
        if good.any():
            ax4c.scatter(xc_vec[good], yc_vec[good], edgecolors='none',
                         c=meta_cube.lbda[good],
                         cmap=M.cm.jet, zorder=3)
            # Plot position selection process
            ax4c.plot(xmids[good], ymids[good], marker='.',
                      mfc=blue, mec=blue, ls='None')  # Selected ref. positions
        if bad.any():
            ax4c.plot(xmids[bad], ymids[bad], marker='.',
                      mfc=red, mec=red, ls='None')   # Discarded ref. positions
        ax4c.plot((xmid, xc), (ymid, yc), 'k-')
        ax4c.plot(xguess, yguess, 'k--')  # Guess ADR
        ax4c.plot(xfit, yfit, green)      # Adjusted ADR
        ax4c.set_autoscale_on(False)
        ax4c.plot((xc,), (yc,), 'k+')
        ax4c.add_patch(M.patches.Circle((xmid, ymid), radius=rmax,
                                        ec='0.8', fc='None'))  # ADR selection
        ax4c.add_patch(M.patches.Rectangle((-7.5, -7.5), 15, 15,
                                           ec='0.8', lw=2, fc='None'))  # FoV
        txt = u'Guess: x0,y0=%+4.2f,%+4.2f  airmass=%.2f parangle=%+.0f°' % \
              (xc, yc, airmass, theta0 * TA.RAD2DEG)
        txt += u'\nFit: x0,y0=%+4.2f,%+4.2f  airmass=%.2f parangle=%+.0f°' % \
               (fitpar[2], fitpar[3], adr.get_airmass(), adr.get_parangle())
        txtcol = 'k'
        if accountant:
            if accountant.test_warning('ES_PRIOR_POSITION'):
                txt += '\n%s' % accountant.get_warning('ES_PRIOR_POSITION')
                txtcol = Colors.red
            if accountant.test_warning('ES_PRIOR_AIRMASS'):
                txt += '\n%s' % accountant.get_warning('ES_PRIOR_AIRMASS')
                txtcol = Colors.red
            if accountant.test_warning('ES_PRIOR_PARANGLE'):
                txt += '\n%s' % accountant.get_warning('ES_PRIOR_PARANGLE')
                txtcol = Colors.red
        ax4c.text(0.95, 0.8, txt, transform=ax4c.transAxes,
                  fontsize='small', ha='right', color=txtcol)

        fig4.tight_layout()
        if plot4:
            fig4.savefig(plot4)

        # Plot of the other model parameters ---------------------------------

        print_msg("Producing model parameter plot %s..." % plot6, 1)

        guessEll = libES.polyEval(guessEllCoeffs, lbda_rel)
        if not opts.psf.endswith('powerlaw'):
            guessAlpha = libES.polyEval(guessAlphaCoeffs, lbda_rel)
        else:
            guessAlpha = libES.powerLawEval(
                guessAlphaCoeffs, meta_cube.lbda / LbdaRef)

        # err_ell and err_alpha are definitely wrong, and not only
        # because they do not include correlations between parameters!
        err_xy = dfitpar[4]

        def plot_conf_interval(ax, x, y, dy):
            ax.plot(x, y, green, label="Fit 3D")
            if dy is not None:
                ax.errorband(x, y, dy, color=green)

        fig6 = P.figure()

        ax6a = fig6.add_subplot(2, 1, 1,
                                title='Model parameters '
                                '[%s, seeing %.2f" FWHM]' % (objname, seeing),
                                xticklabels=[],
                                ylabel=u'α [spx]')
        ax6b = fig6.add_subplot(4, 1, 3,
                                xticklabels=[],
                                ylabel=u'y² coeff.')
        ax6c = fig6.add_subplot(4, 1, 4,
                                xlabel=u"Wavelength [Å]",
                                ylabel=u'xy coeff.')

        # WARNING: the so-called `xy` parameter is not the PA of the
        # adjusted ellipse, but half the x*y coefficient. Similarly,
        # ell is not the ellipticity, but the y**2 coefficient: x2 +
        # ell*y2 + 2*PA*x*y + ... = 1. One should use quadEllipse for
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
        # elldata = N.array([ quadEllipse(1, q, ell,
        #                                 -(x0 + q*y0), -(ell*y0 + q*x0),
        #                                 x0**2 + ell*y0**2 + 2*q*x0*y0 - 1)
        #                     for x0,y0,ell,q in
        #                     zip(xfit,yfit,fit_ell,[fitpar[4]]*nmeta)])
        # and associated errors.

        if good.any():
            ax6a.errorbar(meta_cube.lbda[good], alpha_vec[good], dparams[good, 6],
                          marker='.',
                          mfc=blue, mec=blue, ecolor=blue, capsize=0, ls='None',
                          label="Fit 2D")
        if bad.any():
            ax6a.plot(meta_cube.lbda[bad], alpha_vec[bad],
                      marker='.', mfc=red, mec=red, ls='None', label="_")
        ax6a.plot(meta_cube.lbda, guessAlpha, 'k--',
                  label="Guess 3D" if not opts.seeingPrior else "Prior 3D")
        # plot_conf_interval(ax6a, meta_cube.lbda, fit_alpha, err_alpha)
        plot_conf_interval(ax6a, meta_cube.lbda, fit_alpha, None)
        txt = 'Guess: %s' % \
              (', '.join([ 'a%d=%.2f' % (i, a)
                           for i, a in enumerate(guessAlphaCoeffs) ]))
        txt += '\nFit: %s' % \
               (', '.join([ 'a%d=%.2f' % (i, a)
                            for i, a in enumerate(fitpar[6 + ellDeg:npar_psf]) ]))
        txtcol = 'k'
        if accountant and accountant.test_warning('ES_PRIOR_SEEING'):
            txt += '\n%s' % accountant.get_warning('ES_PRIOR_SEEING')
            txtcol = Colors.red
        ax6a.text(0.95, 0.8, txt, transform=ax6a.transAxes,
                  fontsize='small', ha='right', color=txtcol)
        ax6a.legend(loc='upper left', fontsize='small', frameon=False)
        P.setp(ax6a.get_yticklabels(), fontsize='x-small')

        if good.any():
            ax6b.errorbar(meta_cube.lbda[good], ell_vec[good], dparams[good, 5],
                          marker='.',
                          mfc=blue, mec=blue, ecolor=blue, capsize=0, ls='None')
        if bad.any():
            ax6b.plot(meta_cube.lbda[bad], ell_vec[bad],
                      marker='.', mfc=red, mec=red, ls='None')
        ax6b.plot(meta_cube.lbda, guessEll, 'k--')
        # plot_conf_interval(ax6b, meta_cube.lbda, fit_ell, err_ell)
        plot_conf_interval(ax6b, meta_cube.lbda, fit_ell, None)
        txt = 'Guess: %s' % \
              (', '.join([ 'e%d=%.2f' % (i, e)
                           for i, e in enumerate(guessEllCoeffs) ]))
        txt += '\nFit: %s' % \
               (', '.join([ 'e%d=%.2f' % (i, e)
                            for i, e in enumerate(fitpar[5:6 + ellDeg]) ]))
        ax6b.text(0.95, 0.1, txt, transform=ax6b.transAxes,
                  fontsize='small', ha='right', va='bottom')
        P.setp(ax6b.get_yticklabels(), fontsize='x-small')

        if good.any():
            ax6c.errorbar(meta_cube.lbda[good], xy_vec[good], dparams[good, 4],
                          marker='.',
                          mfc=blue, mec=blue, ecolor=blue, capsize=0, ls='None')
        if bad.any():
            ax6c.plot(meta_cube.lbda[bad], xy_vec[bad],
                      marker='.', mfc=red, mec=red, ls='None')
        ax6c.plot([meta_cube.lstart, meta_cube.lend], [xy] * 2, 'k--')
        plot_conf_interval(ax6c,
                           N.asarray([meta_cube.lstart, meta_cube.lend]),
                           N.ones(2) * fitpar[4], N.ones(2) * err_xy)
        ax6c.text(0.95, 0.1,
                  u'Guess: xy=%4.2f\nFit: xy=%4.2f' % (xy, fitpar[4]),
                  transform=ax6c.transAxes,
                  fontsize='small', ha='right', va='bottom')
        P.setp(ax6c.get_xticklabels() + ax6c.get_yticklabels(),
               fontsize='x-small')

        fig6.subplots_adjust(left=0.1, right=0.96, bottom=0.08, top=0.95)
        if plot6:
            fig6.savefig(plot6)

        # Plot of the radial profile -----------------------------------------

        print_msg("Producing radial profile plot %s..." % plot7, 1)

        fig7 = P.figure()
        fig7.suptitle(
            "Radial profile plot [%s, airmass=%.2f]" % (objname, airmass),
            fontsize='large')

        def ellRadius(x, y, x0, y0, ell, xy):
            dx = x - x0
            dy = y - y0
            # BEWARE: can return NaN's if ellipse is ill-defined
            return N.sqrt(dx ** 2 + ell * dy ** 2 + 2 * xy * dx * dy)

        def radialbin(r, f, binsize=20, weighted=True):
            rbins = N.sort(r)[::binsize]  # Bin limits, starting from min(r)
            ibins = N.digitize(r, rbins)  # WARNING: ibins(min(r)) = 1
            ib = N.arange(len(rbins)) + 1  # Bin indices
            ib = [iib for iib in ib if r[ibins == iib].any()]
            rb = N.array([r[ibins == iib].mean() for iib in ib])  # Mean radius
            if weighted:
                fb = N.array([N.average(f[ibins == iib], weights=r[ibins == iib])
                              for iib in ib])  # Mean radius-weighted data
            else:
                fb = N.array([f[ibins == iib].mean()
                              for iib in ib])  # Mean data
            # Error on bin mean quantities
            # snb = N.sqrt([ len(r[ibins==iib]) for iib in ib ]) # sqrt(#points)
            # drb = N.array([ r[ibins==iib].std()/n for iib,n in zip(ib,snb) ])
            # dfb = N.array([ f[ibins==iib].std()/n for iib,n in zip(ib,snb) ])
            return rb, fb

        for i in xrange(nmeta):        # Loop over slices
            ax = fig7.add_subplot(nrow, ncol, i + 1, yscale='log')
            # Use adjusted elliptical radius instead of plain radius
            # r    = N.hypot(meta_cube.x-xfit[i], meta_cube.y-yfit[i])
            # rfit = N.hypot(cube_fit.x-xfit[i], cube_fit.y-yfit[i])
            r = ellRadius(meta_cube.x, meta_cube.y,
                          xfit[i], yfit[i], fit_ell[i], fitpar[4])
            rfit = ellRadius(cube_fit.x, cube_fit.y,
                             xfit[i], yfit[i], fit_ell[i], fitpar[4])
            ax.plot(r, meta_cube.data[i],
                    marker=',', mfc=blue, mec=blue, ls='None')  # Data
            ax.plot(rfit, cube_fit.data[i],
                    marker='.', mfc=red, mec=red, ms=1, ls='None')  # Model
            # ax.set_autoscale_on(False)
            if hasSky and sky_spec.data.any():
                ax.plot(rfit, psf[i], marker='.', mfc=green, mec=green,
                        ms=1, ls='None')  # PSF alone
                ax.plot(rfit, bkg[i], marker='.', mfc=orange, mec=orange,
                        ms=1, ls='None')  # Sky
            P.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                   fontsize='xx-small')
            ax.text(0.05, 0.85, u"%.0f Å" % meta_cube.lbda[i],
                    fontsize='x-small', transform=ax.transAxes)
            if opts.method != 'psf':
                ax.axvline(radius / spxSize, color=orange, lw=2)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Elliptical radius [spx]", fontsize='small')
                ax.set_ylabel("Flux", fontsize='small')
            # ax.axis([0, rfit.max()*1.1,
            #          meta_cube.data[i][meta_cube.data[i]>0].min()/2,
            #          meta_cube.data[i].max()*2])
            ax.axis([0, rfit.max() * 1.1,
                     cube_fit.data[i].min() / 2, cube_fit.data[i].max() * 2])

            # Binned values
            rb, db = radialbin(r, meta_cube.data[i])
            ax.plot(rb, db, 'c.')
            rfb, fb = radialbin(rfit, cube_fit.data[i])
            ax.plot(rfb, fb, 'm.')

        fig7.subplots_adjust(left=0.07, right=0.96, bottom=0.06, top=0.94)
        if plot7:
            fig7.savefig(plot7)

        # Missing energy (not activated by default)
        if opts.verbosity >= 3:
            print_msg("Producing missing energy plot...", 1)

            figB = P.figure()
            for i in xrange(nmeta):        # Loop over slices
                ax = figB.add_subplot(nrow, ncol, i + 1, yscale='log')
                r = ellRadius(meta_cube.x, meta_cube.y,
                              xfit[i], yfit[i], fit_ell[i], fitpar[4])
                rfit = ellRadius(cube_fit.x, cube_fit.y,
                                 xfit[i], yfit[i], fit_ell[i], fitpar[4])
                # Binned values
                rb, db = radialbin(r, meta_cube.data[i])
                rfb, fb = radialbin(rfit, cube_fit.data[i])
                tb = N.cumsum(rb * db)
                norm = tb.max()
                ax.plot(rb, 1 - tb / norm, 'c.')
                ax.plot(rfb, 1 - N.cumsum(rfb * fb) / norm, 'm.')
                if hasSky and sky_spec.data.any():
                    rfb, pb = radialbin(rfit, psf[i])
                    rfb, bb = radialbin(rfit, bkg[i])
                    ax.plot(rfb, 1 - N.cumsum(rfb * pb) / norm,
                            marker='.', mfc=green, mec=green, ls='None')
                    ax.plot(rfb, 1 - N.cumsum(rfb * bb) / norm,
                            marker='.', mfc=orange, mec=orange, ls='None')
                P.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                       fontsize='xx-small')
                ax.text(0.05, 0.85, u"%.0f Å" % meta_cube.lbda[i],
                        fontsize='x-small', transform=ax.transAxes)
                if opts.method != 'psf':
                    ax.axvline(radius / spxSize, color=orange, lw=2)
                if ax.is_last_row() and ax.is_first_col():
                    ax.set_xlabel("Elliptical radius [spx]",
                                  fontsize='small')
                    ax.set_ylabel(
                        "Missing energy [fraction]", fontsize='small')

            figB.tight_layout()
            if opts.graph != 'pylab':
                figB.savefig(
                    os.path.extsep.join((basename + "_nrj", opts.graph)))

        # Radial Chi2 plot (not activated by default)
        if opts.verbosity >= 3:
            print_msg("Producing radial chi2 plot...", 1)

            figA = P.figure()
            for i in xrange(nmeta):        # Loop over slices
                ax = figA.add_subplot(nrow, ncol, i + 1, yscale='log')
                rfit = ellRadius(cube_fit.x, cube_fit.y,
                                 xfit[i], yfit[i], fit_ell[i], fitpar[4])
                chi2 = (meta_cube.slice2d(i, coord='p') -
                        cube_fit.slice2d(i, coord='p')) ** 2 / \
                    meta_cube.slice2d(i, coord='p', var=True)
                ax.plot(rfit, chi2.flatten(),
                        marker='.', ls='none', mfc=blue, mec=blue)
                P.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                       fontsize='xx-small')
                ax.text(0.05, 0.85, u"%.0f Å" % meta_cube.lbda[i],
                        fontsize='x-small', transform=ax.transAxes)
                if opts.method != 'psf':
                    ax.axvline(radius / spxSize, color=orange, lw=2)
                if ax.is_last_row() and ax.is_first_col():
                    ax.set_xlabel("Elliptical radius [spx]",
                                  fontsize='small')
                    ax.set_ylabel(u"χ²", fontsize='small')

            figA.tight_layout()
            if opts.graph != 'pylab':
                figA.savefig(
                    os.path.extsep.join((basename + "_chi2", opts.graph)))

        # Contour plot of each slice -----------------------------------------

        print_msg("Producing PSF contour plot %s..." % plot8, 1)

        fig8 = P.figure()
        fig8.suptitle(
            "Data and fit [%s, airmass=%.2f]" % (objname, airmass),
            fontsize='large')

        extent = (meta_cube.x.min() - 0.5, meta_cube.x.max() + 0.5,
                  meta_cube.y.min() - 0.5, meta_cube.y.max() + 0.5)
        for i in xrange(nmeta):        # Loop over meta-slices
            ax = fig8.add_subplot(ncol, nrow, i + 1, aspect='equal')
            data = meta_cube.slice2d(i, coord='p')
            fit = cube_fit.slice2d(i, coord='p')
            vmin, vmax = N.percentile(data[data > 0], (5., 95.))  # Percentiles
            lev = N.logspace(N.log10(vmin), N.log10(vmax), 5)
            ax.contour(data, lev, origin='lower', extent=extent,
                       cmap=M.cm.jet)                      # Data
            ax.contour(fit, lev, origin='lower', extent=extent,
                       linestyles='dashed', cmap=M.cm.jet)  # Fit
            ax.errorbar((xc_vec[i],), (yc_vec[i],),
                        xerr=(dparams[i, 2],), yerr=(dparams[i, 3],),
                        fmt=None, ecolor=blue if good[i] else red)
            ax.plot((xfit[i],), (yfit[i],), marker='*', color=green)
            if opts.method != 'psf':
                ax.add_patch(M.patches.Circle((xfit[i], yfit[i]),
                                              radius / spxSize,
                                              fc='None', ec=orange, lw=2))
            P.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                   fontsize='xx-small')
            ax.text(0.05, 0.85, u"%.0f Å" % meta_cube.lbda[i],
                    fontsize='x-small', transform=ax.transAxes)
            ax.axis(extent)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I [spx]", fontsize='small')
                ax.set_ylabel("J [spx]", fontsize='small')
            if not ax.is_last_row():
                P.setp(ax.get_xticklabels(), visible=False)
            if not ax.is_first_col():
                P.setp(ax.get_yticklabels(), visible=False)

        fig8.subplots_adjust(left=0.05, right=0.96, bottom=0.06, top=0.95,
                             hspace=0.02, wspace=0.02)
        if plot8:
            fig8.savefig(plot8)

        # Residuals of each slice --------------------------------------------

        print_msg("Producing residual plot %s..." % plot5, 1)

        fig5 = P.figure()
        fig5.suptitle(
            "Residual plot [%s, airmass=%.2f]" % (objname, airmass),
            fontsize='large')

        images = []
        for i in xrange(nmeta):        # Loop over meta-slices
            ax = fig5.add_subplot(ncol, nrow, i + 1, aspect='equal')
            data = meta_cube.slice2d(i, coord='p')  # Signal
            fit = cube_fit.slice2d(i, coord='p')  # Model
            if opts.chi2fit:    # Chi2 fit: display residuals in units of sigma
                var = meta_cube.slice2d(i, coord='p', var=True, NAN=False)
                res = N.nan_to_num((data - fit) / N.sqrt(var))
            else:               # Least-squares: display relative residuals
                res = N.nan_to_num((data - fit) / fit) * 100  # [%]

            # List of images, to be commonly normalized latter on
            images.append(ax.imshow(res, origin='lower', extent=extent,
                                    cmap=M.cm.RdBu_r, interpolation='nearest'))

            ax.plot((xfit[i],), (yfit[i],), marker='*', color=green)
            P.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                   fontsize='xx-small')
            ax.text(0.05, 0.85, u"%.0f Å" % meta_cube.lbda[i],
                    fontsize='x-small', transform=ax.transAxes)
            ax.axis(extent)

            # Axis management
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I [spx]", fontsize='small')
                ax.set_ylabel("J [spx]", fontsize='small')
            if not ax.is_last_row():
                ax.xaxis.set_major_formatter(M.ticker.NullFormatter())
            if not ax.is_first_col():
                ax.yaxis.set_major_formatter(M.ticker.NullFormatter())

        # Common image normalization
        vmin, vmax = N.percentile(
            [im.get_array().filled() for im in images], (3., 97.))
        norm = M.colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        # Colorbar
        cax = fig5.add_axes([0.90, 0.07, 0.02, 0.87])
        cbar = fig5.colorbar(images[0], cax, orientation='vertical')
        P.setp(cbar.ax.get_yticklabels(), fontsize='small')
        if opts.chi2fit:    # Chi2 fit
            cbar.set_label(u'Residuals [σ]', fontsize='small')
        else:
            cbar.set_label(u'Residuals [%]', fontsize='small')

        fig5.subplots_adjust(left=0.06, right=0.89, bottom=0.06, top=0.95,
                             hspace=0.02, wspace=0.02)
        if plot5:
            fig5.savefig(plot5)

        # Show figures -------------------------------------------------

        if opts.graph == 'pylab':
            P.show()

# End of extract_star.py ======================================================
