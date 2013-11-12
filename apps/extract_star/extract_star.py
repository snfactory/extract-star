#!/usr/bin/env python
# -*- coding: utf-8 -*-
##############################################################################
## Filename:          extract_star.py
## Version:           $Revision$
## Description:       Standard star spectrum extraction
## Author:            Clément BUTON <buton@physik.uni-bonn.de>
## Author:            $Author$
## Created at:        $Date$
## Modified at:       2012/11/16 16:13:34
## $Id$
##############################################################################

"""3D PSF-based point-source extractor. The PSF is a constrained
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

The polynomial approximations for alpha and ellipticity are expressed
internally as function of lr := (2*lambda - (lmin+lmax))/(lmax-lmin), to
minimize correlations. But the coeffs stored in header keywords are for
polynoms of lr := lambda/lref - 1.

N.B: the The polynomial approximations for alpha and ellipticity
stored in the log files (options '-F' and '-f') are expressed as it is
done internally.
"""

__author__ = "C. Buton, Y. Copin, E. Pecontal"
__version__ = '$Id$'

import os

import pyfits as F
import numpy as N 
import scipy.linalg as SL

import pySNIFS
import pySNIFS_fit
import libExtractStar as libES
from ToolBox.Atmosphere import ADR
from ToolBox.Optimizer import approx_deriv, cov2corr
from ToolBox.Arrays import metaslice
from ToolBox import MPL

# Numpy setup
N.set_printoptions(linewidth=999)       # X-wide lines

LbdaRef = 5000.     # Use constant ref. for easy comparison

# Non-default colors
blue   = MPL.blue
red    = MPL.red
green  = MPL.green
orange = MPL.orange
purple = MPL.purple

# Definitions ================================================================

def print_msg(str, limit):
    """Print message 'str' if verbosity level (opts.verbosity) >= limit."""

    libES.print_msg(str, limit, verb=opts.verbosity)


@MPL.make_method(pySNIFS_fit.model)
def param_covariance(self, param=None, order=3):
    """Overrides pySNIFS_fit.model.param_error for covariance
    computation."""

    if param is None:
        param = self.fitpar

    hes = approx_deriv(self.objgrad, param, order=order) # Chi2 hessian
    try:
        cov = 2 * SL.pinv2(hes)         # Covariance matrix
    except SL.LinAlgError, error:
        print "Error while inverting chi2 hessian:", error
        cov = N.zeros_like(hes)

    return cov


def spec_covariance(cube, psf, skyDeg, covpar):
    """Compute point-source spectrum full covariance from
    parameter covariance."""

    psfFn, psfCtes, fitpar = psf

    # Function fitpar to point-source spectrum (nslice,)
    func = lambda fitpar: libES.extract_specs(cube,
                                              (psfFn, psfCtes, fitpar),
                                              skyDeg=skyDeg)[1][:,0]
    # Associated jacobian (numerical evaluation) (npar,nslice)
    jac = approx_deriv(func, fitpar)

    # Covariance propagation
    return N.dot(N.dot(jac.T, covpar), jac) # (nslice,nslice)


@MPL.make_method(pySNIFS.spectrum)
def write_fits(self, filename=None, header=None):
    """Overrides pySNIFS_fit.spectrum.WR_fits_file. Allows full header
    propagation (including comments) and covariance matrix storage."""

    assert None not in (self.start, self.step, self.data)

    # Primary HDU: signal
    hdusig = F.PrimaryHDU(self.data, header=header)
    for key in ['EXTNAME','CTYPES','CRVALS','CDELTS','CRPIXS']:
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
        #hducov = F.CompImageHDU(N.tril(self.cov), name='COVAR')
        hducov = F.ImageHDU(N.tril(self.cov), name='COVAR')
        hducov.header['CRVAL1'] = self.start
        hducov.header['CDELT1'] = self.step
        hducov.header['CRVAL2'] = self.start
        hducov.header['CDELT2'] = self.step

        hduList.append(hducov)

    if filename:                        # Save hduList to disk
        hduList.writeto(filename, output_verify='silentfix', clobber=True)

    return hduList                      # For further handling if needed


def create_2Dlog(opts, cube, params, dparams, chi2):
    """Dump an informative text log about the PSF (metaslice) 2D-fit."""

    logfile = file(opts.log2D,'w')

    logfile.write('# cube    : %s   \n' % os.path.basename(opts.input))
    logfile.write('# object  : %s   \n' % cube.e3d_data_header["OBJECT"])
    logfile.write('# airmass : %.3f \n' % cube.e3d_data_header["AIRMASS"])
    logfile.write('# efftime : %.3f \n' % cube.e3d_data_header["EFFTIME"])

    npar_sky = (opts.skyDeg+1)*(opts.skyDeg+2)/2

    delta,theta  = params[:2]
    xc,yc        = params[2:4]
    PA,ell,alpha = params[4:7]
    intensity    = params[-npar_sky-1]
    sky          = params[-npar_sky:]

    labels = '# lbda  ' + \
        '  '.join('%8s +/- d%-8s' % (n,n)
                  for n in ['delta','theta','xc','yc','PA','ell','alpha','I'] + 
                  ['sky%d' % d for d in xrange(npar_sky)])
    if cube.var is None:        # Least-square fit: compute Res. Sum of Squares
        labels += '        RSS\n'
    else:                       # Chi2 fit: compute chi2 per slice
        labels += '        chi2\n'
    logfile.write(labels)
    fmt = '%6.0f  ' + '  '.join(["%10.4g"]*((8+npar_sky)*2+1)) + '\n'

    for n in xrange(cube.nslice):
        list2D  = [cube.lbda[n],
                   delta[n], dparams[n][0],
                   theta[n], dparams[n][1],
                   xc[n]   , dparams[n][2],
                   yc[n]   , dparams[n][3],
                   PA[n]   , dparams[n][4],
                   ell[n]  , dparams[n][5],
                   alpha[n], dparams[n][6],
                   intensity[n], dparams[n][-npar_sky-1]]
        if npar_sky:
            tmp = N.array((sky[:,n],dparams[n][-npar_sky:]))
            list2D += tmp.T.flatten().tolist()
        list2D += [chi2[n]]
        logfile.write(fmt % tuple(list2D))

    logfile.close()


def create_3Dlog(opts, cube, cube_fit, fitpar, dfitpar, chi2):
    """Dump an informative text log about the PSF (full-cube) 3D-fit."""

    logfile = file(opts.log3D,'w')

    logfile.write('# cube    : %s   \n' % os.path.basename(opts.input))
    logfile.write('# object  : %s   \n' % cube.e3d_data_header["OBJECT"])
    logfile.write('# airmass : %.3f \n' % cube.e3d_data_header["AIRMASS"])
    logfile.write('# efftime : %.3f \n' % cube.e3d_data_header["EFFTIME"])

    # Global parameters
    # lmin  lmax  delta +/- ddelta  ...  alphaN +/- dalphaN chi2|RSS
    labels = '# lmin  lmax' + \
        '  '.join('%8s +/- d%-8s' % (n,n)
                  for n in ['delta','theta','xc','yc','PA'] + 
                  ['ell%d' % d for d in xrange(ellDeg+1)] +
                  ['alpha%d' % d for d in xrange(alphaDeg+1)])
    if cube.var is None:        # Least-square fit: Residual Sum of Squares
        labels += '        RSS\n'
    else:                       # Chi2 fit: true chi2
        labels += '        chi2\n'
    logfile.write(labels)
    fmt = '%6.0f  %6.0f  ' + \
          '  '.join(["%10.4g"]*((5+(ellDeg+1)+(alphaDeg+1))*2+1)) + '\n'
    list3D = [cube.lstart, cube.lend,
              fitpar[0], dfitpar[0],
              fitpar[1], dfitpar[1],
              fitpar[2], dfitpar[2],
              fitpar[3], dfitpar[3],
              fitpar[4], dfitpar[4]]
    for i in xrange(ellDeg+1):   # Ellipticity coefficiens
        list3D += [fitpar[5+i], dfitpar[5+i]]
    for i in xrange(alphaDeg+1): # Alpha coefficients
        list3D += [fitpar[6+ellDeg+i], dfitpar[6+ellDeg+i]]
    list3D += [chi2]             # chi2|RSS
    logfile.write(fmt % tuple(list3D))

    # Metaslice parameters
    # lbda  I -/- dI  sky0 +/- dsky0  sky1 +/- dsky1  ...  chi2|RSS
    npar_psf = 7 + ellDeg + alphaDeg
    npar_sky = (opts.skyDeg+1)*(opts.skyDeg+2)/2

    labels = '# lbda  ' + \
        '  '.join( '%8s +/- d%-8s' % (par,par) 
                   for par in ['I']+['sky%d' % d for d in range(npar_sky)] )
    if cube.var is None:        # Least-square fit: compute Res. Sum of Squares
        labels += '        RSS\n'
    else:                       # Chi2 fit: compute chi2 per slice
        labels += '        chi2\n'
    logfile.write(labels)
    fmt = '%6.0f  ' + '  '.join(["%10.4g"]*((1+npar_sky)*2+1)) + '\n'
    for n in xrange(cube.nslice):       # Loop over metaslices
        # Wavelength, intensity and error on intensity
        list2D = [cube.lbda[n], fitpar[npar_psf+n], dfitpar[npar_psf+n]]
        for i in xrange(npar_sky): # Add background parameters
            list2D.extend([fitpar[npar_psf+nslice+n*npar_sky+i],
                           dfitpar[npar_psf+nslice+n*npar_sky+i]])
        # Compute chi2|RSS
        chi2 = N.nan_to_num((cube.slice2d(n, coord='p') - 
                             cube_fit.slice2d(n, coord='p'))**2)
        if cube.var is not None:    # chi2: divide by variance
            chi2 /= cube.slice2d(n, coord='p', var=True)
        list2D += [chi2.sum()]      # Slice chi2|RSS
        logfile.write(fmt % tuple(list2D))

    logfile.close()


def fill_header(hdr, psfname, param, adr, cube, opts, chi2, seeing, fluxes):
    """Fill header *hdr* with PSF fit-related keywords."""

    # Convert reference position from lref=(lmin+lmax)/2 to LbdaRef
    lmin,lmax = cube.lstart,cube.lend
    x0,y0 = adr.refract(param[2],param[3], LbdaRef, unit=cube.spxSize)
    print_msg("Reference position [%.0fA]: %.2f x %.2f spx" % 
              (LbdaRef,x0,y0), 1)

    # Convert polynomial coeffs from lr=(2*lambda - (lmin+lmax))/(lmax-lmin)
    # to lr~ = lambda/LbdaRef - 1 = a + b*lr
    a = (lmin+lmax) / (2*LbdaRef) - 1
    b = (lmax-lmin) / (2*LbdaRef)
    c_ell = libES.polyConvert(param[5:6+opts.ellDeg], trans=(a,b))
    c_alp = libES.polyConvert(param[6+opts.ellDeg:7+opts.ellDeg+opts.alphaDeg],
                              trans=(a,b))
    if 'powerlaw' in psfname:
        print "WARNING: ES_Axx keywords are just *WRONG* for powerlaw PSF"

    hdr['ES_VERS'] = __version__
    hdr['ES_CUBE'] = (opts.input, 'Input cube')
    hdr['ES_LREF'] = (LbdaRef, 'Lambda ref. [A]')
    hdr['ES_SDEG'] = (opts.skyDeg,'Polynomial bkgnd degree')
    hdr['ES_CHI2'] = (chi2, 'Chi2 of 3D fit')
    hdr['ES_AIRM'] = (adr.get_airmass(), 'Effective airmass')
    hdr['ES_PARAN'] = (adr.get_parangle(), 'Effective parangle [deg]')
    hdr['ES_XC'] = (x0, 'xc @lbdaRef [spx]')
    hdr['ES_YC'] = (y0, 'yc @lbdaRef [spx]')
    hdr['ES_XY'] = (param[4], 'XY coeff.')
    hdr['ES_LMIN'] = (lmin, 'Meta-slices minimum lambda')
    hdr['ES_LMAX'] = (lmax, 'Meta-slices maximum lambda')

    for i in xrange(opts.ellDeg + 1):
        hdr['ES_E%i' % i] = (c_ell[i], 'Y2 coeff. e%d' % i)
    for i in xrange(opts.alphaDeg + 1):
        hdr['ES_A%i' % i] = (c_alp[i], 'Alpha coeff. a%d' % i)

    hdr['ES_METH'] = (opts.method, 'Extraction method')
    if opts.method == 'psf':
        hdr['ES_PSF'] = (psfname, 'PSF name')
    else:
        hdr['ES_APRAD'] = (opts.radius, 'Aperture radius [arcsec or sigma]')

    tflux, sflux = fluxes
    hdr['ES_TFLUX'] = (tflux, 'Sum of the spectrum flux')
    if opts.skyDeg >= 0:
        hdr['ES_SFLUX'] = (sflux, 'Sum of the sky flux')

    hdr['SEEING'] = (seeing, 'Seeing @lbdaRef [arcsec] (extract_star)')

    if opts.supernova:
        hdr['ES_SNMOD'] = (opts.supernova, 'Supernova mode')
    if opts.psf3Dconstraints:
        for i,constraint in enumerate(opts.psf3Dconstraints):
            hdr['ES_BND%d' % (i+1)] = (constraint, "Constraint on 3D-PSF")


def setPSF3Dconstraints(psfConstraints, params, bounds):
    """Decipher psf3Dconstraints=[constraint] option and set initial
    guess params and/or bounds accordingly. Each constraint is a
    string 'n:val' (strict constraint) or 'n:val1,val2' (loose
    constraint), for n=0 (delta), 1 (theta), 2,3 (position), 4 (PA),
    5...6+ellDeg (ellipticity polynomial coefficients) and
    7+ellDeg...8+ellDeg+alphaDeg (alpha polynomial coefficients)."""

    for psfConstraint in psfConstraints:
        try:
            n,constraintStr = psfConstraint.split(':')
            n = int(n)
            vals = map(float, constraintStr.split(','))
            assert len(vals) in (1,2)
        except (ValueError, AssertionError):
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
                vmin, vmax = sorted(vals)
                params[n] = min(max(params[n], vmin), vmax)
                bounds[n] = [vmin, vmax]
                print "WARNING: Constraining PSF param[%d] in %f,%f" % \
                      (n,vmin,vmax)


# ########## MAIN ##############################

if __name__ == "__main__":

    import optparse

    # Options ================================================================

    methods = ('psf','aperture','subaperture','optimal')

    usage = "[%prog] [options] incube.fits"

    parser = optparse.OptionParser(usage, version=__version__)

    parser.add_option("-i", "--in", type="string", dest="input",
                      help="Input datacube (or use argument)")
    parser.add_option("-o", "--out", type="string",
                      help="Output point source spectrum")
    parser.add_option("-s", "--sky", type="string",
                      help="Output sky spectrum")

    # Covariance management
    parser.add_option("-V", "--covariance", action='store_true',
                      help="Compute and store covariance matrix in extension",
                      default=False)

    # Parameters
    parser.add_option("-S", "--skyDeg", type="int",
                      help="Sky polynomial background degree [%default]",
                      default=0)
    parser.add_option("-A", "--alphaDeg", type="int",
                      help="Alpha polynomial degree [%default]",
                      default=2)
    parser.add_option("-E", "--ellDeg", type="int",
                      help="Ellipticity polynomial degree [%default]",
                      default=0)

    parser.add_option("-N", "--nmeta", type='int',
                      help="Number of meta-slices [%default].",
                      default=12)

    # PSF model
    parser.add_option("--psf",
                      choices=('classic','classic-powerlaw','chromatic'),
                      help="PSF model " \
                      "(classic[-powerlaw]|chromatic) [%default]",
                      default='classic')

    # Extraction method and parameters
    parser.add_option("-m", "--method",
                      choices=('psf','optimal','aperture','subaperture'),
                      help="Extraction method " \
                      "(psf|optimal|[sub]aperture) [%default]",
                      default="psf")
    parser.add_option("-r", "--radius", type="float",
                      help="Aperture radius for non-PSF extraction " \
                           "(>0: in arcsec, <0: in seeing sigma) [%default]",
                      default=-5.)
    parser.add_option("-L", "--leastSquares", 
                      dest="chi2fit", action="store_false",
                      help="Least-square fit [default is a chi2 fit].",
                      default=True)

    # Plotting
    parser.add_option("-g", "--graph", type="string",
                      help="Graphic output format (eps,pdf,png,pylab)")
    parser.add_option("-p", "--plot", action='store_true',
                      help="Plot flag (='-g pylab')")
    parser.add_option("-v", "--verbosity", type="int",
                      help="Verbosity level (<0: quiet) [%default]",
                      default=0)

    # Debug logs
    parser.add_option("-f", "--file", type="string", dest="log2D",
                      help="2D adjustment logfile name.")
    parser.add_option("-F", "--File", type="string", dest="log3D",
                      help="3D adjustment logfile name.")

    # Expert options
    parser.add_option("--supernova", action='store_true',
                      help="SN mode (no final 3D fit).")
    parser.add_option("--keepmodel", action='store_true',
                      help="Store meta-slices and adjusted model in 3D cubes.")
    parser.add_option("--psf3Dconstraints", type='string', action='append',
                      help="Constraints on PSF parameters (n:val,[val]).")

    opts,args = parser.parse_args()
    if not opts.input:
        if args:
            opts.input = args[0]
        else:
            parser.error("No input datacube specified.")

    if opts.graph:
        opts.plot = True
    elif opts.plot:
        opts.graph = 'pylab'

    if opts.skyDeg < 0:
        opts.skyDeg = -1
        if opts.sky:
            print "WARNING: cannot extract sky spectrum in no-sky mode."

    if opts.verbosity<=0:
        N.seterr(all='ignore')

    # Input datacube ===========================================================

    print "Opening datacube %s" % opts.input

    # The pySNIFS e3d_data_header dictionary is not enough for later
    # updates in fill_hdr, which requires a *true* pyfits header.

    try:                                # Try to read a Euro3D cube
        inhdr = F.getheader(opts.input, 1) # 1st extension
        full_cube = pySNIFS.SNIFS_cube(e3d_file=opts.input)
        isE3D = True
    except ValueError:                  # Try to read a 3D FITS cube
        inhdr = F.getheader(opts.input, 0) # Primary extension
        full_cube = pySNIFS.SNIFS_cube(fits3d_file=opts.input)
        isE3D = False
    step = full_cube.lstep

    print_msg("Cube %s [%s]: %d slices [%.2f-%.2f], %d spaxels" % 
              (os.path.basename(opts.input), 'E3D' if isE3D else '3D',
               full_cube.nslice,
               full_cube.lbda[0], full_cube.lbda[-1], full_cube.nlens), 1)

    objname = inhdr.get('OBJECT', 'Unknown')
    efftime = inhdr['EFFTIME']            # [s]
    airmass = inhdr['AIRMASS']
    try:
        parangle = inhdr['PARANG']        # [deg]
    except KeyError:                      # Not in original headers
        from ToolBox.Astro import Coords
        phi = inhdr['LATITUDE']
        ha,dec = Coords.altaz2hadec(inhdr['ALTITUDE'], inhdr['AZIMUTH'],
                                   phi=phi, deg=True)
        zd,parangle = Coords.hadec2zdpar(ha, dec, phi=phi, deg=True)
        print "WARNING: computing PARANG from ALTITUDE, AZIMUTH and LATITUDE."

    channel = inhdr['CHANNEL'][0].upper() # 'B' or 'R'
    pressure,temp = libES.read_PT(inhdr)  # Include validity tests and defaults

    ellDeg   = opts.ellDeg
    alphaDeg = opts.alphaDeg
    npar_psf = 7 + ellDeg + alphaDeg

    skyDeg   = opts.skyDeg
    npar_sky = (skyDeg+1)*(skyDeg+2)/2

    # Select the PSF
    if opts.psf == 'chromatic':         # Includes chromatic correlations
        # Chromatic parameter description (short or long, red or blue)
        if (efftime > 12.) and (channel=='B'):
            psfFn = libES.LongBlue_ExposurePSF
        elif (efftime > 12.) and (channel=='R'):
            psfFn = libES.LongRed_ExposurePSF
        elif (efftime <= 12.) and (channel=='B'):
            psfFn = libES.ShortBlue_ExposurePSF
        elif (efftime <= 12.) and (channel=='R'):
            psfFn = libES.ShortRed_ExposurePSF
    elif opts.psf.startswith('classic'): # Achromatic correlations
        # Classical parameter description (short or long)
        psfFn = libES.Long_ExposurePSF if (efftime > 12.) \
                else libES.Short_ExposurePSF
        if opts.psf.endswith('powerlaw'):
            psfFn.model += '-powerlaw'
    else:
        parser.error("Invalid PSF model '%s'" % opts.psf)

    print "  Object: %s, Airmass: %.2f, Efftime: %.1fs, PSF: %s" % \
          (objname, airmass, efftime, ', '.join((psfFn.model,psfFn.name)))

    # Test channel and set default output name
    if channel not in ('B','R'):
        parser.error("Input datacube %s has no valid CHANNEL keyword (%s)" % 
                     (opts.input, channel))
    if not opts.out:
        opts.out = 'spec_%s.fits' % (channel)

    # Meta-slice definition (min,max,step [px])

    slices = metaslice(full_cube.nslice, opts.nmeta, trim=10)
    print "  Channel: '%s', extracting slices: %s" % (channel, slices)

    if isE3D:
        meta_cube = pySNIFS.SNIFS_cube(e3d_file=opts.input, slices=slices)
    else:
        meta_cube = pySNIFS.SNIFS_cube(fits3d_file=opts.input, slices=slices)
    meta_cube.x = meta_cube.i - 7       # From I,J to spx coords
    meta_cube.y = meta_cube.j - 7
    spxSize = meta_cube.spxSize
    nmeta = meta_cube.nslice

    print_msg("  Meta-slices before selection: "
              "%d from %.2f to %.2f by %.2f A" % 
              (nmeta, meta_cube.lstart, meta_cube.lend, meta_cube.lstep), 0)

    # Normalisation of the signal and variance in order to avoid numerical
    # problems with too small numbers
    norm = meta_cube.data.max(axis=-1).reshape(-1,1) # (nmeta,1)
    print_msg("  Meta-slice normalization (max): %s" % (norm.squeeze()), 2)
    meta_cube.data /= norm                           # (nmeta,nspx)
    meta_cube.var  /= norm**2

    if opts.keepmodel:                  # Store meta-slices in 3D-cube
        path,name = os.path.split(opts.out)
        outpsf = os.path.join(path,'meta_'+name)
        print "Saving meta-slices in 3D-fits cube '%s'..." % outpsf
        meta_cube.WR_3d_fits(outpsf)

    # Computing guess parameters from slice by slice 2D fit ====================

    print "Meta-slice 2D-fitting (%s)..." % \
          ('chi2' if opts.chi2fit else 'least-squares')
    params,chi2s,dparams = libES.fit_metaslices(
        meta_cube, psfFn, skyDeg=skyDeg, chi2fit=opts.chi2fit,
        verbosity=opts.verbosity)
    print_msg("", 1)

    params = params.T             # (nparam,nslice)
    delta_vec,theta_vec = params[:2]
    xc_vec,yc_vec       = params[2:4]
    PA_vec,ell_vec,alpha_vec = params[4:7]
    int_vec = params[7]
    if skyDeg >= 0:
        sky_vec = params[8:]

    # Save 2D adjusted parameter file ==========================================

    if opts.log2D:
        print "Producing 2D adjusted parameter logfile %s..." % opts.log2D
        create_2Dlog(opts, meta_cube, params, dparams, chi2s)

    # 3D model fitting =========================================================

    print "Datacube 3D-fitting (%s)..." % \
          ('chi2' if opts.chi2fit else 'least-squares')

    # Initial guess ------------------------------
    # Computing the initial guess for the 3D fitting from the results
    # of the slice by slice 2D fit
    lmid = (meta_cube.lstart + meta_cube.lend)/2
    lbda_rel = libES.chebNorm(
        meta_cube.lbda, meta_cube.lstart, meta_cube.lend) # in [-1,1]

    # 1) Reference position
    # Convert meta-slice centroids to position at ref. lbda, and clip around
    # median position
    adr = ADR(pressure, temp, lref=lmid, airmass=airmass, parangle=parangle)
    delta0 = adr.delta                  # ADR power = tan(zenithal distance)
    theta0 = adr.theta                  # ADR angle = parallactic angle [rad]
    print_msg(str(adr), 1)
    xref,yref = adr.refract(
        xc_vec, yc_vec, meta_cube.lbda, backward=True, unit=spxSize)
    valid = chi2s > 0                   # Discard unfitted slices
    xref0 = N.median(xref[valid])       # Robust to outliers
    yref0 = N.median(yref[valid])
    r = N.hypot(xref - xref0, yref - yref0)
    rmax = 5*N.median(r[valid])         # Robust to outliers
    good = valid & (r <= rmax)          # Valid fit and reasonable position
    bad  = valid & (r > rmax)           # Valid fit but discarded position
    if bad.any():
        print "WARNING: %d metaslices discarded after ADR selection" % \
              (len(N.nonzero(bad)))
    print_msg("%d/%d centroids found withing %.2f spx of (%.2f,%.2f)" % 
              (len(xref[good]),len(xref),rmax,xref0,yref0), 1)
    xc,yc = xref[good].mean(), yref[good].mean()
    # We could use a weighted average, but does not make much of a difference
    # dx,dy = dparams[:,2],dparams[:,3]
    # xc = N.average(xref[good], weights=1/dx[good]**2)
    # yc = N.average(yref[good], weights=1/dy[good]**2)

    if not good.all():                   # Invalid slices + discarded centroids
        print "%d/%d centroid positions discarded for initial guess" % \
              (len(xc_vec[~good]),nmeta)
        if len(xc_vec[good]) <= max(alphaDeg+1,ellDeg+1):
            raise ValueError('Not enough points for initial guesses')
    print_msg("  Reference position guess [%.0fA]: %.2f x %.2f spx" % 
              (lmid,xc,yc), 1)
    print_msg("  ADR guess: delta=%.2f, theta=%.1f deg" % 
              (delta0, theta0/N.pi*180), 1) # N.degrees() from python-2.5 only

    # 2) Other parameters
    PA     = N.median(PA_vec[good])
    # Polynomial-fit with 3-MAD clipping
    polEll = pySNIFS.fit_poly(ell_vec[good],3,ellDeg,lbda_rel[good])
    if opts.psf.endswith('powerlaw'):
        guessAlphaCoeffs = libES.powerLawFit(
            meta_cube.lbda[good]/lmid, alpha_vec[good], alphaDeg)
    else:
        # Polynomial-fit with 3-MAD clipping
        polAlpha = pySNIFS.fit_poly(alpha_vec[good],3,alphaDeg,lbda_rel[good])
        guessAlphaCoeffs = polAlpha.coeffs[::-1]

    # Filling in the guess parameter arrays (px) and bounds arrays (bx)
    p1     = [None]*(npar_psf+nmeta)
    p1[:5] = [delta0, theta0, xc, yc, PA]
    p1[5:6+ellDeg]        = polEll.coeffs[::-1]
    p1[6+ellDeg:npar_psf] = guessAlphaCoeffs
    p1[npar_psf:npar_psf+nmeta] = int_vec.tolist()

    # Bounds ------------------------------
    if opts.supernova:                  # Fix all parameters but intensities
        print "WARNING: supernova-mode, no 3D PSF-fit"
        # This mode completely discards 3D fit. In pratice, a 3D-fit
        # is still performed on intensities, just to be coherent w/
        # the remaining of the code.
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
        if opts.psf.endswith('powerlaw'):
            b1 += [[None, None]]*alphaDeg + [[0, None]] # a[-1] > 0
        else:
            b1 += [[0, None]] + [[None, None]]*alphaDeg # a0 > 0
    b1 += [[0, None]]*nmeta            # Intensities

    if opts.psf3Dconstraints:           # Read and set constraints from option
        setPSF3Dconstraints(opts.psf3Dconstraints, p1, b1)

    func = [ '%s;%f,%f,%f,%f' % 
             (psfFn.name, spxSize, lmid, alphaDeg, ellDeg) ] # PSF
    param = [p1]
    bounds = [b1]

    if skyDeg >= 0:
        p2 = N.ravel(sky_vec.T)
        b2 = ([[0,None]] + [[None,None]]*(npar_sky-1)) * nmeta
        func += ['poly2D;%d' % skyDeg]  # Add background
        param += [p2]
        bounds += [b2]

    print_msg("  Adjusted parameters: delta,theta,xc,yc,PA,"
              "%d ellCoeffs,%d alphaCoeffs,%d intensities, %d bkgndCoeffs" % 
              (ellDeg+1,alphaDeg+1,nmeta, npar_sky*nmeta if skyDeg>=0 else 0),
              2)
    print_msg("  Initial guess [PSF]: %s" % p1[:npar_psf], 2)
    print_msg("  Initial guess [Intensities]: %s" % 
              p1[npar_psf:npar_psf+nmeta], 3)
    if skyDeg >= 0:
        print_msg("  Initial guess [Bkgnd]: %s" % p2, 3)

    # Chi2 vs. Least-square fit
    if not opts.chi2fit:
        meta_cube.var = None    # Will be handled by pySNIFS_fit.model

    # Instantiate the model and perform the 3D-fit (fmin_tnc)
    data_model = pySNIFS_fit.model(data=meta_cube, func=func,
                                   param=param, bounds=bounds,
                                   myfunc={psfFn.name:psfFn})
    data_model.fit(maxfun=2000, save=True, msge=(opts.verbosity>=4))

    # Store guess and fit parameters
    fitpar = data_model.fitpar          # Adjusted parameters
    data_model.khi2 *= data_model.dof   # Restore real chi2 (or RSS)
    chi2 = data_model.khi2              # Total chi2 of 3D-fit
    covpar = data_model.param_covariance(fitpar) # Parameter covariance matrix
    dfitpar = N.sqrt(covpar.diagonal()) # Diagonal errors on adjusted parameters

    print_msg("  Fit result [%d]: chi2/dof=%.2f/%d" % 
              (data_model.status, chi2, data_model.dof), 1)
    print_msg("  Fit result [PSF param]: %s" % fitpar[:npar_psf], 2)
    print_msg("  Fit result [Intensities]: %s" % 
              fitpar[npar_psf:npar_psf+nmeta], 3)
    if skyDeg >= 0:
        print_msg("  Fit result [Background]: %s" % 
                  fitpar[npar_psf+nmeta:], 3)

    if opts.verbosity >= 3:
        print "Gradient checks:"
        data_model.check_grad()

    print_msg("  Reference position fit [%.0fA]: %.2f x %.2f spx" % 
              (lmid,fitpar[2],fitpar[3]), 1)
    adr.set_param(delta=fitpar[0], theta=fitpar[1]) # Update ADR params
    print_msg("  ADR fit: delta=%.2f, theta=%.1f deg" % 
              (adr.delta, adr.get_parangle()), 1)
    print "  Effective airmass: %.2f" % adr.get_airmass()

    # Compute seeing (FWHM in arcsec)
    seeing = data_model.func[0].FWHM(fitpar[:npar_psf], LbdaRef) * spxSize
    print '  Seeing estimate @%.0fA: %.2f" FWHM' % (LbdaRef,seeing)

    if not (0.4<seeing<4. and 1.<adr.get_airmass()<4.):
        raise ValueError('Unphysical seeing (%.2f") or airmass (%.3f)' % 
                         (seeing, adr.get_airmass()))

    # Test positivity of alpha and ellipticity. At some point, maybe it would
    # be necessary to force positivity in the fit (e.g. fmin_cobyla).
    if opts.psf.endswith('powerlaw'):
        fit_alpha = libES.powerLawEval(
            fitpar[6+ellDeg:npar_psf], meta_cube.lbda/lmid)
    else:
        fit_alpha = libES.polyEval(fitpar[6+ellDeg:npar_psf], lbda_rel)
    if fit_alpha.min() < 0:
        raise ValueError("Alpha is negative (%.2f) at %.0fA" % 
                         (fit_alpha.min(), meta_cube.lbda[fit_alpha.argmin()]))
    fit_ell = libES.polyEval(fitpar[5:6+ellDeg], lbda_rel)
    if fit_ell.min() < 0:
        raise ValueError("Ellipticity is negative (%.2f) at %.0fA" % 
                         (fit_ell.min(), meta_cube.lbda[fit_ell.argmin()]))

    # Computing final spectra for object and background ======================

    # Compute aperture radius
    if opts.method == 'psf':
        radius = None
        method = 'psf, %s' % ('chi2' if opts.chi2fit else 'least-squares')
    else:
        if opts.radius < 0:     # Aperture radius [sigma]
            radius = -opts.radius * seeing/2.355 # [arcsec]
            method = '%s r=%.1f sigma=%.2f"' % \
                     (opts.method, -opts.radius, radius)
        else:                   # Aperture radius [arcsec]
            radius = opts.radius # [arcsec]
            method = '%s r=%.2f"' % (opts.method, radius)
    print "Extracting the point-source spectrum (method=%s)..." % method
    if skyDeg < 0:
        print "WARNING: no background adjusted"

    # Spectrum extraction (point-source, sky, etc.) 
    psfCtes = [spxSize, lmid, alphaDeg, ellDeg]
    lbda,sigspecs,varspecs = libES.extract_specs(
        full_cube, (psfFn, psfCtes, fitpar[:npar_psf]),
        skyDeg=skyDeg, method=opts.method,
        radius=radius, chi2fit=opts.chi2fit,
        verbosity=opts.verbosity)

    if skyDeg >= 0:          # Convert sky spectrum to "per arcsec**2"
        sigspecs[:,1] /= spxSize**2
        varspecs[:,1] /= spxSize**4

    # Full covariance matrix of point-source spectrum
    if opts.covariance:
        print "Computing point-source spectrum covariance..."
        covspec = spec_covariance(full_cube,
                                  (psfFn, psfCtes, fitpar[:npar_psf]), skyDeg,
                                  covpar[:npar_psf,:npar_psf])

        # Add diagonal contribution from signal noise
        covspec += N.diag(varspecs[:,0])
        # Extract diagonal term
        varspecs[:,0] = covspec.diagonal()

    # Creating a standard SNIFS cube with the adjusted data
    # We cannot directly use data_model.evalfit() because 1. we want
    # to keep psf and bkg separated; 2. cube_fit will always have 225
    # spx, data_model.evalfit() might have less.  But in the end,
    # psf+bkg ~= data_model.evalfit()
    cube_fit = pySNIFS.SNIFS_cube(lbda=meta_cube.lbda) # Always 225 spx
    cube_fit.x = cube_fit.i - 7                        # x in spaxel
    cube_fit.y = cube_fit.j - 7                        # y in spaxel

    psf_model = psfFn(psfCtes, cube=cube_fit)
    psf = psf_model.comp(fitpar[:psf_model.npar])
    cube_fit.data = psf.copy()

    if skyDeg >= 0:
        bkg_model = pySNIFS_fit.poly2D(skyDeg, cube_fit)
        bkg = bkg_model.comp(
            fitpar[psf_model.npar:psf_model.npar+bkg_model.npar])
        cube_fit.data += bkg

    # Update header ==========================================================

    tflux = sigspecs[:,0].sum()     # Total flux of extracted spectrum
    if skyDeg >= 0:
        sflux = sigspecs[:,1].sum() # Total flux of sky (per arcsec**2)
    else:
        sflux = 0                   # Not stored anyway

    fill_header(inhdr, ', '.join((psfFn.model, psfFn.name)),
                fitpar[:npar_psf], adr, meta_cube, opts,
                chi2, seeing, (tflux,sflux))

    # Save star spectrum =====================================================

    print "Saving output point-source spectrum to '%s'" % opts.out

    # Store variance as extension to signal
    star_spec = pySNIFS.spectrum(data=sigspecs[:,0], var=varspecs[:,0],
                                 start=lbda[0], step=step)
    if opts.covariance: # Append covariance directly to pySNIFS.spectrum
        star_spec.cov = covspec
    star_spec.write_fits(opts.out, inhdr)

    # Save sky spectrum/spectra ==============================================

    if skyDeg >= 0:
        if not opts.sky:
            opts.sky = 'sky_%s.fits' % (channel)
        print "Saving output sky spectrum to '%s'" % opts.sky

        # Store variance as extension to signal
        sky_spec = pySNIFS.spectrum(data=sigspecs[:,1], var=varspecs[:,1],
                                    start=lbda[0], step=step)
        sky_spec.write_fits(opts.sky, inhdr)

    # Save 3D adjusted parameter file ========================================

    if opts.log3D:
        print "Producing 3D adjusted parameter logfile %s..." % opts.log3D
        create_3Dlog(opts, meta_cube, cube_fit, fitpar, dfitpar, chi2)

    # Save adjusted PSF ========================================================

    if opts.keepmodel:
        path,name = os.path.split(opts.out)
        outpsf = os.path.join(path,'psf_'+name)
        print "Saving adjusted meta-slice PSF in 3D-fits cube '%s'..." % outpsf
        cube_fit.WR_3d_fits(outpsf, header=[]) # No header in cube_fit

    # Create output graphics =================================================

    if opts.plot:
        print "Producing output figures [%s]..." % opts.graph

        import matplotlib as M
        backends = {'png':'Agg','eps':'PS','pdf':'PDF','svg':'SVG'}
        if opts.graph.lower() in backends:
            M.use(backends[opts.graph.lower()])
            basename = os.path.splitext(opts.out)[0]
            plot1 = os.path.extsep.join((basename+"_plt" , opts.graph))
            plot2 = os.path.extsep.join((basename+"_fit1", opts.graph))
            plot3 = os.path.extsep.join((basename+"_fit2", opts.graph))
            plot4 = os.path.extsep.join((basename+"_fit3", opts.graph))
            plot5 = os.path.extsep.join((basename+"_fit4", opts.graph))
            plot6 = os.path.extsep.join((basename+"_fit5", opts.graph))
            plot7 = os.path.extsep.join((basename+"_fit6", opts.graph))
            plot8 = os.path.extsep.join((basename+"_fit7", opts.graph))
        else:
            opts.graph = 'pylab'
            plot1 = plot2 = plot3 = plot4 = plot5 = plot6 = plot7 = plot8 = ''
        import pylab

        # Plot of the star and sky spectra -----------------------------------

        print_msg("Producing spectra plot %s..." % plot1, 1)

        fig1 = pylab.figure()

        if skyDeg >= 0 and sky_spec.data.any():
            axS = fig1.add_subplot(3, 1, 1)
            axB = fig1.add_subplot(3, 1, 2)
            axN = fig1.add_subplot(3, 1, 3)
        else:
            axS = fig1.add_subplot(2, 1, 1)
            axN = fig1.add_subplot(2, 1, 2)

        axS.text(0.95, 0.8, os.path.basename(opts.input),
                 fontsize='small', horizontalalignment='right',
                 transform=axS.transAxes)

        axS.plot(star_spec.x, star_spec.data, blue)
        axN.plot(star_spec.x, star_spec.data/N.sqrt(varspecs[:,0]), blue)

        if skyDeg >= 0 and sky_spec.data.any():
            axB.plot(sky_spec.x, sky_spec.data, green)
            axB.set(title=u"Background spectrum (per arcsec²)",
                    xlim=(sky_spec.x[0],sky_spec.x[-1]),
                    xticklabels=[])
            axN.plot(sky_spec.x, sky_spec.data/N.sqrt(varspecs[:,1]), green)

        axS.set(title="Point-source spectrum [%s, %s]" % (objname,method),
                xlim=(star_spec.x[0],star_spec.x[-1]), xticklabels=[])
        axN.set(title="Signal/Noise",
                xlabel=u"Wavelength [Å]",
                xlim=(star_spec.x[0],star_spec.x[-1]),
                yscale='log')

        fig1.tight_layout()
        if plot1:
            fig1.savefig(plot1)

        # Plot of the fit on each slice --------------------------------------

        print_msg("Producing slice fit plot %s..." % plot2, 1)

        ncol = int(N.floor(N.sqrt(nmeta)))
        nrow = int(N.ceil(nmeta/float(ncol)))

        fig2 = pylab.figure()
        fig2.suptitle("Slices plot [%s, airmass=%.2f]" % (objname,airmass),
                      fontsize='large')

        mod = data_model.evalfit()      # Total model (same nb of spx as cube)
        fmin = 0

        # Compute PSF & bkgnd models on incomplete cube
        sno = N.sort(meta_cube.no)
        psf2 = psfFn(psfCtes, cube=meta_cube).comp(fitpar[:psf_model.npar])
        if skyDeg >= 0 and sky_spec.data.any():
            bkg2 = pySNIFS_fit.poly2D(skyDeg, meta_cube).comp(
                fitpar[psf_model.npar:psf_model.npar+bkg_model.npar])

        for i in xrange(nmeta):        # Loop over meta-slices
            data = meta_cube.data[i,:]
            fit = mod[i,:]
            #fmin = min(data.min(),fit.min()) - max(data.max(),fit.max())/1e4
            ax = fig2.add_subplot(nrow, ncol, i+1,
                                  xlim=(0,len(data)), yscale='log')
            ax.plot(sno, data - fmin, color=blue, ls='-', lw=2)  # Signal
            ax.plot(sno, fit - fmin,  color=red, ls='-')         # Model
            ax.set_autoscale_on(False)
            if skyDeg >= 0 and sky_spec.data.any():
                ax.plot(sno, psf2[i,:] - fmin, color=green, ls='-') # PSF alone
                ax.plot(sno, bkg2[i,:] - fmin, color=orange, ls='-') # Bkgnd
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=8)
            ax.text(0.1,0.8, u"%.0f Å" % meta_cube.lbda[i],
                    fontsize='x-small', horizontalalignment='left',
                    transform=ax.transAxes)

            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Spaxel #", fontsize='small')
                ax.set_ylabel("Flux + cte", fontsize='small')
            ax.set_ylim(data.min()/1.2,data.max()*1.2)
            ax.set_xlim(-1,226)

        fig2.subplots_adjust(left=0.07, right=0.96, bottom=0.06, top=0.94)
        if plot2:
            fig2.savefig(plot2)

        # Plot of the fit on rows and columns sum ----------------------------

        print_msg("Producing profile plot %s..." % plot3, 1)

        if not opts.covariance:     # Plot fit on rows and columns sum

            fig3 = pylab.figure()
            fig3.suptitle("Rows and columns plot [%s, airmass=%.2f]" % 
                          (objname,airmass), fontsize='large')

            for i in xrange(nmeta):        # Loop over slices
                ax = fig3.add_subplot(nrow, ncol, i+1)

                # Signal
                sigSlice = meta_cube.slice2d(i, coord='p', NAN=False)
                prof_I = sigSlice.sum(axis=0) # Sum along rows
                prof_J = sigSlice.sum(axis=1) # Sum along columns
                # Errors
                if opts.chi2fit: # Chi2 fit: plot errorbars
                    varSlice = meta_cube.slice2d(
                        i, coord='p', var=True, NAN=False)
                    err_I = N.sqrt(varSlice.sum(axis=0))
                    err_J = N.sqrt(varSlice.sum(axis=1))
                    ax.errorbar(range(len(prof_I)),prof_I,err_I, 
                                fmt='o', c=blue, ecolor=blue, ms=3)
                    ax.errorbar(range(len(prof_J)),prof_J,err_J, 
                                fmt='^', c=red, ecolor=red, ms=3)
                else:            # Least-square fit
                    ax.plot(range(len(prof_I)),prof_I, 
                            marker='o', c=blue, ms=3, ls='None')
                    ax.plot(range(len(prof_J)),prof_J, 
                            marker='^', c=red, ms=3, ls='None')
                # Model
                modSlice = cube_fit.slice2d(i, coord='p')
                mod_I = modSlice.sum(axis=0)
                mod_J = modSlice.sum(axis=1)
                ax.plot(mod_I, ls='-', color=blue)
                ax.plot(mod_J, ls='-', color=red)

                pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(),fontsize=8)
                ax.text(0.1,0.8, u"%.0f Å" % meta_cube.lbda[i],
                        fontsize='x-small', horizontalalignment='left',
                        transform=ax.transAxes)
                if ax.is_last_row() and ax.is_first_col():
                    ax.set_xlabel("I (blue) or J (red)", fontsize='small')
                    ax.set_ylabel("Flux", fontsize='small')

                fig3.subplots_adjust(left=0.06, right=0.96,
                                     bottom=0.06, top=0.95)
        else:                           # Plot correlation matrices

            # Parameter correlation matrix
            corrpar = covpar / N.outer(dfitpar,dfitpar)
            parnames = data_model.func[0].parnames # PSF param names
            if skyDeg >= 0:                 # Add background param names
                coeffnames = [ "00" ] + \
                             [ "%d%d" % (d-j,j)
                               for d in range(1,skyDeg+1) for j in range(d+1) ]
                parnames += [ "b%02d_%s" % (s+1,c)
                              for c in coeffnames for s in range(nmeta) ]

            assert len(parnames)==corrpar.shape[0]
            # Remove some of the names for clarity
            parnames[npar_psf+1::2] = ['']*len(parnames[npar_psf+1::2])

            fig3 = pylab.figure(figsize=(7,6))
            ax3 = fig3.add_subplot(1,1,1,
                                   title="Parameter correlation matrix")
            im3 = ax3.imshow(N.absolute(corrpar),
                             vmin=1e-3, vmax=1,
                             norm=pylab.matplotlib.colors.LogNorm(),
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

        xguess = xc + delta0*psf_model.ADRcoeffs[:,0]*N.sin(theta0)
        yguess = yc - delta0*psf_model.ADRcoeffs[:,0]*N.cos(theta0)
        xfit = fitpar[2] + fitpar[0]*psf_model.ADRcoeffs[:,0]*N.sin(fitpar[1])
        yfit = fitpar[3] - fitpar[0]*psf_model.ADRcoeffs[:,0]*N.cos(fitpar[1])

        fig4 = pylab.figure()

        ax4c = fig4.add_subplot(2, 1, 1,
                                aspect='equal', adjustable='datalim',
                                xlabel="X center [spx]",
                                ylabel="Y center [spx]",
                                title="ADR plot [%s, airmass=%.2f]" %
                                (objname,airmass))
        ax4a = fig4.add_subplot(2, 2, 3,
                                xlabel=u"Wavelength [Å]",
                                ylabel="X center [spx]")
        ax4b = fig4.add_subplot(2, 2, 4,
                                xlabel=u"Wavelength [Å]",
                                ylabel="Y center [spx]")

        ax4a.errorbar(meta_cube.lbda[good], xc_vec[good], yerr=dparams[good,2],
                      marker='.', mfc=blue, mec=blue, capsize=0, ecolor=blue,
                      ls='None', label="Fit 2D")
        if bad.any():
            ax4a.plot(meta_cube.lbda[bad], xc_vec[bad],
                      mfc=red, mec=red, marker='.', ls='None', label='_')
        ax4a.plot(meta_cube.lbda, xguess, 'k--', label="Guess 3D")
        ax4a.plot(meta_cube.lbda, xfit, green, label="Fit 3D")
        pylab.setp(ax4a.get_xticklabels()+ax4a.get_yticklabels(), fontsize=8)
        leg = ax4a.legend(loc='best')
        pylab.setp(leg.get_texts(), fontsize='small')

        ax4b.errorbar(meta_cube.lbda[good], yc_vec[good], yerr=dparams[good,3],
                      marker='.', mfc=blue, mec=blue, ecolor=blue,
                      capsize=0, ls='None')
        if bad.any():
            ax4b.plot(meta_cube.lbda[bad],yc_vec[bad],
                      marker='.', mfc=red, mec=red, ls='None')
        ax4b.plot(meta_cube.lbda, yfit, green)
        ax4b.plot(meta_cube.lbda, yguess, 'k--')
        pylab.setp(ax4b.get_xticklabels()+ax4b.get_yticklabels(), fontsize=8)

        ax4c.errorbar(xc_vec[valid], yc_vec[valid],
                      xerr=dparams[valid,2], yerr=dparams[valid,3],
                      fmt=None, ecolor=green)
        ax4c.scatter(xc_vec[good],yc_vec[good], edgecolors='none',
                     c=meta_cube.lbda[good],
                     cmap=M.cm.Spectral_r, zorder=3)
        # Plot position selection process
        ax4c.plot(xref[good],yref[good], marker='.',
                  mfc=blue, mec=blue, ls='None') # Selected ref. positions
        ax4c.plot(xref[bad],yref[bad], marker='.',
                  mfc=red, mec=red, ls='None')   # Discarded ref. positions
        ax4c.plot((xref0,xc),(yref0,yc),'k-')
        ax4c.plot(xguess, yguess, 'k--') # Guess ADR
        ax4c.plot(xfit, yfit, green)     # Adjusted ADR
        ax4c.set_autoscale_on(False)
        ax4c.plot((xc,),(yc,),'k+')
        ax4c.add_patch(M.patches.Circle((xref0,yref0),radius=rmax,
                                                 ec='0.8',fc='None'))
        ax4c.add_patch(M.patches.Rectangle((-7.5,-7.5),15,15,
                                                 ec='0.8',lw=2,fc='None')) # FoV
        ax4c.text(0.03, 0.85,
                  u'Guess: x0,y0=%4.2f,%4.2f  airmass=%.2f parangle=%.1f°' % 
                  (xc, yc, airmass, parangle),
                  transform=ax4c.transAxes, fontsize='small')
        ax4c.text(0.03, 0.75,
                  u'Fit: x0,y0=%4.2f,%4.2f  airmass=%.2f parangle=%.1f°' % 
                  (fitpar[2], fitpar[3], adr.get_airmass(), adr.get_parangle()),
                  transform=ax4c.transAxes, fontsize='small')

        fig4.tight_layout()
        if plot4:
            fig4.savefig(plot4)

        # Plot of the other model parameters ---------------------------------

        print_msg("Producing model parameter plot %s..." % plot6, 1)

        guess_ell = N.polyval(polEll.coeffs,   lbda_rel)
        if opts.psf.endswith('powerlaw'):
            guess_alpha = libES.powerLawEval(
                guessAlphaCoeffs, meta_cube.lbda/lmid)
        else:
            guess_alpha = libES.polyEval(guessAlphaCoeffs, lbda_rel)

        # err_ell and err_alpha are definitely wrong, and not only
        # because they do not include correlations between parameters!
        err_PA = dfitpar[4]

        def plot_conf_interval(ax, x, y, dy):
            ax.plot(x, y, green, label="Fit 3D")
            if dy is not None:
                ax.plot(x, y+dy, ls=':', color=green, label='_')
                ax.plot(x, y-dy, ls=':', color=green, label='_')

        fig6 = pylab.figure()

        ax6a = fig6.add_subplot(2, 1, 1,
                                title='Model parameters '
                                '[%s, seeing %.2f" FWHM]' % (objname,seeing),
                                xticklabels=[],
                                ylabel=u'α [spx]')
        ax6b = fig6.add_subplot(4, 1, 3,
                                xticklabels=[],
                                ylabel=u'y² coeff.')
        ax6c = fig6.add_subplot(4, 1, 4,
                                xlabel=u"Wavelength [Å]",
                                ylabel=u'xy coeff.')

        # WARNING: the so-called PA parameter is not the PA of the
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

        ax6a.errorbar(meta_cube.lbda[good], alpha_vec[good], dparams[good,6],
                      marker='.',
                      mfc=blue, mec=blue, ecolor=blue, capsize=0, ls='None',
                      label="Fit 2D")
        if bad.any():
            ax6a.plot(meta_cube.lbda[bad], alpha_vec[bad],
                      marker='.', mfc=red, mec=red, ls='None', label="_")
        ax6a.plot(meta_cube.lbda, guess_alpha, 'k--', label="Guess 3D")
        #plot_conf_interval(ax6a, meta_cube.lbda, fit_alpha, err_alpha)
        plot_conf_interval(ax6a, meta_cube.lbda, fit_alpha, None)
        ax6a.text(0.03, 0.15,
                  'Guess: %s' % (', '.join(
                      [ 'a%d=%.2f' % (i,a) for i,a in
                        enumerate(guessAlphaCoeffs) ]) ),
                  transform=ax6a.transAxes, fontsize='small')
        ax6a.text(0.03, 0.05,
                  'Fit: %s' % (', '.join(
                      ['a%d=%.2f' % (i,a) for i,a in
                       enumerate(fitpar[6+ellDeg:npar_psf]) ]) ),
                  transform=ax6a.transAxes, fontsize='small')
        leg = ax6a.legend(loc='best')
        pylab.setp(leg.get_texts(), fontsize='small')
        pylab.setp(ax6a.get_yticklabels(), fontsize=10)

        ax6b.errorbar(meta_cube.lbda[good], ell_vec[good], dparams[good,5],
                      marker='.',
                      mfc=blue, mec=blue, ecolor=blue, capsize=0, ls='None')
        if bad.any():
            ax6b.plot(meta_cube.lbda[bad],ell_vec[bad],
                      marker='.', mfc=red, mec=red, ls='None')
        ax6b.plot(meta_cube.lbda, guess_ell, 'k--')
        #plot_conf_interval(ax6b, meta_cube.lbda, fit_ell, err_ell)
        plot_conf_interval(ax6b, meta_cube.lbda, fit_ell, None)
        ax6b.text(0.03, 0.3,
                  'Guess: %s' % (', '.join(
                      [ 'e%d=%.2f' % (i,e)
                        for i,e in enumerate(polEll.coeffs[::-1]) ]) ),
                  transform=ax6b.transAxes, fontsize='small')
        ax6b.text(0.03, 0.1,
                  'Fit: %s' % (', '.join(
                      [ 'e%d=%.2f' % (i,e)
                        for i,e in enumerate(fitpar[5:6+ellDeg]) ]) ),
                  transform=ax6b.transAxes, fontsize='small')
        pylab.setp(ax6b.get_yticklabels(), fontsize=10)

        ax6c.errorbar(meta_cube.lbda[good], PA_vec[good], dparams[good,4],
                      marker='.',
                      mfc=blue, mec=blue, ecolor=blue, capsize=0, ls='None')
        if bad.any():
            ax6c.plot(meta_cube.lbda[bad],PA_vec[bad],
                      marker='.', mfc=red, mec=red, ls='None')
        ax6c.plot([meta_cube.lstart,meta_cube.lend], [PA]*2, 'k--')
        plot_conf_interval(ax6c,
                           N.asarray([meta_cube.lstart,meta_cube.lend]),
                           N.ones(2)*fitpar[4], N.ones(2)*err_PA)
        ax6c.text(0.03, 0.1,
                  u'Guess: xy=%4.2f  Fit: xy=%4.2f' % (PA,fitpar[4]),
                  transform=ax6c.transAxes, fontsize='small')
        pylab.setp(ax6c.get_xticklabels() + ax6c.get_yticklabels(),
                   fontsize=10)

        fig6.subplots_adjust(left=0.1, right=0.96, bottom=0.08, top=0.95)
        if plot6:
            fig6.savefig(plot6)

        # Plot of the radial profile -----------------------------------------

        print_msg("Producing radial profile plot %s..." % plot7, 1)

        fig7 = pylab.figure()
        fig7.suptitle("Radial profile plot [%s, airmass=%.2f]" %
                      (objname,airmass), fontsize='large')

        def ellRadius(x,y, x0,y0, ell, q):
            dx = x - x0
            dy = y - y0
            # BEWARE: can return NaN's if ellipse is ill-defined
            return N.sqrt(dx**2 + ell*dy**2 + 2*q*dx*dy)

        def radialbin(r,f, binsize=20, weighted=True):
            rbins = N.sort(r)[::binsize] # Bin limits, starting from min(r)
            ibins = N.digitize(r, rbins) # WARNING: ibins(min(r)) = 1
            ib = N.arange(len(rbins))+1 # Bin index
            rb = N.array([ r[ibins==b].mean() for b in ib ]) # Mean radius
            if weighted:
                fb = N.array([ N.average(f[ibins==b], weights=r[ibins==b])
                               for b in ib ]) # Mean radius-weighted data
            else:
                fb = N.array([ f[ibins==b].mean() for b in ib ]) # Mean data
            # Error on bin mean quantities
            #snb = N.sqrt([ len(r[ibins==b]) for b in ib ]) # sqrt(nb of points)
            #drb = N.array([ r[ibins==b].std()/n for b,n in zip(ib,snb) ])
            #dfb = N.array([ f[ibins==b].std()/n for b,n in zip(ib,snb) ])
            return rb,fb

        fmin = 0
        for i in xrange(nmeta):        # Loop over slices
            ax = fig7.add_subplot(nrow, ncol, i+1, yscale='log')
            # Use adjusted elliptical radius instead of plain radius
            #r    = N.hypot(meta_cube.x-xfit[i], meta_cube.y-yfit[i])
            #rfit = N.hypot(cube_fit.x-xfit[i], cube_fit.y-yfit[i])
            r = ellRadius(meta_cube.x, meta_cube.y,
                          xfit[i],yfit[i], fit_ell[i], fitpar[4])
            rfit = ellRadius(cube_fit.x, cube_fit.y,
                             xfit[i],yfit[i], fit_ell[i], fitpar[4])
            #fmin = min(meta_cube.data[i].min(),cube_fit.data[i].min()) - \
            #       max(meta_cube.data[i].max(),cube_fit.data[i].max())/1e4
            ax.plot(r, meta_cube.data[i] - fmin,
                    marker=',', mfc=blue, mec=blue, ls='None') # Data
            ax.plot(rfit, cube_fit.data[i] - fmin,
                    marker='.', mfc=red, mec=red, ms=1, ls='None') # Model
            ax.set_autoscale_on(False)
            if skyDeg >= 0 and sky_spec.data.any():
                ax.plot(rfit, psf[i] - fmin, marker='.', mfc=green, mec=green,
                        ms=1, ls='None') # PSF alone
                ax.plot(rfit, bkg[i] - fmin, marker='.', mfc=orange, mec=orange,
                        ms=1, ls='None') # Sky
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=8)
            ax.text(0.9,0.8, u"%.0f Å" % meta_cube.lbda[i], fontsize='x-small',
                    horizontalalignment='right', transform=ax.transAxes)
            if opts.method != 'psf':
                ax.axvline(radius/spxSize, color=orange, lw=2)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Elliptical radius [spx]", fontsize='small')
                ax.set_ylabel("Flux + cte", fontsize='small')
            # ax.axis([0, rfit.max()*1.1,
            #          meta_cube.data[i][meta_cube.data[i]>0].min()/1.2,
            #          meta_cube.data[i].max()*1.2])

            # Binned values
            rb,db = radialbin(r, meta_cube.data[i])
            ax.plot(rb, db - fmin, 'c.')
            rfb,fb = radialbin(rfit, cube_fit.data[i])
            ax.plot(rfb, fb - fmin, 'm.')

        fig7.subplots_adjust(left=0.07, right=0.96, bottom=0.06, top=0.94)
        if plot7:
            fig7.savefig(plot7)

        # Missing energy (not activated by default)
        if opts.verbosity>=1:
            print_msg("Producing missing energy plot...", 1)
            
            figB = pylab.figure()
            for i in xrange(nmeta):        # Loop over slices
                ax = figB.add_subplot(nrow, ncol, i+1, yscale='log')
                r = ellRadius(meta_cube.x, meta_cube.y,
                              xfit[i],yfit[i], fit_ell[i], fitpar[4])
                rfit = ellRadius(cube_fit.x, cube_fit.y,
                                 xfit[i],yfit[i], fit_ell[i], fitpar[4])
                # Binned values
                rb,db  = radialbin(r,    meta_cube.data[i])
                rfb,fb = radialbin(rfit, cube_fit.data[i])
                tb = N.cumsum(rb*db)
                norm = tb.max()
                ax.plot(rb, 1 - tb/norm, 'c.')
                ax.plot(rfb, 1 - N.cumsum(rfb*fb)/norm, 'm.')
                if skyDeg >= 0 and sky_spec.data.any():
                    rfb,pb = radialbin(rfit, psf[i])
                    rfb,bb = radialbin(rfit, bkg[i])
                    ax.plot(rfb, 1 - N.cumsum(rfb*pb)/norm,
                            marker='.', mfc=green,   mec=green, ls='None')
                    ax.plot(rfb, 1 - N.cumsum(rfb*bb)/norm,
                            marker='.', mfc=orange, mec=orange, ls='None')
                pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(),
                           fontsize=8)
                ax.text(0.9,0.8, u"%.0f Å" % meta_cube.lbda[i],
                        fontsize='x-small', horizontalalignment='right',
                        transform=ax.transAxes)
                if opts.method != 'psf':
                    ax.axvline(radius/spxSize, color=orange, lw=2)
                if ax.is_last_row() and ax.is_first_col():
                    ax.set_xlabel("Elliptical radius [spx]",
                                  fontsize='small')
                    ax.set_ylabel("Missing energy [fraction]", fontsize='small')

            figB.tight_layout()
            if opts.graph != 'pylab':
                figB.savefig(os.path.extsep.join((basename+"_nrj", opts.graph)))

        # Radial Chi2 plot (not activated by default)
        if opts.verbosity>=1:
            print_msg("Producing radial chi2 plot...", 1)
            
            figA = pylab.figure()
            for i in xrange(nmeta):        # Loop over slices
                ax = figA.add_subplot(nrow, ncol, i+1, yscale='log')
                rfit = ellRadius(cube_fit.x, cube_fit.y,
                                 xfit[i],yfit[i], fit_ell[i], fitpar[4])
                chi2 = ( meta_cube.slice2d(i,coord='p') - 
                         cube_fit.slice2d(i,coord='p') )**2 / \
                         meta_cube.slice2d(i,coord='p',var=True)
                ax.plot(rfit, chi2.flatten(),
                        marker='.', ls='none', mfc=blue, mec=blue)
                pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(),
                           fontsize=8)
                ax.text(0.9,0.8, u"%.0f Å" % meta_cube.lbda[i],
                        fontsize='x-small', horizontalalignment='right',
                        transform=ax.transAxes)
                if opts.method != 'psf':
                    ax.axvline(radius/spxSize, color=orange, lw=2)
                if ax.is_last_row() and ax.is_first_col():
                    ax.set_xlabel("Elliptical radius [spx]",
                                  fontsize='small')
                    ax.set_ylabel(u"χ²", fontsize='small')

            figA.tight_layout()
            if opts.graph != 'pylab':
                figA.savefig(
                    os.path.extsep.join((basename+"_chi2", opts.graph)))
                
        # Contour plot of each slice -----------------------------------------

        print_msg("Producing PSF contour plot %s..." % plot8, 1)

        fig8 = pylab.figure()
        fig8.suptitle("Contour plot [%s, airmass=%.2f]" % (objname,airmass),
                      fontsize='large')

        extent = (meta_cube.x.min()-0.5, meta_cube.x.max()+0.5,
                  meta_cube.y.min()-0.5, meta_cube.y.max()+0.5)
        for i in xrange(nmeta):        # Loop over meta-slices
            ax = fig8.add_subplot(ncol, nrow, i+1, aspect='equal')
            data = meta_cube.slice2d(i, coord='p')
            fit  = cube_fit.slice2d(i, coord='p')
            vmin,vmax = pylab.prctile(fit, (5.,95.)) # Percentiles
            lev = N.logspace(N.log10(vmin),N.log10(vmax),5)
            ax.contour(data, lev, origin='lower', extent=extent)
            cnt = ax.contour(fit, lev, ls='--', origin='lower', extent=extent)
            pylab.setp(cnt.collections, linestyle='dotted')
            ax.errorbar((xc_vec[i],), (yc_vec[i],),
                        xerr=(dparams[i,2],), yerr=(dparams[i,3],),
                        fmt=None, ecolor='k' if good[i] else red)
            ax.plot((xfit[i],),(yfit[i],), marker='.', color=green)
            if opts.method != 'psf':
                ax.add_patch(M.patches.Circle((xfit[i],yfit[i]),
                                              radius/spxSize,
                                              fc='None', ec=orange, lw=2))
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=8)
            ax.text(0.1,0.1, u"%.0f Å" % meta_cube.lbda[i],
                    fontsize='x-small', horizontalalignment='left',
                    transform=ax.transAxes)
            ax.axis(extent)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I [spx]", fontsize='small')
                ax.set_ylabel("J [spx]", fontsize='small')
            if not ax.is_last_row():
                pylab.setp(ax.get_xticklabels(), visible=False)
            if not ax.is_first_col():
                pylab.setp(ax.get_yticklabels(), visible=False)

        fig8.subplots_adjust(left=0.05, right=0.96, bottom=0.06, top=0.95,
                             hspace=0.02, wspace=0.02)
        if plot8:
            fig8.savefig(plot8)

        # Residuals of each slice --------------------------------------------

        print_msg("Producing residual plot %s..." % plot5, 1)

        fig5 = pylab.figure()
        fig5.suptitle("Residual plot [%s, airmass=%.2f]" % (objname,airmass),
                      fontsize='large')

        images = []
        for i in xrange(nmeta):        # Loop over meta-slices
            ax   = fig5.add_subplot(ncol, nrow, i+1, aspect='equal')
            data = meta_cube.slice2d(i, coord='p') # Signal
            fit  = cube_fit.slice2d(i, coord='p') # Model
            if opts.chi2fit:    # Chi2 fit: display residuals in units of sigma
                var = meta_cube.slice2d(i, coord='p', var=True, NAN=False)
                res = N.nan_to_num((data - fit)/N.sqrt(var))
            else:               # Least-squares: display relative residuals
                res  = N.nan_to_num((data - fit)/fit)*100 # [%]

            # List of images, to be commonly normalized latter on
            images.append(ax.imshow(res, origin='lower', extent=extent,
                                    cmap=M.cm.jet, interpolation='nearest'))

            ax.plot((xfit[i],),(yfit[i],), marker='*', color='k')
            pylab.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=8)
            ax.text(0.1,0.1, u"%.0f Å" % meta_cube.lbda[i],
                    fontsize='x-small', horizontalalignment='left',
                    transform=ax.transAxes)
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
        vmin,vmax = pylab.prctile(N.ravel([ im.get_array().filled()
                                            for im in images ]), (3.,97.))
        norm = M.colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        # Colorbar
        cax = fig5.add_axes([0.90,0.07,0.02,0.87])
        cbar = fig5.colorbar(images[0], cax, orientation='vertical')
        pylab.setp(cbar.ax.get_yticklabels(), fontsize='small')
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
            pylab.show()

# End of extract_star.py =======================================================
