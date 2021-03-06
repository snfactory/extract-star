#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import optparse
import numpy as N
import pySNIFS
from ToolBox.MPL import get_backend, errorband
from ToolBox.ReST import rst_table
from ToolBox.Arrays import metaslice

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"
__version__ = '$Id$'

parser = optparse.OptionParser(version=__version__)

parser.add_option("-N", "--nmeta", type='int',
                  help="Number of meta-slices [%default].",
                  default=6)
parser.add_option("-r", "--range", type='string',
                  help="Flux range in percentile [%default].",
                  default='3,97')
parser.add_option("-R", "--rangePerCube", action='store_true',
                  help="Compute flux range for complete cube.",
                  default=False)
parser.add_option("-S", "--spatialStats", action='store_true',
                  help="Compute spatial statistics.",
                  default=False)
parser.add_option("-g", "--graph", type="string",
                  help="Graphic output format [%default]",
                  default='pylab')
parser.add_option("-V", "--variance", action='store_true',
                  help="Plot variance rather than signal.",
                  default=False)
parser.add_option("-s", "--spec", type='string',
                  help="Plotted spectra (mean|stack|none|l1[,l2]) [%default]",
                  default='mean')
parser.add_option("-T", "--title", type="string",
                  help="Plot title")
parser.add_option("-L", "--label", action='store_true',
                  help="Label spaxels.",
                  default=False)

opts,args = parser.parse_args()

try:
    fmin,fmax = [ float(t) for t in opts.range.split(',') ]
    assert 0<=fmin<100 and 0<fmax<=100 and fmin<fmax
except (ValueError,AssertionError,):
    parser.error("invalid option '-r/--range %s'" % opts.range)

nos = []
if opts.spec not in ['mean','stack','none']:
    try:
        nos = [ int(no) for no in opts.spec.split(',') ]
    except ValueError:
        parser.error("Invalid option '-s/--spec %s'" % opts.spec)

# Matplolib backend
import matplotlib as M
backend,figext = get_backend(opts.graph, name='')
if backend:
    M.use(backend)
import pylab as P

# Output
hdr = ["#/%d" % opts.nmeta, "lcen", "lmin", "lmax", "mean", "std", "[%]"]
fmt = ['%3s','%5.0f','%5.0f','%5.0f','%8.3g','%8.3g','%6.2f']
rows = []

for n,inname in enumerate(args):
    basename,ext = os.path.splitext(os.path.basename(inname))
    if len(args)>1:             # Multiple arguments
        rows += [[basename]]

    try:                                # Try to read a Euro3D cube
        fcube = pySNIFS.SNIFS_cube(e3d_file=inname)
        isE3D = True
    except ValueError:                  # Try to read a 3D FITS cube
        fcube = pySNIFS.SNIFS_cube(fits3d_file=inname)
        isE3D = False
    objname = fcube.e3d_data_header.get("OBJECT", 'unknown')
    print "%s [%s]: %d spaxels, %d slices [%.0f-%.0f A]" % \
        (basename, objname, fcube.nlens, fcube.nslice, fcube.lstart, fcube.lend)
    if fcube.var is None:
        print "WARNING: input cube has no variance"

    if fcube.nslice > 10*opts.nmeta:
        trim = min(10, int(0.05*fcube.nslice))
        print "Trimming: %d px" % trim
    else:
        trim = 0

    # Meta-slice definition (min,max,step [px])
    imin,imax,istep = metaslice(fcube.nslice, opts.nmeta, trim=trim)
    slices = [imin,imax,istep] # This is what pySNIFS.SNIFS_cube wants...

    # Metaslice boundaries: metaslice #i is ibounds[i]:ibounds[i+1]
    ibounds = range(imin,imax+1,istep)  # (fcube.nslice + 1,)

    if istep==1:
        cube = fcube    # No binning needed (buggy pySNIFS.SNIFS_cube)
    else:
        if opts.nmeta==1:               # Buggy pySNIFS.SNIFS_cube
            slices = [imin,imax,istep-1]
        if isE3D:
            cube = pySNIFS.SNIFS_cube(e3d_file=inname, slices=slices)
        else:
            cube = pySNIFS.SNIFS_cube(fits3d_file=inname, slices=slices)
    cube.x = cube.i - 7                 # From arcsec to spx
    cube.y = cube.j - 7

    fig = P.figure(figsize=(10,8))
    fig.subplots_adjust(left=0.06, top=0.96, hspace=0.03, wspace=0.03)
    if opts.spec != 'none':             # Leave room for spec plot
        fig.subplots_adjust(bottom=0.25)
    else:
        fig.subplots_adjust(bottom=0.05)

    efftime = fcube.e3d_data_header.get("EFFTIME", N.NaN)
    airmass = fcube.e3d_data_header.get("AIRMASS", N.NaN)
    if opts.title:
        title = opts.title
    else:
        title = u"%s [%s, %ds @%.2f], slices of %.0f Å" % \
                (basename, objname, efftime, airmass, fcube.lstep*istep)
    fig.suptitle(title, fontsize='large', y=0.99)

    ncol = int(N.floor(N.sqrt(cube.nslice)))
    nrow = int(N.ceil(cube.nslice/float(ncol)))
    extent = (cube.x.min()-0.5, cube.x.max()+0.5,
              cube.y.min()-0.5, cube.y.max()+0.5)

    if opts.rangePerCube:
        if opts.variance:
            if cube.var is None:
                parser.error("Cube %s has no variance" % basename)
            else:
                data = cube.var
        else:
            data = cube.data
        if cube.var is not None:
            vmin,vmax = N.percentile(data[N.isfinite(cube.var)], (fmin,fmax))
        else:
            vmin,vmax = N.percentile(data, (fmin,fmax))
        print "%s range [%.0f-%.0f%%]: %g,%g" % \
              (opts.variance and 'Variance' or 'Flux', fmin,fmax,vmin,vmax)
        fig.subplots_adjust(right=0.90)
    else:
        fig.subplots_adjust(right=0.95)

    # Loop over slices
    for i in xrange(cube.nslice):        # Loop over meta-slices
        ax = fig.add_subplot(ncol, nrow, i+1, aspect='equal')
        if opts.variance:
            data = cube.slice2d(i, coord='p', var=True)
        else:
            data = cube.slice2d(i, coord='p')

        gdata = data[N.isfinite(data)] # Non-NaN values
        m,s = gdata.mean(),gdata.std()
        rows += [[i+1, cube.lbda[i],
                  fcube.lbda[ibounds[i]], fcube.lbda[ibounds[i+1]-1],
                  m, s, s/m*100]]

        if not opts.rangePerCube:
            if cube.var is not None:
                var = cube.slice2d(i, coord='p', var=True)
                vmin,vmax = N.percentile(data[N.isfinite(var)], (fmin,fmax))
            else:
                vmin,vmax = N.percentile(data, (fmin,fmax))

        im = ax.imshow(data,
                       origin='lower', extent=extent, interpolation='nearest',
                       vmin=vmin, vmax=vmax)

        lbl = u"%.0f Å [%.0f-%.0f]" % (cube.lbda[i],
                                       fcube.lbda[ibounds[i]],
                                       fcube.lbda[ibounds[i+1]-1])
        if opts.spatialStats:
            lbl += "\nRMS=%.2f%%" % (s/m*100)
        ax.text(0.1, 0.1, lbl, 
                fontsize='small', ha='left', transform=ax.transAxes)
        ax.axis(extent)

        # Axis
        P.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize='x-small')
        if ax.is_last_row() and ax.is_first_col():
            ax.set_xlabel("X [spx]", fontsize='small')
            ax.set_ylabel("Y [spx]", fontsize='small')
        if not ax.is_last_row():
            P.setp(ax.get_xticklabels(), visible=False)
        if not ax.is_first_col():
            P.setp(ax.get_yticklabels(), visible=False)

        if opts.label and (i==0):
            for no,ii,jj in zip(cube.no,cube.i,cube.j):
                ax.text(ii-7, jj-7, str(no),
                        size='x-small', ha='center', va='center')

    # Colorbar
    if opts.rangePerCube:
        cax = fig.add_axes([0.92,0.25,0.02,0.7])
        cbar = fig.colorbar(im, cax, orientation='vertical')
        P.setp(cbar.ax.get_yticklabels(), fontsize='small')

    if opts.spec != 'none':             # Add a spectrum plot

        ax2 = fig.add_axes([0.07,0.06,0.88,0.15])

        if opts.spec == 'stack':        # Stacked spectra
            extent = (fcube.lstart-fcube.lstep/2, fcube.lend+fcube.lstep/2,
                      0.5, fcube.nlens+0.5)
            if opts.variance:
                vmin,vmax = N.percentile(fcube.var, (fmin,fmax))
                ax2.imshow(fcube.var.T, vmin=vmin, vmax=vmax, extent=extent)
            else:
                if fcube.var is not None:
                    vmin,vmax = N.percentile(fcube.data[N.isfinite(fcube.var)],
                                             (fmin,fmax))
                else:
                    vmin,vmax = N.percentile(fcube.data, (fmin,fmax))
                ax2.imshow(fcube.data.T, vmin=vmin, vmax=vmax, extent=extent)
            ax2.set_aspect('auto', adjustable='box')
            ax2.set_xlabel(u"Wavelength [Å]", fontsize='small')
            ax2.set_ylabel("Spx #", fontsize='small')
            P.setp(ax2.get_xticklabels()+ax2.get_yticklabels(),
                   fontsize='x-small')

        else:
            if opts.spec == 'mean':     # Mean spectrum
                if not opts.variance:
                    specs = [ fcube.data.mean(axis=1) ]
                    if fcube.var is not None:
                        dspecs = [ N.sqrt(fcube.var.sum(axis=1))/fcube.nlens ]
                        ax2.errorbar(
                            cube.lbda, cube.data.mean(axis=1),
                            yerr=N.sqrt(cube.var.sum(axis=1))/cube.nlens,
                            fmt='ko')
                    else:
                        dspecs = [ None ]
                        ax2.plot(cube.lbda, cube.data.mean(axis=1), 'ko')
                else:
                    specs = [ fcube.var.mean(axis=1) ]
                    dspecs = [ None ]
                fxlabel = "Mean variance" if opts.variance else "Mean flux"
                nos = [ -1 ]
            else:                       # Individual spectra
                if not opts.variance:
                    specs = [ fcube.spec(no=no) for no in nos ]
                    if fcube.var is not None:
                        dspecs = [
                            N.sqrt(fcube.spec(no=no, var=True)) for no in nos ]
                    else:
                        dspecs = [ None for no in nos ]
                else:
                    specs = [ fcube.spec(no=no, var=True) for no in nos ]
                    dspecs = [ None for no in nos ]
                fxlabel = "Variance" if opts.variance else "Flux"
            for no,spec,dspec in zip(nos,specs,dspecs):
                l, = ax2.plot(fcube.lbda, spec,
                              label='#%d' % no if no>=0 else '_')
                if dspec is not None:
                    col = l.get_color()
                    ax2.errorband(fcube.lbda, spec, dspec,
                                  fc=col, ec=col, alpha=0.3)
            ax2.set_xlim(fcube.lstart, fcube.lend)
            ax2.set_xlabel(u"Wavelength [Å]", fontsize='small')
            fxunits = fcube.e3d_data_header.get("FLXUNITS", 'none given')
            if fxunits.lower() != 'none given':
                fxlabel += " [%s]" % fxunits
                if opts.variance:
                    fxlabel += u"²"
            ax2.set_ylabel(fxlabel, fontsize='small')
            P.setp(ax2.get_xticklabels()+ax2.get_yticklabels(),
                   fontsize='x-small')
            ax2.legend(loc='best', fontsize='small')

        # Metaslice boundaries
        for i in ibounds[:-1]:
            ax2.axvline(fcube.lbda[i], c='0.8', zorder=5)
        ax2.axvline(fcube.lbda[ibounds[-1]-1], c='0.8', zorder=5)

    if backend:
        figname = ('slices_%s' % basename) + figext
        print "Saving plot in", figname
        fig.savefig(figname)

print rst_table(rows, fmt, hdr)

if not backend:
    P.show()
