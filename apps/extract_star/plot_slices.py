#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import optparse
import numpy as N
import pySNIFS
from ToolBox.MPL import get_backend
from ToolBox.ReST import rst_table

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"
__version__ = '$Id$'

parser = optparse.OptionParser(version=__version__)

parser.add_option("-N", "--nmeta", type='int',
                  help="Number of meta-slices [%default].",
                  default=12)
parser.add_option("-r", "--range", type='string',
                  help="Flux range in percentile [%default].",
                  default='3,97')
parser.add_option("-R", "--rangePerSlice", action='store_true',
                  help="Compute flux range per slice.",
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
parser.add_option("-s", "--stack", action='store_true',
                  help="Plot stacked spectra.",
                  default=False)

opts,args = parser.parse_args()

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

for inname in args:
    basename,ext = os.path.splitext(os.path.basename(inname))
    if len(args)>1:             # Multiple arguments
        rows += [[basename]]

    try:
        fmin,fmax = [ float(t) for t in opts.range.split(',') ]
        assert 0<=fmin<100 and 0<fmax<=100 and fmin<fmax
    except (ValueError,AssertionError):
        parser.error("invalid option '-r/--range %s'" % opts.range)

    try:                                # Try to read a Euro3D cube
        fcube = pySNIFS.SNIFS_cube(e3d_file=inname)
        isE3D = True
    except ValueError:                  # Try to read a 3D FITS cube
        fcube = pySNIFS.SNIFS_cube(fits3d_file=inname)
        isE3D = False
    print "%s: %d spaxels, %d slices [%.0f-%.0f A]" % \
        (basename, fcube.nlens, fcube.nslice, fcube.lstart, fcube.lend)

    # Meta-slice definition (min,max,step [px])
    imin = 10                           # 1st slice [px]
    imax = fcube.nslice - 10            # Last slice [px]
    istep = (imax-imin)//opts.nmeta     # Metaslice thickness [px]
    imax = imin + opts.nmeta*istep
    slices = [imin,imax,istep]

    if isE3D:
        cube = pySNIFS.SNIFS_cube(e3d_file=inname, slices=slices)
    else:
        cube = pySNIFS.SNIFS_cube(fits3d_file=inname, slices=slices)
    cube.x = cube.i - 7                 # From arcsec to spx
    cube.y = cube.j - 7

    lbounds = fcube.lbda[range(imin,imax+1,istep)]
    assert len(lbounds)==len(cube.lbda)+1

    fig = P.figure(figsize=(10,8))
    fig.subplots_adjust(left=0.06, bottom=0.25, top=0.95,
                        hspace=0.03, wspace=0.03)

    objname = fcube.e3d_data_header.get("OBJECT", 'unknown')
    fig.text(0.5, 0.97, 
             u"%s [%s], slices of %.0f Å" % (basename, objname,
                                             fcube.lstep*istep),
             fontsize='large', ha='center', va='center')

    ncol = int(P.floor(P.sqrt(cube.nslice)))
    nrow = int(P.ceil(cube.nslice/float(ncol)))
    extent = (cube.x.min()-0.5,cube.x.max()+0.5,
              cube.y.min()-0.5,cube.y.max()+0.5)

    if not opts.rangePerSlice:
        if opts.variance:
            data = cube.var
        else:
            data = cube.data
        vmin,vmax = P.prctile(data[P.isfinite(cube.var)], (fmin,fmax))
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
        #print "Slice #%02d/%d, %.0f Å [%.0f-%.0f]: " \
        #    "mean=%g, stddev=%g (%.2f%%)" % \
        #    (i+1,cube.nslice,cube.lbda[i],lbounds[i],lbounds[i+1],
        #     m,s,s/m*100)
        rows += [[i+1,cube.lbda[i],lbounds[i],lbounds[i+1],m,s,s/m*100]]

        if opts.rangePerSlice:
            var = cube.slice2d(i, coord='p', var=True)
            vmin,vmax = P.prctile(data[P.isfinite(var)], (fmin,fmax))

        im = ax.imshow(data,
                       origin='lower', extent=extent, interpolation='nearest',
                       vmin=vmin, vmax=vmax, cmap=M.cm.jet)

        lbl = u"%.0f Å [%.0f-%.0f]" % (cube.lbda[i],lbounds[i],lbounds[i+1])
        if opts.spatialStats:
            lbl += "\nRMS=%.2f%%" % (s/m*100)
        ax.text(0.1,0.1, lbl, 
                fontsize=9, horizontalalignment='left', 
                transform=ax.transAxes)
        ax.axis(extent)

        # Axis
        P.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=8)
        if ax.is_last_row() and ax.is_first_col():
            ax.set_xlabel("X [spx]", fontsize=8)
            ax.set_ylabel("Y [spx]", fontsize=8)
        if not ax.is_last_row():
            P.setp(ax.get_xticklabels(), visible=False)
        if not ax.is_first_col():
            P.setp(ax.get_yticklabels(), visible=False)

    # Colorbar
    if not opts.rangePerSlice:
        cax = fig.add_axes([0.92,0.25,0.02,0.7])
        cbar = fig.colorbar(im, cax, orientation='vertical')

    ax2 = fig.add_axes([0.07,0.06,0.88,0.15])

    if opts.stack:                  # Stacked spectra
        vmin,vmax = P.prctile(fcube.data[P.isfinite(fcube.var)], (fmin,fmax))
        ax2.imshow(fcube.data.T, vmin=vmin, vmax=vmax,
                   extent=(fcube.lstart,fcube.lend,0,fcube.nlens-1))
        ax2.set_aspect('auto', adjustable='box')
        ax2.set_xlabel(u"Wavelength [Å]", fontsize=8)
        ax2.set_ylabel("Spx #", fontsize=8)
        P.setp(ax2.get_xticklabels()+ax2.get_yticklabels(), fontsize=8)

    else:                           # Mean spectrum
        spec = fcube.data.mean(axis=1)
        dspec = P.sqrt(fcube.var.sum(axis=1)/fcube.nlens**2)

        ax2.plot(fcube.lbda, spec, 'b-')
        xp,yp = M.mlab.poly_between(fcube.lbda, spec-dspec, spec+dspec)
        ax2.fill(xp,yp, fc='b', ec='b', alpha=0.3)

        ax2.errorbar(cube.lbda, cube.data.mean(axis=1),
                     yerr=P.sqrt(cube.var.sum(axis=1))/cube.nlens,
                     fmt='go')

        ax2.set_xlim(fcube.lstart,fcube.lend)
        ax2.set_xlabel(u"Wavelength [Å]", fontsize=8)
        if opts.variance:
            fxlabel = "Variance"
        else:
            fxlabel = "Flux"
        fxunits = fcube.e3d_data_header.get("FLXUNITS", 'none given')
        if fxunits.lower() != 'none given':
            fxlabel += " [%s]" % fxunits
            if opts.variance:
                fxlabel += u"²"
        ax2.set_ylabel(fxlabel, fontsize=8)
        P.setp(ax2.get_xticklabels()+ax2.get_yticklabels(), fontsize=8)

    # Metaslice boundaries
    for l in lbounds:
        ax2.axvline(l, c='0.8', zorder=0)

    if backend:
        figname = ('slices_%s' % basename) + figext
        print "Saving plot in", figname
        fig.savefig(figname)

print rst_table(rows,fmt,hdr)

if not backend:
        P.show()
