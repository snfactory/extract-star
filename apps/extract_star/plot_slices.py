#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import optparse
import pySNIFS
from ToolBox.MPL import get_backend

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"
__version__ = '$Id$'

if __name__=='__main__':

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
    parser.add_option("-g", "--graph", type="string",
                      help="Graphic output format [%default]",
                      default='pylab')

    opts,args = parser.parse_args()
    inname = args[0]
    basename,ext = os.path.splitext(os.path.basename(args[0]))

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

    import matplotlib as M
    backend,figname = get_backend(opts.graph, name='slices_%s' % args[0])
    if backend:
        M.use(backend)
    import pylab as P

    fig = P.figure(figsize=(10,8))
    fig.subplots_adjust(left=0.06, bottom=0.25, top=0.95,
                        hspace=0.03, wspace=0.03)

    fig.text(0.5, 0.97, "%s [%.0f A width]" % (basename,fcube.lstep*istep),
             fontsize='large', ha='center', va='center')

    ncol = P.floor(P.sqrt(cube.nslice))
    nrow = P.ceil(cube.nslice/float(ncol))
    extent = (cube.x.min()-0.5,cube.x.max()+0.5,
              cube.y.min()-0.5,cube.y.max()+0.5)

    if not opts.rangePerSlice:
        vmin,vmax = P.prctile(cube.data[P.isfinite(cube.var)], (fmin,fmax))
        print "Flux range [%.0f-%.0f%%]: %g,%g" % (fmin,fmax,vmin,vmax)

        fig.subplots_adjust(right=0.91)
    else:
        fig.subplots_adjust(right=0.95)


    # Loop over slices
    for i in xrange(cube.nslice):        # Loop over meta-slices
        ax = fig.add_subplot(ncol, nrow, i+1, aspect='equal')
        data = cube.slice2d(i, coord='p')

        if opts.rangePerSlice:
            var = cube.slice2d(i, coord='p', var=True)
            vmin,vmax = P.prctile(data[P.isfinite(var)], (fmin,fmax))
        
        im = ax.imshow(data,
                       origin='lower', extent=extent, interpolation='nearest',
                       vmin=vmin, vmax=vmax, cmap=M.cm.jet)
            
        ax.text(0.1,0.1, "%.0f A [%.0f-%.0f]" % \
                (cube.lbda[i],lbounds[i],lbounds[i+1]),
                fontsize=8, horizontalalignment='left', transform=ax.transAxes)
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

    # Spectrum
    spec = fcube.data.mean(axis=1)
    dspec = P.sqrt(fcube.var.sum(axis=1)/fcube.nlens**2)
    
    ax2 = fig.add_axes([0.06,0.06,0.88,0.15])
    
    ax2.plot(fcube.lbda, spec, 'b-')
    xp,yp = M.mlab.poly_between(fcube.lbda, spec-dspec, spec+dspec)
    ax2.fill(xp,yp, fc='b', ec='b', alpha=0.3)

    ax2.errorbar(cube.lbda, cube.data.mean(axis=1),
                 yerr=P.sqrt(cube.var.sum(axis=1))/cube.nlens,
                 fmt='go')
    
    ax2.set_xlim(fcube.lstart,fcube.lend)
    ax2.set_xlabel("Wavelength [A]", fontsize=8)
    ax2.set_ylabel("Flux", fontsize=8)
    P.setp(ax2.get_xticklabels()+ax2.get_yticklabels(), fontsize=8)
    
    # Metaslice boundaries
    for l in lbounds:
        ax2.axvline(l, c='0.8', zorder=0)
                
    if backend:
        print "Saving plot in", figname
        fig.savefig(figname)
    else:
        P.show()
