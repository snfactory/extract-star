######################################################################
## Filename:      pySNIFS.py
## Version:       $Revision$
## Description:   
## Author:        Emmanuel Pecontal
## Author:        $Author$
## $Id$
######################################################################

import numpy as num
from scipy import interpolate as I
from scipy.ndimage import filters as F
import pyfits

__author__ = '$Author$'
__version__ = '$Revision$'
__docformat__ = "epytext en"

########################   I/O functions   #########################

class spectrum:
    """
    1D spectrum class.
    """
    def __init__(self,data_file=None,var_file=None,no=None,x=None,data=None,var=None,start=None,step=None,nx=None):
        """
        Initiating the class.
        @param data_file: fits file from which the data are read. It can be a
            1D fits image or a euro3d datacube. In the euro3d case, the
            variance spectrum is read from this cube if present.  In the 1D
            fits image file case, if this file contains two extensions, the
            variance spectrum is read from the second one.
        @param var_file: fits file from which the variance is read if it is
            not present in the data file.  It must be a 1D fits image.
        @param no: number of the spaxel in the datacube. Needed only if the
            input file is a euro3d cube.
        @param x: array containing the x coordinates in the spectrum. It is
            used only if the spectrum is not regularly sampled    
        @param data: array containing the data of the spectrum
        @param var: array containing the variance of the spectrum
        @param start: coordinate in user coordinates of the first point of the
            data array.
        @param step: step in user coordinates of the data.
        @param nx: number of data point. It is usefull when the user want to
            create an array of zero data. otherwise, this value is the size of
            the data array.
        @group parameters used when the spectrum is read from a fits spectrum
            or a euro3d datacube: data_file,var_file,no
        @group parameters used when the spectrum is not read from a file:
               x,data,var,start,step,nx
        """
        self.file = file
        if data_file is not None:
            data_fits = pyfits.open(data_file, ignore_missing_end=True)
            if 'EURO3D' in data_fits[0].header:
                # Case where the spectrum belongs to an e3d datacube
                if no is None:
                    raise ValueError('The user must provide the spectrum '
                                     'number in the datacube')
                else:
                    if no in data_fits[1].data.field('SPEC_ID'):
                        i = data_fits[1].data.field('SPEC_ID').tolist().index(no)
                        self.data = num.array(data_fits[1].data.field('DATA_SPE')[i],'f')
                        if 'STAT_SPE' in data_fits[1].columns.names:
                            self.var = num.array(data_fits[1].data.field('STAT_SPE')[i],'f')
                        else:
                            self.var = None 
                        self.len = data_fits[1].data.field('SPEC_LEN')[i]
                        self.step = data_fits[1].header.get('CDELTS')
                        self.start = data_fits[1].header.get('CRVALS') + \
                                     data_fits[1].data.field('SPEC_STA')[i] * \
                                     self.step
                        self.x = num.arange(self.len)*self.step + self.start
                    else:
                        self.data = None
                        self.var = None
                        self.len = None
                        self.step = None
                        self.start = None
                        self.x = None
                data_fits.close()
            else:
                # Case where the data and variance spectra are read from fits
                # files
                self.data = num.array(data_fits[0].data,'f')
                if len(data_fits) == 2:
                    # The data fits spectrum has an extension containing the
                    # variance spectrum
                    self.var = num.array(data_fits[1].data)
                elif var_file is not None:
                    # The variance is read from a fits file 
                    var_fits = pyfits.open(var_file, ignore_missing_end=True)
                    self.var = num.array(var_fits[0].data)
                else:
                    self.var = None
                if self.var is not None:
                    if len(self.var) != len(self.data):
                        raise ValueError('Data and variance spectra '
                                         'must have the same length')
                self.len = data_fits[0].header.get('NAXIS1')
                self.x = data_fits[0].header.get('CRVAL1') + \
                         num.arange(data_fits[0].header.get('NAXIS1')) * \
                         data_fits[0].header.get('CDELT1')
                self.step = data_fits[0].header.get('CDELT1')
                self.start = data_fits[0].header.get('CRVAL1')
        else:                   # data_file is None
            if x is None:
                # Case for a regularly sampled spectrum

                if None not in (start, step, nx):
                    self.start = start
                    self.step = step
                    self.len = nx
                    self.x = start + num.arange(nx)*step
                    if data is None:
                        self.data = num.zeros(nx)
                        self.var = num.zeros(nx)
                        self.x = start + num.arange(nx)*step
                    else:
                        print 'WARNING: When nx is given, the data array ' \
                              'is not taken into account and the ' \
                              'spectrum data is zet to zeros(nx)'
                                            
                else:           # One of start,step,nx is None
                    if data is None:
                        if nx is None:
                            raise ValueError('Not enough parameters to fill '
                                             'the spectrum data field')
                        else:
                            self.data = num.zeros(nx)
                        self.var = None
                        self.len = nx
                        if start is not None:
                            self.start = start
                        else:
                            self.start = 0
                        if step is not None:
                            self.step = step
                        else:
                            self.step = 1
                        self.x = self.start + num.arange(nx)*self.step
                        
                    else:       # data is not None
                        self.data = data
                        self.len = len(data)
                        if var is not None:
                            if len(var) != len(data):
                                raise ValueError('data and variance array must '
                                                 'have the same length')
                            else:
                                self.var = var
                        else:
                            self.var = None
                        if start is not None:
                            self.start = start
                        else:
                            self.start = 0
                        if step is not None:
                            self.step = step
                        else:
                            self.step = 1
                        self.x = self.start + num.arange(self.len)*self.step
                        
            else:
                # Case for a not regularly sampled spectrum
                self.start = None
                self.step = None
                self.len = len(x)
                if data is None:
                        self.data = num.zeros(len(x))
                        self.var = num.zeros(len(x))
                        self.x = x
                else:
                    if len(data) != len(x):
                        raise ValueError('x and data arrays must '
                                         'have the same size')
                    if var is not None:
                        if len(var) != len(data):
                            raise ValueError('data and var arrays must '
                                             'have the same size')
                    self.data = data
                    self.var = var
                    self.x = x
        
        self.data = num.array(self.data)
        self.x = num.array(self.x)
        if self.var is not None:
            self.var = num.array(self.var)
              
        if self.var is None:
            self.has_var = False
        else:
            self.has_var = True
        
        if self.len is not None:
            self.index_list = num.arange(self.len).tolist()
            self.intervals = [(0,self.len)]
         
        self.curs_val = []
        self.cid = None       
        
    def subset(self,intervals=None,reject=False):
        """
        Compute the indexes corresponding to the intervals bounds given in x coordinates and store them in the
        intervals field of the spectrum.
        @param intervals: A list of two elements tuples defining the intervals in x to be plotted
        @param reject: If set to True, the selection will be considered as negative. NOT YET IMPLEMENTED.  
        """
        self.index_list = []
        self.intervals = []
        if intervals is None:
            return

        if not isinstance(intervals,list):
            raise ValueError("Interval must be provided as a list of 2 elements tuples")

        for interval in intervals:
            if not isinstance(interval,tuple):
                raise ValueError("Intervals must be provided as a list of 2 elements tuples")
            if len(interval) == 2:
                if interval[0] < min(self.x) or interval[1] > max(self.x):
                    raise ValueError("Interval bounds must be numbers between xmin and xmax.")
                else:
                    ind_min = self.index(min(interval))
                    ind_max = self.index(max(interval))
                    self.index_list = self.index_list + (num.arange(ind_max-ind_min+1)+ind_min).tolist()
                    self.intervals.append((ind_min,ind_max))
            else:
                        raise ValueError("Interval must be provided as a list of 2 elements tuples")

    def reset_interval(self):
        """
        Erase the intervals selections
        """
        self.index_list = num.arange(self.len).tolist()
        self.intervals = [(1,self.len)]

 
     
    def index(self,x):
        """
        Return the index of the closest point to x in the spectrum
        """
        if min(self.x) > x or x > max(self.x):
            raise ValueError('x out of the spectrum x range')
        else:
            return num.argmin((self.x - x)**2,axis=-1)

                        
    def WR_fits_file(self,filename,header_list=None):
        """
        Write the spectrum in a fits file.
        @param filename: Name of the output fits file
        @param header_list: List of 2 elements lists containing the name and value of the header to be
            saved. If set to None, only the mandatory fits header will be stored.
        """
        if self.start is None or self.step is None or self.data is None:
            raise ValueError('Only regularly sampled spectra can be saved as fits files.')
        
        hdu = pyfits.PrimaryHDU()
        hdu.data = num.array(self.data)
        hdu.header['NAXIS'] = 1
        hdu.header.set('NAXIS1', self.len, after='NAXIS')
        hdu.header['CRVAL1'] = self.start
        hdu.header['CDELT1'] = self.step
        if header_list is not None:
            for desc in header_list:
                if desc[0][0:5] not in ['TUNIT', 'TTYPE', 'TFORM', 'TDISP', 'NAXIS',
                                        'CRVAL', 'CDELT', 'CRPIX'] and \
                   desc[0] not in ['EXTNAME', 'XTENSION', 'GCOUNT', 'PCOUNT', 'BITPIX',
                                   'CTYPES', 'CRVALS', 'CDELTS', 'CRPIXS', 'TFIELDS']:
                    hdu.header[desc[0]] = desc[1]

        hdulist = pyfits.HDUList()
        hdulist.append(hdu)

        if self.has_var:
            hdu_var = pyfits.ImageHDU(self.var, name='VARIANCE')
            hdu_var.header['NAXIS'] = 1
            hdu_var.header.set('NAXIS1', self.len, after='NAXIS')
            hdu_var.header['CRVAL1'] = self.start
            hdu_var.header['CDELT1'] = self.step
            hdulist.append(hdu_var)

        hdulist.writeto(filename, clobber=True) # Overwrite


class spec_list:
    def __init__(self,slist):
        for s in slist:
            if not isinstance(s,spectrum):
                raise TypeError('The list elements must be pySNIFS.spectrum objects')
        self.list = slist
        
    def WR_fits_file(self,filename,header_list=None):
                
        hdulist = pyfits.HDUList()
        hdu = pyfits.PrimaryHDU()
        if header_list is not None:
            for desc in header_list:
                if desc[0][0:5] not in ['TUNIT', 'TTYPE', 'TFORM', 'TDISP', 'NAXIS'] and \
                   desc[0] not in ['EXTNAME', 'XTENSION', 'GCOUNT', 'PCOUNT', 'BITPIX',
                                   'CTYPES', 'CRVALS', 'CDELTS', 'CRPIXS', 'TFIELDS']:
                    hdu.header[desc[0]] = desc[1]
        hdulist.append(hdu)
        for s in self.list:
            hdu = pyfits.ImageHDU()
            hdu.data = num.array(s.data)
            hdu.header['NAXIS'] = 1
            hdu.header.set('NAXIS1', s.len, after='NAXIS')
            hdu.header['CRVAL1'] = s.start
            hdu.header['CDELT1'] = s.step
            hdulist.append(hdu)
            if s.has_var:
                hdu_var = pyfits.ImageHDU()
                hdu_var.data = num.array(s.var)
                hdu_var.header['NAXIS'] = 1
                hdu_var.header.set('NAXIS1', s.len, after='NAXIS')
                hdu_var.header['CRVAL1'] = s.start
                hdu_var.header['CDELT1'] = s.step
                hdulist.append(hdu_var)
        hdulist.writeto(filename, clobber=True) # Overwrite

########################## Image #######################################

class image_array:
    """
    2D image class
    """
    def __init__(self,data_file=None,data=None,var=None,startx=1,stepx=1,starty=1,stepy=1,endx=None,endy=None,labx=None,laby=None,header=None):
        """
        Initiating the class.
        @param data_file: 2D fits image. If given, the data and all the image parameters will be read from this file.
        @param data: 2D array containing the data
        @param var: optional array containing the variance (not yet used)
        @param startx: coordinate in user coordinates of the first point of the data array for the second
            dimension.
        @param starty: coordinate in user coordinates of the first point of the data array for the first
            dimension.
        @param endx: coordinate in user coordinates of the last point of the data array for the second
            dimension. If not provided, it will be computed from startx and stepx
        @param endy: coordinate in user coordinates of the last point of the data array for the first
            dimension.If not provided, it will be computed from starty and stepy
        @param stepx: step in user coordinates of the data for the second dimension. If not provided, it
            will be computed from startx and endx
        @param stepy: step in user coordinates of the data for the first dimension. If not provided, it
            will be computed from starty and endy
        @param labx: String containing the x unit (used for plotting)
        @param laby: String containing the y unit (used for plotting)
        @param header: List of 2 elements lists containing the headers name and values to be saved in
            fits file.
        """
        if data_file is not None:
            data_fits = pyfits.open(data_file, ignore_missing_end=True)
            if data_fits[0].header.get('NAXIS') != 2:
                raise TypeError('This is not a 2D fits image')
            else:
                self.startx = data_fits[0].header.get('CRVAL1')
                self.starty = data_fits[0].header.get('CRVAL2')
                self.stepx = data_fits[0].header.get('CDELT1')
                self.stepy = data_fits[0].header.get('CDELT2')
                self.nx = data_fits[0].header.get('NAXIS1')
                self.ny = data_fits[0].header.get('NAXIS2')
                self.endx = self.startx + (self.nx-1)*self.stepx
                self.data = num.array(data_fits[0].data,'f')
                self.endy = self.starty + (self.ny-1)*self.stepy
                self.header = data_fits[0].header.items()
                self.var = var
        else:
            if (stepx is None and endx is None) or startx is None:
                raise ValueError('The user must provide either startx and stepx or startx and endx')
            if (stepy is None and endy is None) or starty is None:
                raise ValueError('The user must provide either starty and stepy or starty and endy')

            if data is None:
                raise ValueError('The user must provide a data array.')
            elif len(num.shape(data)) != 2:
                raise ValueError('The user must provide a two dimensions data array.')

            self.data = data
            self.var = var
            self.startx = startx
            self.starty = starty
            self.nx = num.shape(data)[1]
            self.ny = num.shape(data)[0]
            if endx is None:
                self.stepx = stepx
                self.endx = startx + (self.nx-1)*stepx
            else:
                self.endx = endx
                self.stepx = float(endx-startx)/(self.nx-1)
            if endy is None:
                self.stepy = stepy
                self.endy = starty + (self.ny-1)*stepy
            else:
                self.endy = endy
                self.stepy = float(endy-starty)/(self.ny-1)            
            
            self.header = header
            
        self.labx = labx
        self.laby = laby
        self.vmin = float(self.data.min())
        self.vmax = float(self.data.max())

        
    def WR_fits_file(self,file_name,mode='w+'):
        """
        Write the image in a fits file.
        @param file_name: name of the fits file
        @param mode: writing mode. if w+, the file is overwritten. Otherwise the writing will fail. 
        """
        hdulist = pyfits.HDUList()
        hdu = pyfits.PrimaryHDU()
        hdu.header['NAXIS'] = 2
        hdu.header.set('NAXIS1', self.nx, after='NAXIS')
        hdu.header.set('NAXIS2', self.ny, after='NAXIS1')
        hdu.header['CRPIX1'] = 1
        hdu.header['CRVAL1'] = self.startx
        hdu.header['CDELT1'] = self.stepx
        hdu.header['CRPIX2'] = 1
        hdu.header['CRVAL2'] = self.starty
        hdu.header['CDELT2'] = self.stepy
        if self.header is not None:
            for desc in self.header:
                if desc[0] not in ['SIMPLE', 'BITPIX', 'EXTEND', 'NAXIS', 'NAXIS1',
                                   'NAXIS2', 'CRPIX1', 'CRVAL1', 'CDELT1', 'CRPIX2',
                                   'CRVAL2', 'CDELT2']:
                    hdu.header[desc[0]] = desc[1]
        hdu.data = num.array(self.data)
        hdulist.append(hdu)
        hdulist.writeto(file_name, clobber=(mode=='w+'))


########################## Cube ########################################

class SNIFS_cube:
    """
    SNIFS datacube class.
    """

    spxSize = 0.43                      # Spx size in arcsec
    
    def __init__(self,e3d_file=None,fits3d_file=None,slices=None,lbda=None,threshold=1e20,nodata=False):
        """
        Initiating the class.
        @warning: If the spectra in the datacube do not have the same lengthes, they will be truncated
            to the larger common range.

        @param e3d_file: Euro3d fits file
        @param slices: slices range that will be extracted from the euro3d file and stored in the object.
            This is either a list of two integers [lmin,lmax], or a list of three integer [lmin,lmax,lstep].
            In this last case, the slices will be coadded by stacks of lstep size.
        @param lbda: Array of lambdas if the datacube is created from scratch...
            This case is not really implemented yet...
        @param threshold: In the variance image, pixels where variance is not available for some reason are
            set to an arbitrarily high value. As this values seems to change from one version to another of
            the processing pipeline, we allow to pass it as the threshold parameter.
        @param nodata: If set to True, only the descriptors (start,step...) will be saved. It may be useful when the user
            wants to read many cubes to compare their descriptors without using to much memory.
        """

        if slices is not None:
            if not isinstance(slices,list):
                raise ValueError('The wavelength range must be given as a list of two or three integer positive values')
            if len(slices) != 2 and len(slices) != 3:
                raise ValueError('The wavelength range must be given as a list of two or three integer positive values')
            if False in [isinstance(sl,int) for sl in slices]:
                raise ValueError('The wavelength range must be given as a list of two or three integer positive values')
            if False in [sl>=0 for sl in slices]:
                raise ValueError('The wavelength range must be given as a list of two or three integer positive values')
            if len(slices) == 3:
                if slices[2] == 0:
                    raise ValueError('The slices step cannot be set to 0')
                else:
                    s = True
            else:
                s = False
            if slices[0] > slices[1]:
                tmp = slices[0]
                slices[0] = slices[1]
                slices[1] = tmp
            l = [int(sl) for sl in slices]
        else:
            l = slices
            s = False

        self.data = None
        self.var = None
                
        if e3d_file is not None:
            e3d_cube = pyfits.open(e3d_file, ignore_missing_end=True)
            gen_header = dict(e3d_cube[0].header.items())
            if 'EURO3D' not in gen_header or \
                   gen_header['EURO3D'] not in ('T', pyfits.TRUE):
                raise ValueError("Invalid E3d file ('EURO3D' keyword)")
            
            self.from_e3d_file = True
            self.e3d_file = e3d_file
            # The header of the data extension is stored in a field of the
            # class
            self.e3d_data_header = dict(e3d_cube[1].header.items())
            # The group definition HDU and the optional extensions HDU are
            # stored in fields of the class
            self.e3d_grp_hdu = e3d_cube[2]
            self.e3d_extra_hdu_list = [ e3d_cube[i]
                                        for i in xrange(3,len(e3d_cube)) ]
            ref_start = e3d_cube[1].header['CRVALS']
            step      = e3d_cube[1].header['CDELTS']
            self.lstep = step
            if 'STAT_SPE' in e3d_cube[1].columns.names:
                var = e3d_cube[1].data.field('STAT_SPE')
            else:
                var = None
            data  = e3d_cube[1].data.field('DATA_SPE')
            spec_sta  = e3d_cube[1].data.field('SPEC_STA')
            spec_len  = e3d_cube[1].data.field('SPEC_LEN')
            spec_end  = spec_len + spec_sta
            #npts      = e3d_cube[1].data.getshape()[0]
            common_lstart,common_lend,lstep = max(spec_sta),min(spec_end),1
            common_start = ref_start + common_lstart * step

            tdata = num.array([ data[i][common_lstart-spec_sta[i]: \
                                        common_lend-spec_sta[i]] \
                                for i in range(len(data)) ]).T
            if var is not None:
                tvar = num.array([ var[i][common_lstart-spec_sta[i]: \
                                          common_lend-spec_sta[i]] \
                                   for i in range(len(var)) ]).T
                                                          
                # In the variance image, pixels where variance is not
                # available for some reason are set to an arbitrarily high
                # value. As this values seems to change from one version to
                # another of the processing pipeline, we allow to pass it as a
                # parameter: threshold
                tvar = num.where((tvar>threshold),threshold,tvar)
                tvar = num.where((tvar<-threshold),threshold,tvar)
            lbda = num.arange(len(tdata))*step + common_start

            if l is not None:
                lmin,lmax = max([common_lstart,l[0]]),min([common_lend,l[1]])
                if len(l) == 3:
                    lstep = l[2]
                else:
                    lstep=1
            else:
                lmin,lmax = common_lstart,common_lend
                lstep = 1

            if lmax-lmin < lstep:
                raise ValueError('Slice step incompatible with requested slices interval')
                         
            if not s:
                if not nodata:
                    self.data = tdata[lmin - common_lstart: \
                                      lmax - common_lstart:lstep]
                    if var is not None:
                        self.var = tvar[lmin - common_lstart: \
                                        lmax - common_lstart:lstep]
                self.lbda = lbda[lmin - common_lstart: \
                                 lmax - common_lstart:lstep]
            else:
                if not nodata:
                    self.data = F.uniform_filter(tdata,(lstep,1)) \
                                [lmin - common_lstart + lstep/2: \
                                 lmax - common_lstart+lstep/2:lstep]
                    if var is not None:
                        self.var = F.uniform_filter(tvar,(lstep,1)) \
                                   [lmin - common_lstart + lstep/2: \
                                    lmax - common_lstart+lstep/2:lstep]/lstep
                self.lbda = lbda[lmin - common_lstart+lstep/2: \
                                 lmax - common_lstart+lstep/2:lstep]
            self.lstep *= lstep
            self.lstart = self.lbda[0]
            
            self.x = e3d_cube[1].data.field('XPOS')
            self.y = e3d_cube[1].data.field('YPOS')
            self.no = e3d_cube[1].data.field('SPEC_ID')
            # We read in the extension table the I and J index of the
            # lenslets. As the extension table may have a different number of
            # lenslets than the data table, we first search the index of the
            # common lenslets.
            if len(e3d_cube)>3 and e3d_cube[3].name=='TIGERTBL':
                nos = e3d_cube[3].data.field('NO').tolist()
                ind = [ nos.index(i) for i in self.no ]
                self.i = e3d_cube[3].data.field('I')[ind] + 7
                self.j = e3d_cube[3].data.field('J')[ind] + 7
            else:
                # There's no native I,J spx coords, compute them from arcsec
                # coords X,Y
                self.i = num.round(self.x / self.spxSize).astype('i') + 7
                self.j = num.round(self.y / self.spxSize).astype('i') + 7
                
            self.nslice = len(self.lbda)
            self.lend = self.lstart + (self.nslice-1)*self.lstep
            self.nlens = len(self.x)
            
        elif fits3d_file is not None:
            
            self.from_e3d_file = False

            fits3d_cube = pyfits.open(fits3d_file, ignore_missing_end=True)
            # Get keywords from primary header
            gen_header = dict(fits3d_cube[0].header.items())
            if gen_header['NAXIS'] != 3:
                raise ValueError("Invalid 3D file: NAXIS=%d != 3" % \
                                 gen_header['NAXIS'])

            self.e3d_data_header = gen_header # Not really e3d_data_header...
            nslice = gen_header['NAXIS3']
            nx = gen_header['NAXIS1']
            ny = gen_header['NAXIS2']
            startx = gen_header['CRVAL1']
            starty = gen_header['CRVAL2']
            startl = gen_header['CRVAL3']
            stepx = gen_header['CDELT1']
            stepy = gen_header['CDELT2']
            stepl = gen_header['CDELT3']
            self.fits3d_file = fits3d_file
            self.lstep = stepl

            lbda = num.arange(nslice)*stepl + startl
            data = num.reshape(fits3d_cube[0].data[:,:,::-1],(nslice,nx*ny))
            if 'VARIANCE' in [ h.name for h in fits3d_cube ]:
                assert fits3d_cube[0].data.shape == \
                       fits3d_cube['VARIANCE'].data.shape, \
                       "Variance and primary extensions have different shapes"
                var = num.reshape(fits3d_cube['VARIANCE'].data[:,:,::-1],
                                  (nslice,nx*ny))
            else:
                var = None

            if l is not None:
                lmin,lmax = max([0,l[0]]),min([nslice-1,l[1]])
                if len(l) == 3:
                    lstep = l[2]
                else:
                    lstep=1
            else:
                lmin,lmax = 0,nslice-1
                lstep = 1

            if lmax-lmin < lstep:
                raise ValueError('Slice step incompatible with '
                                 'requested slices interval')

            if not s:
                if not nodata:
                    self.data = data
                    if var is not None:
                        self.var = var
                self.lbda = lbda
            else:
                if not nodata:
                    self.data = F.uniform_filter(num.array(data,'d'),
                                                 (lstep,1))[lmin + lstep/2:
                                                            lmax +lstep/2:
                                                            lstep]
                    if var is not None:
                        self.var = F.uniform_filter(num.array(var,'d'),
                                                    (lstep,1))[lmin + lstep/2:
                                                               lmax+lstep/2:
                                                               lstep] / lstep
                self.lbda = lbda[lmin+lstep/2:lmax+lstep/2:lstep]
            self.lstep *= lstep
            self.lstart = self.lbda[0]
            self.i = nx - 1-num.ravel(num.indices((nx,ny))[1]) 
            self.j = num.ravel(num.indices((nx,ny))[0])
            self.x = self.i*stepx+startx
            self.y = self.j*stepy+starty
            self.no = nx*(num.arange(nx*ny)%nx)+num.arange(nx*ny)//nx+1

            # Search the indexes of the spectra containing only nan
            ind = num.where(num.min(num.isnan(num.transpose(self.data)),1)==False)[0]
            self.data = num.transpose(num.transpose(self.data)[ind])
            if self.var is not None:
               self.var = num.transpose(num.transpose(self.var)[ind])
            self.i = self.i[ind]
            self.j = self.j[ind]
            self.x = self.x[ind]
            self.y = self.y[ind]
            self.no = self.no[ind]
 
            self.nslice = len(self.lbda)
            self.lend = self.lstart + (self.nslice-1)*self.lstep
            self.nlens = len(self.x)
            
        else:
            # If neither euro3d nor fits3d file is given, we create an SNIFS
            # cube from
            self.from_e3d_file = False
            self.nlens = 225
            if lbda is None:
                self.data = num.zeros(self.nlens,num.float32)
                self.lbda = None  
                self.nslice = None
            else:
                # Check 1st that lbda is a linear ramp
                self.lbda = num.array(lbda)
                delta = self.lbda[1:] - self.lbda[:-1]
                self.lstep = delta.mean()
                if not num.allclose(delta, self.lstep):
                    raise ValueError("Input wavelength ramp is not linear.")
                self.lstart = lbda[0]
                self.lend = lbda[-1]
                self.nslice = len(lbda)
                data = num.zeros((self.nslice,self.nlens),num.float64)
                i,j = [ arr.ravel()
                        for arr in num.meshgrid(num.arange(15),
                                                num.arange(14,-1,-1)) ]
                x = (i-7)*self.spxSize
                y = (j-7)*self.spxSize
                no = num.arange(1,self.nlens+1).reshape(15,15).T.ravel()
                self.data = num.array(data,'f')
                self.x = num.array(x)
                self.y = num.array(y)
                self.i = num.array(i)
                self.j = num.array(j)
                self.no = num.array(no)
          
    def slice2d(self,n=None,coord='w',weight=None,var=False,nx=15,ny=15,NAN=True):
        """
        Extract a 2D slice (individual slice or average of several ones) from a cube and return it as an array
        @param n: If n is a list of 2 values [n1,n2], the function returns the sum of the slices between n1
            and n2 if it is an integer n , it returns the slice n
        @param coord: The type of coordinates:
                      - 'w' -> wavelength coordinates
                      - 'p' -> pixel coordinates
        @param nx: dimension of the slice in x
        @param ny: dimension of the slice in y
        @param weight: Spectrum giving the weights to be applied to each slice before lambda integration
        @param NAN: Flag to set non existing spaxel value to nan. if NAN=False, non existing spaxels will be set to 0
        """
        if weight is None and n is None:
            raise ValueError("Slices to be averaged must be given either as a list or as a weight spectrum")
        
        if weight is None:
            if isinstance(n,list):
                if len(n) != 2:
                    raise ValueError("The list must have 2 values")
                if n[0] > n[1]:
                    n[1],n[0] = n[0],n[1]
                if coord == 'p':
                    n1,n2 = n
                elif coord == 'w':
                    n1 = num.argmin((self.lbda-n[0])**2,axis=-1)
                    n2 = num.argmin((self.lbda-n[1])**2,axis=-1)
                else:
                    raise ValueError("Coord. flag should be 'p' or 'w'")
                if n1 == n2:
                    n2 += 1
            else:
                if coord == 'p':
                    if n%1 != 0:
                        print "WARNING: slice index %s rounded to int." % \
                              (str(n))
                    n = int(num.round(n))
                    n1 = n
                    n2 = n+1
                elif coord == 'w':
                    n1 = num.argmin((self.lbda-n)**2,axis=-1)
                    n2 = n1+1
                else:
                    raise ValueError("Coord. flag should be 'p' or 'w'")

            if n1 >= 0 and n2 <= num.shape(self.data)[0]:
                slice_2D = num.zeros((nx,ny),num.float32)
                if NAN:
                    slice_2D *= num.nan
                i = num.array(self.i)
                j = num.array(self.j)
                if var:
                    if var is True: # Boolean
                        arr = self.var
                    else:   # Attribute string
                        arr = getattr(self,var)
                else:
                    arr = self.data
                slice_2D[j,i] = num.sum(arr[n1:n2],axis=0)

                return slice_2D
            else:
                raise IndexError("No slice #%d" % n)
        else:
            if not isinstance(weight,spectrum):
                raise TypeError("The weights must be a pySNIFS.spectrum")
            else:
                if max(weight.x) < self.lstart or min(weight.x) > self.lend:
                    raise ValueError("The weight spectrum range is outside the cube limits")
                else:
                    lmin = max(self.lstart,min(weight.x))
                    lmax = min(self.lend,max(weight.x))
                    imin = int((lmin-self.lstart)/self.lstep)
                    imax = int((lmax-self.lstart)/self.lstep)
                    lbda = self.lstart+num.arange(imax-imin)*self.lstep
                    tck = I.splrep(weight.x,weight.data,s=0)
                    w = I.splev(lbda,tck)
                    slice_2D = num.zeros((nx,ny),num.float32)
                    if NAN:
                        slice_2D *= num.nan
                    i = num.array(self.i)
                    j = num.array(self.j)
                    if var:
                        if var is True: # Boolean
                            arr = self.var
                        else:   # Attribute string
                            arr = getattr(self,var)
                    else:
                        arr = self.data
                    slice_2D[j,i] = num.sum((arr[imin:imax,:].T*w).T,axis=0)

                    return slice_2D
                
    def spec(self,no=None,ind=None,mask=None,var=False):
        """
        Extract a spectrum from the datacube and return it as a 1D array.
        @param no: lenslet number in the datacube
        @param ind: index in the data array
        @param mask: optional array having the same shape than the data field of the cube. If given, the
            spectrum returned is the sum of the data array multiplied by the mask.
        @param var: Variance flag. If set to True, the variance spectrum is extracted.
        """
        if var:
            data = self.var
        else:
            data = self.data
            
        if mask is not None:
            if num.shape(mask) != num.shape(data):
                raise ValueError('mask array must have the same shape than the data array')
            data = data * mask
            return num.sum(data,1)
        
        if (no is not None and ind is not None) or \
               (no is None and ind is None):
            raise TypeError("lens number (no) OR spec index (ind) should be given.")
        else:
            if (not isinstance(no,list) and not isinstance(no,num.ndarray)) \
                   and no is not None:
                if no is not None:
                    if no in self.no.tolist():
                        return data[:,num.argmax(self.no == no,axis=-1)]
                    else:
                        raise IndexError("no lens #%d" % no)
            elif no is not None:
                s = num.sum(num.array([num.transpose(data)[self.get_lindex(n)] for n in no]),axis=0)
                return s
            else:
                if not isinstance(ind,list):
                    if 0 <= ind < num.shape(data)[1]:
                        return data[:,ind]
                    else:
                        raise IndexError("no index #%d" % ind)
                else:
                    if 0 <= ind[0] and ind[1] < num.shape(data)[1]:
                        ind[1] += 1
                        return num.sum(data[:,ind[0]:ind[1]],1)
                    else:
                        raise IndexError("Index list out of range")
                        
    def get_spec(self,no):
        """
        Extract the spectrum corresponding to lenslet no and return it as a pySNIFS.spectrum object
        @param no: lenslet number of the spectrum to be extracted
        """
        spec = spectrum(x=self.lbda,data=self.spec(no),
                        var=self.spec(no,var=True))
        
        if hasattr(self,'lstep'):
            spec.step = self.lstep
        if hasattr(self,'lstart'):
            spec.start = self.lstart

        return spec
    
    def get_no(self,i,j):
        """
        Get the lenslet number from its coordinates i,j 
        """
        if i>max(self.i) or i<min(self.i) or j>max(self.j) or j<min(self.j):
            raise ValueError("Index out of range.")
        no = self.no[num.argmax((self.i == i)*(self.j == j),axis=-1)]
        return no

    def get_ij(self,no):
        """
        Get the lenslet coordinates i,j from its number no
        """
        if no>max(self.no) or no<min(self.no):
            raise ValueError("Lens number out of range.")
        i = self.i[num.argmax(self.no == no,axis=-1)]
        j = self.j[num.argmax(self.no == no,axis=-1)]
        return i,j

    def get_lindex(self,val):
        """
        Return the index of the spec (i,j) or no in the stacked array data
        @param val: tuple (i,j) or integer no defining the lenslet
        """
        if isinstance(val,tuple):
            if not min(self.i)<=val[0]<=max(self.i) or \
               not min(self.j)<=val[1]<=max(self.j):
                raise ValueError("Index out of range.")
            ind = num.argmax((self.i == val[0]) & (self.j == val[1]),axis=-1)
        else:
            ind = num.argmax(self.no == val,axis=-1)

        return ind

    def WR_e3d_file(self, filename):
        """
        Write the datacube as a euro3d fits file.
        @param filename: Name of the output file
        """
        if not self.from_e3d_file:
            raise NotImplementedError("Writing e3d file from scratch not yet implemented")
        data_list = num.transpose(self.data)
        if self.var is not None:
            var_list = num.transpose(self.var)
        else:
            var_list = None
        start_list = [self.lstart for i in xrange(self.nlens)]
        no_list = self.no.tolist()
        xpos_list = self.x.tolist()
        ypos_list = self.y.tolist()
        WR_e3d_file(data_list,var_list,no_list,start_list,self.lstep,
                    xpos_list,ypos_list, filename, self.e3d_data_header,
                    self.e3d_grp_hdu,self.e3d_extra_hdu_list,nslice=self.nslice)

    def WR_3d_fits(self, filename, header=None, mode='w+'):
        """
        Write the datacube in a NAXIS=3 FITS file.
        @param filename: name of the output FITS file
        @param mode: writing mode. If w+, the file is overwritten;
                     otherwise the writing will fail. 
        """
        if header is None:
            header = self.e3d_data_header.copy()
            for h in self.e3d_data_header:
                if h in ('XTENSION','BITPIX','GCOUNT','PCOUNT','TFIELDS',
                         'EXTNAME','TFORM','TUNIT','TDISP',
                         'CTYPES','CRVALS','CDELTS','CRPIXS') or \
                         h[:5] in ('NAXIS','TTYPE'):
                    del header[h]

        hdulist = pyfits.HDUList()

        hdu_data = pyfits.PrimaryHDU()
        hdu_data.data = num.array([ self.slice2d(i,coord='p')
                                    for i in xrange(self.nslice) ])
        for h in header:
            hdu_data.header[h] = header[h]
        stepx  = stepy  = self.spxSize
        startx = starty = -7*self.spxSize
        hdu_data.header['CRVAL1'] = startx
        hdu_data.header['CRVAL2'] = starty
        hdu_data.header['CRVAL3'] = self.lstart
        hdu_data.header['CDELT1'] = stepx
        hdu_data.header['CDELT2'] = stepy
        hdu_data.header['CDELT3'] = self.lstep
        hdu_data.header['CRPIX1'] = 1
        hdu_data.header['CRPIX2'] = 1
        hdu_data.header['CRPIX3'] = 1
        hdulist.append(hdu_data)

        if self.var is not None:
            hdu_var = pyfits.ImageHDU(name='VARIANCE')
            hdu_var.data = num.array([ self.slice2d(i,coord='p',var=True)
                                       for i in xrange(self.nslice) ])
            hdu_var.header['CRVAL1'] = startx
            hdu_var.header['CRVAL2'] = starty
            hdu_var.header['CRVAL3'] = self.lstart
            hdu_var.header['CDELT1'] = stepx
            hdu_var.header['CDELT2'] = stepy
            hdu_var.header['CDELT3'] = self.lstep
            hdu_var.header['CRPIX1'] = 1
            hdu_var.header['CRPIX2'] = 1
            hdu_var.header['CRPIX3'] = 1
            hdulist.append(hdu_var)
            
        hdulist.writeto(filename,clobber=(mode=='w+'))

        
#####################     SNIFS masks     ########################

class SNIFS_mask:
    """
    SNIFS extraction mask class
    """
    def __init__(self,mask_file,offx=0,offy=0,step=200,order=1):
        """
        Initiating the class.
        @param mask_file: mask fits file to be read
        @param offx,offy: offset in x and y to be given to the mask.
        @param step: step in pixels along the dispersion direction at which the mask is computed. The other
            values will be interpolated
        @param order: diffraction order (0,1 or 2). The useful signal is expected at order 1.
        
        """

        import commands, os

        def runCmd(command, grep=None):
            """Run 'command' with proper status check. Display grepped output
            and raise RuntimeError if status. Return status."""

            status,output = commands.getstatusoutput(command)
            if status:                  # Something went wrong
                if grep:                # Grepped output
                    print '\n'.join([ l for l in output.split('\n')
                                      if grep in l ])
                else:                   # Full output
                    print output
                raise RuntimeError("Error %d during '%s'." % \
                                   (status, command.split()[0]))
            return status

        # Copy mask to tempfile for total selection
        runCmd('cp -f %s tmp_mask.fits' % mask_file)
        os.chmod('tmp_mask.fits', 0644) # quickmasks are 0444
        runCmd('sel_table -in tmp_mask.fits -sel all', grep='ERROR')
        runCmd('plot_optics -mask tmp_mask.fits -offset %f,%f -orders %d,%d ' \
               '-step %d -local -table tmp_tbl.fits ' \
               '-inputformat euro3d -outputformat euro3d' % \
               (offx,offy,order,order,step), grep='ERROR')

        mask_tbl = pyfits.open('tmp_mask.fits', ignore_missing_end=True)
        tmp_tbl = pyfits.open('tmp_tbl.fits', ignore_missing_end=True)
        self.no = mask_tbl[1].data.field('no').tolist()
        self.lbda_list = []
        i = 1
        while 'LBDA%d'%i in tmp_tbl[1].header:
            self.lbda_list.append(tmp_tbl[1].header.get('LBDA%d'%i))
            i += 1

        self.x = {}
        self.y = {}
        for i,l in enumerate(self.lbda_list):
            self.x[l] = tmp_tbl[1].data.field('XLBDA%d'%(i+1))
            self.y[l] = tmp_tbl[1].data.field('YLBDA%d'%(i+1))

    def get_spec_lens(self,no):
        """
        Get the x,y positions of lens  #no for each wavelength in lbda_list
        @param no: lens number
        """
        i = self.no.index(no)
        x = [self.x[l][i] for l in self.lbda_list]
        y = [self.y[l][i] for l in self.lbda_list]
        return x,y

    def get_coord_lens(self,no,lbda):
        """
        Get the x,y position of lens #no for wavelength lbda. If lbda is not in lbda_list, the position is
        interpolated.
        @param no: lens number
        @param lbda: wavelength
        """
        x,y = self.get_spec_lens(no)
        tckx = I.splrep(self.lbda_list,x,s=0)
        tcky = I.splrep(self.lbda_list,y,s=0)
        x = I.splev(lbda,tckx)
        y = I.splev(lbda,tcky)
        return x,y
    
    def interpolate(self,no,yy):
        """
        Interpolate the spectra x position for spectrum #no at position yy
        @param no: lens number
        @param yy: y position where to compute the x value
        """
        if no not in self.no:
            raise ValueError('lens #%d not present in mask'%no)
        x,y = self.get_spec_lens(no)
        x = num.array(x)[num.argsort(y)]
        y = num.sort(y)
        tck = I.splrep(y,x,s=0)
        return I.splev(yy,tck)


#####################  Utility functions  ########################
       
def convert_tab(table,colx,coly,ref_pos):
    """
    Interpolate a spline in a set of (x,y) points where x and y are given by two columns of
    a pyfits table.
    @param table: Input table
    @param colx: x column in the table
    @param coly: y column in the table
    @param ref_pos: array giving the x positions where to compute the interpolated values
    """
    tck = I.splrep(table[1].data.field(colx),
                   table[1].data.field(coly),s=0)
    tab = I.splev(ref_pos,tck)
    return tab

def histogram(data,nbin=None,Min=None,Max=None,bin=None,cumul=False):
    """
    Compute the histogram of an array
    @param data: Input array
    @param nbin: number of bins in the histogram
    @param Min,Max: Interval of values between which the histogram is computed
    @param bin: Size of the bins. If not given it is computed from the number of bins requested.
    @param cumul: If True, compute a cumulative histogram
    """
    if Min is None:
        Min = min(data)
    if Max is None:
        Max = max(data)
    if bin is None:
        bin = (Max-Min)/nbin
        
    bin_array = num.arange(nbin)*bin + Min
    n = num.searchsorted(num.sort(data), bin_array)
    n = num.concatenate([n, [len(data)]])
    data = n[1:]-n[:-1]
    x = bin_array
    hist = spectrum(data=data,x=x)
    
    hist.len = len(bin_array)
    if cumul:
        hist.data = num.array([ float(num.sum(hist.data[0:i+1]))
                                for i in xrange(hist.len) ]) / \
                                float(num.sum(hist.data))
    return hist

def common_bounds_cube(cube_list):
    """
    Computes the common bounds of a list of pySNIFS cubes
    @param cube_list: Input list of spectra
    @return : imin,imax: lists of the indexes of the lower/upper common bound for each cube
    """
    eps=1e-8
    if False in [hasattr(cube,'lstep') for cube in cube_list]:
        raise ValueError("Missing attribute lstep in datacubes.")
    else:
        meanstep = num.mean([cube.lstep for cube in cube_list])
        if num.std([cube.lstep for cube in cube_list]) > eps*meanstep:
            raise ValueError("All cubes should have the same step.")
        xinf = max([min(cube.lbda) for cube in cube_list])
        xsup = min([max(cube.lbda) for cube in cube_list])
        imin = [num.argmin(abs(cube.lbda-xinf),axis=-1) for cube in cube_list]
        imax = [num.argmin(abs(cube.lbda-xsup),axis=-1) for cube in cube_list]

    return imin,imax

def common_bounds_spec(spec_list):
    """
    Computes the common bounds of a list of pySNIFS spectra
    @param spec_list: Input list of spectra
    @return: imin,imax: lists of the indexes of the lower/upper common bound for each spectrum
    """
    eps=1e-8
    if False in [hasattr(spec,'step') for spec in spec_list]:
        raise ValueError("Missing attribute lstep in spectra.")
    else:
        meanstep = num.mean([spec.step for spec in spec_list])
        if num.std([spec.step for spec in spec_list]) > eps*meanstep:
            raise ValueError("All spectra should have the same step.")
        xinf = max([min(spec.x) for spec in spec_list])
        xsup = min([max(spec.x) for spec in spec_list])
        imin = [num.argmin(abs(spec.x-xinf),axis=-1) for spec in spec_list]
        imax = [num.argmin(abs(spec.x-xsup),axis=-1) for spec in spec_list]

    return imin,imax

def common_lens(cube_list):
    """
    Give the common lenses of a list of pySNIFS cubes
    @param cube_list: input list of datacubes
    @return: inters: list of the lenses common to all the cubes of the list
    """
    inters = cube_list[0].no
    for i in xrange(1,len(cube_list)):
        inters = filter(lambda x:x in cube_list[i].no,inters) 
    return inters

def fit_poly(y,n,deg,x=None):
    """
    Fit a polynomial whith median sigma clipping on an array y
    @param y: Input array giving the ordinates of the points to be fitted
    @param n: rejection threshold (MAD)
    @param deg: Degree of the polynomial
    @param x: Optional input array giving the abscissae of the points to be fitted. If not given
       the abscissae are taken as an array [1:len(y)]
    """
    if x is None:
        x = num.arange(len(y))
    else:
        x = num.asarray(x)
    y = num.asarray(y)
    old_l = 0
    l = len(y)
    while l != old_l:        
        old_l = len(y)
        poly = num.poly1d(num.polyfit(x,y,deg))
        delta = num.abs(poly(x) - y)
        ind = delta < n*num.median(delta)
        x = x[ind]
        y = y[ind]
        l = len(x)
        if l<deg+1:
            print "WARNING(pySNIFS.fit_poly): not enough points to make a fit!"
            break
    return poly

def WR_e3d_file(data_list, var_list, no_list,
                start_list, step, xpos_list, ypos_list,
                filename, data_header, grp_hdu, extra_hdu_list, nslice=None):
    """
    Write a data cube as a euro3d file on disk.
       @param data_list: List of the spectra of the datacube
       @param var_list: List of the variance spectra of the data_cube
       @param no_list: List of the spaxel ident number of the datacube
       @param start_list: List of the start wavelength of each spectrum of the datacube
       @param step: Wavelength step of the datacube
       @param xpos_list: List of the x position of each spaxel of the datacube
       @param ypos_list: List of the x position of each spaxel of the datacube
       @param filename: Output fits file to be written on disk
       @param data_header: data header of the new e3d file. All the standards e3d
           headers containing information on the data themselves will be overwritten. Only the non standard
           will be copied from this parameter.
       @param grp_hdu: pyfits HDU describing the spaxel groups.
       @param extra_hdu_list: pyfits HDU containing non e3d-mandatory data.
       @param nslice: number of wavelengthes slices in the datacube. If not given, all the spectra of the datacube may have
                      different lengthes
    """
    
    pri_hdu = pyfits.PrimaryHDU()
    hdulist = pyfits.HDUList([pri_hdu,grp_hdu]+extra_hdu_list)
    
    start = max(start_list)
    spec_sta = [int((s - start)/step+0.5*num.sign(s-start)) for s in start_list]
    spec_len = [len(s) for s in data_list]
    selected = [0]*len(data_list)
    group_n =  [1]*len(data_list) 
    nspax =    [1]*len(data_list) 
    spax_id =  [' ']*len(data_list) 
    
    col_list = [pyfits.Column(name='SPEC_ID', format='J', array=no_list),
                pyfits.Column(name='SELECTED', format='J', array=selected),
                pyfits.Column(name='NSPAX', format='J', array=nspax),
                pyfits.Column(name='SPEC_LEN', format='J', array=spec_len),
                pyfits.Column(name='SPEC_STA', format='J', array=spec_sta),
                pyfits.Column(name='XPOS', format='E', array=xpos_list),
                pyfits.Column(name='YPOS', format='E', array=ypos_list),
                pyfits.Column(name='GROUP_N', format='J', array=group_n),
                pyfits.Column(name='SPAX_ID', format='1A1', array=spax_id)]
    if nslice is None:
        col_list.append(pyfits.Column(name='DATA_SPE',format='PD()',
                                      array=num.array(data_list, dtype='O')))
        qual = num.array([[0 for i in d] for d in data_list], dtype='O')
        col_list.append(pyfits.Column(name='QUAL_SPE',format='PJ()',
                                      array=qual))
        if var_list is not None:
            col_list.append(pyfits.Column(name='STAT_SPE',format='PD()',
                                          array=num.array(var_list, dtype='O')))
    else:
        col_list.append(pyfits.Column(name='DATA_SPE',format='%dD()'%nslice,
                                      array=data_list))
        col_list.append(pyfits.Column(name='QUAL_SPE',format='%dJ'%nslice,
                                      array=[[0 for i in d] for d in data_list]))
        if var_list is not None:
            col_list.append(pyfits.Column(name='STAT_SPE',format='%dD()'%nslice,
                                          array=var_list))

    tb_hdu = pyfits.new_table(col_list)
    tb_hdu.header['CTYPES'] = ' '
    tb_hdu.header['CRVALS'] = start
    tb_hdu.header['CDELTS'] = step
    tb_hdu.header['CRPIXS'] = 1
    tb_hdu.header['EXTNAME'] = 'E3D_DATA'
    
    for desc in data_header:
        if desc not in ('XTENSION','BITPIX','GCOUNT','PCOUNT','TFIELDS',
                        'EXTNAME','TFORM','TUNIT','TDISP',
                        'CTYPES','CRVALS','CDELTS','CRPIXS') and \
                        desc[:5] not in ('NAXIS','TTYPE','TFORM'):
            tb_hdu.header[desc] = data_header[desc]

    pri_hdu = pyfits.PrimaryHDU()
    pri_hdu.header['EURO3D'] = pyfits.TRUE
    pri_hdu.header['E3D_ADC'] = pyfits.FALSE
    pri_hdu.header['E3D_VERS'] = '1.0'
    hdu_list = pyfits.HDUList([pri_hdu,tb_hdu,grp_hdu]+extra_hdu_list)
    hdu_list.writeto(filename, clobber=True) # Overwrite

def gaus_array(ima_shape,center,sigma,I,pa=None):
    """
    Return a 1D or 2D array containing a gaussian.
    @param ima_shape: either an integer or a tuple of two integer defining the size of the array
    @param center: either a number or a tuple of two number giving the center of the gaussian (in pixels unit)
    @param sigma: either a number or a tuple of two number giving the dispersion of the gaussian (in pixels
        unit)
    @param I: intensity at maximum of the gaussian
    @param pa: For a 2D image, the position angle of the gaussian
    """
    if isinstance(ima_shape,int):
        if (not isinstance(center,float)) or (not isinstance(sigma,float)):
            raise TypeError('For 1D image, center and sigma must be float')
    else:
        if (not isinstance(ima_shape,list)) or (not isinstance(center,list)) or\
               (not isinstance(sigma,list)):
            raise TypeError('Shape should be given as a list of integers')
        if len(ima_shape) != 2:
            raise ValueError('More than 2D images not supported')
        if len(ima_shape) != len(center) or len(ima_shape) != len(sigma):
            raise ValueError('center and sigma lists must have the same length than shape list')
    if isinstance(ima_shape,int):
        xc = center
        s = sigma
        x = num.arange(ima_shape)-xc
        gaus = I*num.exp(-0.5*(x/s)**2)
    else:
        if pa is None:
            raise ValueError('Position angle must be supplied by the user')
        pa = pa*num.pi/180.
        nx,ny = ima_shape
        xc,yc = center
        sx,sy = sigma
        x,y = num.indices((nx,ny))      # (nx,ny)
        xr = (x-xc)*num.cos(pa) - (y-yc)*num.sin(pa)
        yr = (x-xc)*num.sin(pa) + (y-yc)*num.cos(pa)
        val = (xr/sx)**2 + (yr/sy)**2
        gaus = I*num.exp(-val/2)

    return gaus

def comp_cdg(ima):

    ima = num.abs(ima).astype('d')
    x,y = num.indices(num.shape(ima))
    norm = ima.sum()
    xc = (ima*x).sum()/norm
    yc = (ima*y).sum()/norm
    sx = num.sqrt(((ima*(x-xc)**2)).sum()/norm)
    sy = num.sqrt(((ima*(y-yc)**2)).sum()/norm)

    return xc,yc,sx,sy

def zerolike(cube):

    newcube = SNIFS_cube()
    newcube.x = cube.x
    newcube.y = cube.y
    newcube.i = cube.i
    newcube.j = cube.j
    newcube.lbda = cube.lbda
    newcube.lend = cube.lend
    newcube.lstart = cube.lstart
    newcube.lstep = cube.lstep
    newcube.data = cube.data* 0.
    newcube.var = cube.var* 0.
    newcube.nlens = cube.nlens
    newcube.no = cube.no
    newcube.nslice = cube.nslice

    try:
        newcube.e3d_data_header = cube.e3d_data_header
        newcube.e3d_extra_hdu_list = cube.e3d_extra_hdu_list
        newcube.e3d_file = cube.e3d_file
        newcube.e3d_grp_hdu = cube.e3d_grp_hdu
    except:
        newcube.e3d_data_header = None
        newcube.e3d_extra_hdu_list = None
        newcube.e3d_file = None 
        newcube.e3d_grp_hdu = None 

    return newcube
