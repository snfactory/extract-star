######################################################################
## Filename:      pySNIFS.py
## Version:       $Revision$
## Description:   
## Author:        Emmanuel Pecontal
## Author:        $Author$
## $Id$
######################################################################

import pyfits

import scipy
from scipy import optimize,size,special
from scipy.special import *
from scipy import interpolate as I

import numarray
from numarray import linear_algebra as L
from numarray import Float32,Float64,sum

import numpy

import pylab, matplotlib

__author__ = '$Author$'
__version__ = '$Revision$'
__docformat__ = "epytext en"

########################   I/O functions   #########################

class spectrum:
    """
    1D spectrum class.
    """
    def __init__(self,data_file=None,var_file=None,no=None,num_array=True,x=None,data=None,var=None,start=None,step=None,nx=None):
        """
        Initiating the class.
        @param data_file: fits file from which the data are read. It can be a 1D fits image or a euro3d
            datacube. In the euro3d case, the variance spectrum is read from this cube if present.
            In the 1D fits image file case, if this file contains two extensions, the variance 
            spectrum is read from the second one.
        @param var_file: fits file from which the variance is read if it is not present in the data file. 
            It must be a 1D fits image.
        @param no: number of the spaxel in the datacube. Needed only if the input file is a euro3d cube.
        @param x: array containing the x coordinates in the spectrum. It is used only if the spectrum
            is not regularly sampled
        @param data: array containing the data of the spectrum
        @param var: array containing the variance of the spectrum
        @param start: coordinate in user coordinates of the first point of the data array.
        @param step: step in user coordinates of the data.
        @param nx: number of data point. It is usefull when the user want to create an array of zero
            data. otherwise, this value is the size of the data array.
        @param num_array: flag to say if the data are stored in numarray or in numeric.
        @group parameters used when the spectrum is read from a fits spectrum or a euro3d datacube:
               data_file,var_file,no  
        @group parameters used when the spectrum is not read from a file:
               x,data,var,start,step,nx
        @group parameters used in both cases: num_array
        
        """
        self.file = file
        if data_file is not None:
            data_fits = pyfits.open(data_file)
            if data_fits[0].header.has_key('EURO3D'):
                # Case where the spectrum belongs to an e3d datacube
                if no is None:
                    raise ValueError('The user must provide the spectrum number in the datacube')
                else:
                    if no in data_fits[1].data.field('SPEC_ID'):
                        i = data_fits[1].data.field('SPEC_ID').tolist().index(no)
                        self.data = data_fits[1].data.field('DATA_SPE')[i]
                        if 'STAT_SPE' in data_fits[1].columns.names:
                            self.var = data_fits[1].data.field('STAT_SPE')[i]
                        else:
                            self.var = None 
                        self.len = data_fits[1].data.field('SPEC_LEN')[i]
                        self.step = data_fits[1].header.get('CDELTS')
                        self.start = data_fits[1].header.get('CRVALS') + data_fits[1].data.field('SPEC_STA')[i]*self.step
                        self.x = arange(self.len)*self.step + self.start
                    else:
                        self.data = None
                        self.var = None
                        self.len = None
                        self.step = None
                        self.start = None
                        self.x = None
            else:
                # Case where the data and variance spectra are read from fits files
                self.data = data_fits[0].data
                if len(data_fits) == 2:
                    # The data fits spectrum has an extension containing the variance spectrum
                    self.var = data_fits[1].data
                elif var_file is not None:
                    # The variance is read from a fits file 
                    var_fits = pyfits.open(var_file)
                    self.var = var_fits[0].data
                else:
                    self.var = None
                if not isinstance(self.var,type(None)):
                    if len(self.var) != len(self.data):
                        raise ValueError('Data and variance spectra must have the same length')
                self.len = data_fits[0].header.get('naxis1')
                self.x = data_fits[0].header.get('crval1') + arange(data_fits[0].header.get('naxis1')) * data_fits[0].header.get('cdelt1')
                self.step = data_fits[0].header.get('cdelt1')
                self.start = data_fits[0].header.get('crval1')
        else:
            if x is None:
                # Case for a regularly sampled spectrum

                if start is not None and step is not None and nx is not None:
                    self.start = start
                    self.step = step
                    self.len = nx
                    self.x = start + arange(nx)*step
                    if data is None:
                        self.data = zeros(nx)
                        self.var = zeros(nx)
                        self.x = start + arange(nx)*step
                    else:
                        print 'Warning: When nx is given, the data array is not taken into account and the\
                        spectrum data is zet to zeros(nx)'
                                            
                else:
                    if data is None:
                        if nx is None:
                            raise ValueError('Not enough parameters to fill the spectrum data field')
                        else:
                            self.data = zeros(nx)
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
                        self.x = self.start + arange(nx)*self.step
                        
                    else:
                        self.data = data
                        self.len = len(data)
                        if var is not None:
                            if len(var) != len(data):
                                raise ValueError('data and variance array must have the same length')
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
                        self.x = self.start + arange(self.len)*self.step
                        
            else:
                # Case for a not regularly sampled spectrum
                self.start = None
                self.step = None
                self.len = len(x)
                if data is None:
                        self.data = zeros(len(x))
                        self.var = zeros(len(x))
                        self.x = x
                else:
                    if len(data) != len(x):
                        raise ValueError("x and data arrays must have the same size")
                    if var is not None:
                        if len(var) != len(data):
                            raise ValueError("data and var arrays must have the same size")
                    self.data = data
                    self.var = var
                    self.x = x
        if num_array:
            self.data = numarray.array(self.data)
            self.x = numarray.array(self.x)
            if not isinstance(self.var,type(None)):
                self.var = numarray.array(self.var)
        else:
            self.data = scipy.array(self.data)
            self.x = scipy.array(self.x)
            if self.var is not None:
                self.var = scipy.array(self.var)
              
        if self.var is None:
            self.has_var = False
        else:
            self.has_var = True
        
        if self.len is not None:
            self.index_list = arange(self.len).tolist()
            self.intervals = [(1,self.len)]
         
        self.curs_val = []
        self.cid = None       
        
    def plot(self,intervals=None,var=False,line='-',color='b'):
        """
        Create a new pylab figure and plot the spectrum.
        @param intervals: A list of 2 elements tuples defining the intervals in x to be plotted
        @param var: Flag to determine if we plot the variance instead of the data
        @param line: line type in pylab syntax
        @param color: line color in pylab syntax
        """
        
        line = line+color
        if var:
            data = self.var
        else:
            data = self.data
        pylab.figure()  #modif du 11-10-05 (GR)
        if intervals is not None:
            self.subset(intervals=intervals)
                                   
        ind_intervals = self.intervals
        
        for ind_interval in ind_intervals:
            if len(ind_interval) == 2:
                pylab.plot(self.x[ind_interval[0]:ind_interval[1]],data[ind_interval[0]:ind_interval[1]],line)
##      handling of different line style for each interval not yet implemented
##             else:
##                 pylab.plot(self.x[ind_interval[0]:ind_interval[1]],data[ind_interval[0]:ind_interval[1]],ind_interval[2])

    def overplot(self,intervals=None,var=False,line='-',color='b'): # modif du 11-10-05 (GR)      
        """
        Plot the spectrum in the current pylab figure.
        @param intervals: A list of 2 elements tuples defining the intervals in x to be plotted
        @param var: Flag to determine if we plot the variance instead of the data
        @param line: line type in pylab syntax
        @param color: line color in pylab syntax
        """
        line = line+color
        if var:
            data = self.var
        else:
            data = self.data
        if intervals is not None:   
            self.subset(intervals=intervals)
        
        ind_intervals = self.intervals
        
        for ind_interval in ind_intervals:
            if len(ind_interval) == 2:
                pylab.plot(self.x[ind_interval[0]:ind_interval[1]],data[ind_interval[0]:ind_interval[1]],line)
##      handling of different line style for each interval not yet implemented
##             else:
##                 pylab.plot(self.x[ind_interval[0]:ind_interval[1]],data[ind_interval[0]:ind_interval[1]],ind_interval[2])
                
  
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
        else:
            for interval in intervals:
                if not isinstance(interval,tuple):
                    raise ValueError("Intervals must be provided as a list of 2 elements tuples")
                else:
                    if len(interval) == 2:
                        if not isscalar(interval[0]) or interval[0] < min(self.x) or interval[1] > max(self.x):
                            raise ValueError("Interval bounds must be numbers between xmin and xmax.")
                        else:
                            ind_min = self.index(min(interval))
                            ind_max = self.index(max(interval))
                            self.index_list = self.index_list + (arange(ind_max-ind_min+1)+ind_min).tolist()
                            self.intervals.append((ind_min,ind_max))
                    else:
                        raise ValueError("Interval must be provided as a list of 2 elements tuples")

    def reset_interval(self):
        """
        Erase the intervals selections
        """
        self.index_list = arange(self.len).tolist()
        self.intervals = [(1,self.len)]

 
     
    def cursor_on(self,print_flag=True):
        """
        Turn on cursor binding. When set, a click on button 2 print the x and y cursor values (if the print_flag is set), and append them
        in the field curs_val of the spectrum. This field is a list of 2 elements lists.
        It is turned off using cursor_off
        """
        def click(event):
            if event.button == 2:
                self.curs_val.append([event.xdata,event.ydata])
                if print_flag:
                    print 'x = %e  y = %e'%(event.xdata,event.ydata)
        if self.cid is not None:
            self.cursor_off()
            
        self.cid = pylab.connect('button_press_event',click)
    
    def cursor_off(self):
        """
        Turn off cursor binding and reset the field curs_val to an empty list.
        """
        if self.cid is not None:
            pylab.disconnect(self.cid)
            self.curs_val = []
            self.cid = None
            
    def index(self,x):
        """
        Return the index of the closest point to x in the spectrum
        """
        if min(self.x) > x or x > max(self.x):
            raise ValueError('x out of the spectrum x range')
        else:
            return argmin((self.x - x)**2,axis=-1)

                        
    def WR_fits_file(self,fits_file,header_list=None):
        """
        Write the spectrum in a fits file.
        @param fits_file: Name of the output fits file
        @param header_list: List of 2 elements lists containing the name and value of the header to be
            saved. If set to None, only the mandatory fits header will be stored.
        """
        if self.start is None or self.step is None or self.data is None:
            raise ValueError('Only regularly sampled spectra can be saved as fits files.')
        
        hdu = pyfits.PrimaryHDU()
        hdu.data = numarray.array(self.data)
        hdu.header.update('NAXIS', 1)
        hdu.header.update('NAXIS1', self.len,after='NAXIS')
        hdu.header.update('CRVAL1', self.start)
        hdu.header.update('CDELT1', self.step)
        if header_list is not None:
            for desc in header_list:
                if desc[0][0:5] != 'TUNIT' and desc[0][0:5] != 'TTYPE' and desc[0][0:5] != 'TFORM' and \
                       desc[0][0:5] != 'TDISP' and desc[0] != 'EXTNAME' and desc[0] != 'XTENSION' and \
                       desc[0] != 'GCOUNT' and desc[0] != 'PCOUNT' and desc[0][0:5] != 'NAXIS' and \
                       desc[0] != 'BITPIX' and desc[0] != 'CTYPES' and desc[0] != 'CRVALS' and \
                       desc[0] != 'CDELTS' and desc[0] != 'CRPIXS' and desc[0]!='TFIELDS':
                    hdu.header.update(desc[0],desc[1])
        if self.has_var:
            hdu_var = pyfits.ImageHDU()
            hdu_var.data = numarray.array(self.var)
            hdu_var.header.update('NAXIS', 1)
            hdu_var.header.update('NAXIS1', self.len,after='NAXIS')
            hdu_var.header.update('CRVAL1', self.start)
            hdu_var.header.update('CDELT1', self.step)
            
        hdulist = pyfits.HDUList()
        hdulist.append(hdu)
        if self.has_var:
            hdulist.append(hdu_var)
        hdulist.writeto(fits_file, clobber=True) # Overwrite

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
            data_fits = pyfits.open(data_file)
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
                self.data = data_fits[0].data            
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
            elif len(shape(data)) != 2:
                raise ValueError('The user must provide a two dimensions data array.')

            self.data = data
            self.var = var
            self.startx = startx
            self.starty = starty
            if endx is None:
                self.stepx = stepx
                self.endx = startx + (len(data[:,0])-1)*stepx
            else:
                self.endx = endx
                self.stepx = float(endx-startx)/(len(data[:,0])-1)
            if endy is None:
                self.stepy = stepy
                self.endy = starty + (len(data[0])-1)*stepy
            else:
                self.endy = endy
                self.stepy = float(endy-starty)/(len(data[0])-1)            
            self.nx = shape(data)[0]
            self.ny = shape(data)[1]
            self.header = header
            
        self.labx = labx
        self.laby = laby
        self.vmin = numpy.float(self.data.min())
        self.vmax = numpy.float(self.data.max())

        
    def display(self,cmap=pylab.cm.jet,aspect='equal',vmin=None,vmax=None,subima=None,ima=True,contour=False):
        """
        Display the image in a pylab figure.
        @param cmap: Colormap in pylab syntax
        @param aspect: aspect ratio in pylab syntax (auto, equal or a number)
        @param vmin: low cut in the image for the display (if None, it is set to the minimum of the image)
        @param vmax: high cut in the image for the display (if None, it is set to the maximum of the image)
        @param subima: list containing the coordinates of the subarea to be displayed if the coordinates
            are given as a list of tuples: [(i1,i2),(j1,j2)], the coordinates are given as indexes in the
            array. If the coordinates are given as a list of lists: [[x1,x2],[y1,y2]], the coordinates are
            given as x and y world coordinates.
        @warning: As in python i holds for the lines and j for the columns, the tuple (i1,i2) correspond
            to y coordinates on the screen and (j1,j2) to the x coordinates.
        """
        if vmin is not None: self.vmin = vmin
        if vmax is not None: self.vmax = vmax
        if subima is not None:
            if isinstance(subima,list):
                if isinstance(subima[0],list) and isinstance(subima[1],list):
                    ii = [int((subima[1][0]-self.starty)/self.stepy),\
                            int((subima[1][1]-self.starty)/self.stepy)]
                    jj = [int((subima[0][0]-self.startx)/self.stepx),\
                            int((subima[0][1]-self.startx)/self.stepx)]
                    extent = [subima[0][0],subima[0][1],subima[1][0],subima[1][1]]
                elif isinstance(subima[0],tuple) and isinstance(subima[1],tuple):
                    ii = subima[0]
                    jj = subima[1]
                    extent = [subima[1][0]*self.stepx+self.startx,subima[1][1]*self.stepx+self.startx,\
                              subima[0][0]*self.stepy+self.starty,subima[0][1]*self.stepy+self.starty]
                else:
                    raise TypeError("Subima must a list of 2 tuples or two lists")
        else:
            extent = [self.starty-self.stepy/2.,self.endy+self.stepy/2.,self.startx-self.stepy/2.,self.endx+self.stepy/2.]
            ii = [0,shape(self.data)[0]]
            jj = [0,shape(self.data)[1]]

        if ima:
            pylab.imshow(self.data[ii[0]:ii[1],jj[0]:jj[1]],interpolation='nearest',aspect=aspect,cmap=cmap,\
                         vmin=self.vmin,vmax=self.vmax,extent=extent,origin='lower')
        if contour:
            levels = self.vmin + arange(10)*(self.vmax-self.vmin)/10.
            pylab.contour(self.data[ii[0]:ii[1],jj[0]:jj[1]],levels,extent=extent,cmap=pylab.cm.gray)
        if self.labx is not None:
            pylab.xlabel(self.labx)
        if self.laby is not None:
            pylab.ylabel(self.laby)

    def WR_fits_file(self,file_name,mode='w+'):
        """
        Write the image in a fits file.
        @param file_name: name of the fits file
        @param mode: writing mode. if w+, the file is overwritten. Otherwise the writing will fail. 
        """
        hdulist = pyfits.HDUList()
        hdu = pyfits.PrimaryHDU()
        hdu.header.update('NAXIS', 2)
        hdu.header.update('NAXIS1',self.nx,after='NAXIS')
        hdu.header.update('NAXIS2',self.ny,after='NAXIS1')
        hdu.header.update('CRPIX1', 1)
        hdu.header.update('CRVAL1', self.startx)
        hdu.header.update('CDELT1', self.stepx)
        hdu.header.update('CRPIX2', 1)
        hdu.header.update('CRVAL2', self.starty)
        hdu.header.update('CDELT2', self.stepy)
        if self.header is not None:
            for desc in self.header:
                if desc[0]!='SIMPLE' and desc[0]!='BITPIX' and desc[0]!='EXTEND' and desc[0]!='NAXIS' and desc[0]!='NAXIS1'\
                       and desc[0]!='NAXIS2' and desc[0]!='CRPIX1' and desc[0]!='CRVAL1' and desc[0]!='CDELT1' and desc[0]!='CRPIX2'\
                       and desc[0]!='CRVAL2' and desc[0]!='CDELT2':
                    hdu.header.update(desc[0],desc[1])
        hdu.data = numarray.array(self.data)
        hdulist.append(hdu)
        hdulist.writeto(file_name, clobber=(mode=='w+'))


########################## Cube ########################################

class SNIFS_cube:
    """
    SNIFS datacube class.
    """
    def __init__(self,e3d_file=None,slices=None,lbda=None,num_array=True,threshold=1e20):
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
        @param num_array: Flag to chose between numarray or numeric array formats
        @param threshold: In the variance image, pixels where variance is not available for some reason are
            set to an arbitrarily high value. As this values seems to change from one version to another of
            the processing pipeline, we allow to pass it as the threshold parameter.
        
        """
        if e3d_file is not None:
            if slices is not None:
                if not isinstance(slices,list):
                    raise ValueError('The wavelength range must be given as a list of two or three integer positive values')
                if len(slices) != 2 and len(slices) != 3:
                    raise ValueError('The wavelength range must be given as a list of two or three integer positive values')
                if 1 in numarray.array([not isinstance(sl,int) for sl in slices]):
                    raise ValueError('The wavelength range must be given as a list of two or three integer positive values')
                if 1 in numarray.array([sl<0 for sl in slices]):
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
            
            e3d_cube = pyfits.open(e3d_file)
            gen_header = dict(e3d_cube[0].header.items())
            if not gen_header.has_key('EURO3D'):
                raise ValueError("This is not a valid e3d file (no 'EURO3D' keyword")
            elif gen_header['EURO3D'] != 'T' and gen_header['EURO3D'] != pyfits.TRUE:
                raise ValueError("This is not a valid e3d file (bad value of the keyword 'EURO3D')")
            
            self.from_e3d_file = True
            self.e3d_file = e3d_file
            # The header of the data extension is stored in a field of the class
            self.e3d_data_header = e3d_cube[1].header.items()
            # The group definition HDU and the optional extensions HDU are stored in fields of the class
            self.e3d_grp_hdu = e3d_cube[2]
            self.e3d_extra_hdu_list = [e3d_cube[i] for i in arange(len(e3d_cube)-3)+3] 
            ref_start = e3d_cube[1].header['CRVALS']
            step      = e3d_cube[1].header['CDELTS']
            self.lstep = step
            if 'STAT_SPE' in e3d_cube[1].columns.names:
                var  = e3d_cube[1].data.field('STAT_SPE')
            else:
                var = None
            data  = e3d_cube[1].data.field('DATA_SPE')
            spec_sta  = e3d_cube[1].data.field('SPEC_STA')
            spec_len  = e3d_cube[1].data.field('SPEC_LEN')
            spec_end  = spec_len + spec_sta
            npts      = e3d_cube[1].data.getshape()[0]
            common_lstart,common_lend,lstep = max(spec_sta),min(spec_end),1
            common_start = ref_start + common_lstart * step

            tdata = numarray.transpose(numarray.array([data[i][common_lstart-spec_sta[i]:common_lend-spec_sta[i]] \
                                                       for i in range(len(data))]))
            if not isinstance(var,type(None)):
                tvar = numarray.transpose(numarray.array([var[i][common_lstart-spec_sta[i]:common_lend-spec_sta[i]] \
                                                          for i in range(len(var))]))
                                                          
                # In the variance image, pixels where variance is not available for some reason are set to
                # an arbitrarily high value. As this values seems to change from one version to another of the
                # processing pipeline, we allow to pass it as a parameter: threshold
                tvar = numarray.where((tvar>threshold),threshold,tvar)
                tvar = numarray.where((tvar<-threshold),threshold,tvar)

            lbda = arange(common_start,common_start+len(tdata)*step,step)

            

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
                raise ValueError('The step in slices is not compatible with the slices interval requested')
                         
            if not s:
                self.data = tdata[lmin - common_lstart:lmax - common_lstart:lstep]
                if not isinstance(var,type(None)):
                    self.var = tvar[lmin - common_lstart:lmax - common_lstart:lstep]
                self.lbda = lbda[lmin - common_lstart:lmax - common_lstart:lstep]
            else:
                self.data = numarray.convolve.boxcar(tdata,(lstep,1))\
                            [lmin - common_lstart + lstep/2:lmax - common_lstart+lstep/2:lstep]
                if not isinstance(var,type(None)):
                    self.var = numarray.convolve.boxcar(tvar,(lstep,1))\
                            [lmin - common_lstart + lstep/2:lmax - common_lstart+lstep/2:lstep]
                self.lbda = lbda[lmin - common_lstart+lstep/2:lmax - common_lstart+lstep/2:lstep]
                self.lstep = self.lstep * lstep
            self.lstart = self.lbda[0]

            self.x = e3d_cube[1].data.field('XPOS')
            self.y = e3d_cube[1].data.field('YPOS')
            self.no = e3d_cube[1].data.field('spec_id')
            # We read in the extension table the I and J index of the lenslets. As the extension table may have
            # a different number of lenslets than the data table, we first search the index of the common
            # lenslets.
            ind = [e3d_cube[3].data.field('NO').tolist().index(i) for i in e3d_cube[1].data.field('spec_id')]
            self.i = e3d_cube[3].data.field('I')[ind]+7
            self.j = e3d_cube[3].data.field('J')[ind]+7
            if not num_array:
                self.data = scipy.array(self.data)
                if not isinstance(var,type(None)):
                    self.var = scipy.array(self.var)
                self.lbda = scipy.array(self.lbda)
                self.x = scipy.array(self.x)
                self.y = scipy.array(self.y)
                self.i = scipy.array(self.i)
                self.j = scipy.array(self.j)
                self.no = scipy.array(self.no)
                
            self.nslice = len(self.lbda)
            self.nlens = len(self.x)
        else:
            # If no euro3d file is given, we create an SNIFS cube from  
            self.from_e3d_file = False
            self.nlens = 225
            if lbda is None:
                if num_array:
                    self.data = numarray.zeros(self.nlens,Float32)
                else:
                    self.data = scipy.zeros(self.nlens,Float32)
                self.lbda = None  
                self.nslice = None
            else:
                self.nslice = len(lbda)
                data = numarray.zeros((self.nslice,self.nlens),Float64) 
                x = numarray.array(((7-arange(15))*0.42).tolist()*15)
                y = numarray.repeat((arange(15)-7)*0.42,15)
                i = numarray.array((14-arange(15)).tolist()*15)
                j = numarray.repeat((arange(15)),15)
                no = numarray.ravel(numarray.transpose(numarray.reshape(numarray.arange(self.nlens)+1,(15,15)))) 
                lbda = numarray.array(lbda)
                if num_array:
                    self.data = data
                    self.x = x
                    self.y = y
                    self.i = i
                    self.j = j
                    self.no = no
                    self.lbda = lbda  
                else:
                    self.data = scipy.array(data)
                    self.x = scipy.array(x)
                    self.y = scipy.array(y)
                    self.i = scipy.array(i)
                    self.j = scipy.array(j)
                    self.no = scipy.array(no)
                    self.lbda = scipy.array(lbda)  
          
    def slice2d(self,n,coord='w',var=False,nx=15,ny=15):
        """
        Extract a 2D slice from a cube and return it as an array
        @param n: If n is a list of 2 values [n1,n2], the function returns the sum of the slices between n1
            and n2 if it is an integer n , it returns the slice n
        @param coord: The type of coordinates:
                      - 'w' -> wavelength coordinates
                      - 'p' -> pixel coordinates
        @param nx: dimension of the slice in x
        @param ny: dimension of the slice in y
        """
        if isinstance(n,list):
            if coord == 'p':
                n1 = n[0]
                n2 = n[1]
            elif coord == 'w':
                n1 = argmin((self.lbda-n[0])**2,axis=-1)
                n2 = argmin((self.lbda-n[1])**2,axis=-1)
            else:
                raise ValueError("Coordinates flag should be either 'p' or 'w'")
            if n1 == n2:n2=n2+1
        else:
            if coord == 'p':
                n1 = n
                n2 = n+1
            elif coord == 'w':
                n1 = argmin((self.lbda-n)**2,axis=-1)
                n2 = n1+1
            else:
                raise ValueError("Coordinates flag should be either 'p' or 'w'")

        if n1 >= 0 and n2 <= numarray.shape(self.data)[0]:
            slice_2D = numarray.zeros((nx,ny),Float32) * nan
            i = numarray.array(self.i)
            j = numarray.array(self.j)
            if var:
                slice_2D[i,j] = sum(self.var[n1:n2])
            else:
                slice_2D[i,j] = sum(self.data[n1:n2])
            return(slice_2D)
        else:
            raise IndexError("no slice #%d" % n)

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
            if shape(mask) != shape(data):
                raise ValueError('mask array must have the same shape than the data array')
            data = data * mask
            return sum(data,1)
        
        if (no is not None and ind is not None) or (no is None and ind is None):
            raise TypeError("lens number (no) OR spec index (ind) should be given.")
        else:
            if (no is not None):
                if no in self.no.tolist():
                    return data[:,argmax(self.no == no,axis=-1)]
                else:
                    raise IndexError("no lens #%d" % no)
            else:
                if not isinstance(ind,list):
                    if 0 <= ind < numarray.shape(data)[1]:
                        return data[:,ind]
                    else:
                        raise IndexError("no index #%d" % ind)
                else:
                    if 0 <= ind[0] and ind[1] < numarray.shape(data)[1]:
                        ind[1] = ind[1]+1
                        return sum(data[:,ind[0]:ind[1]],1)
                    else:
                        raise IndexError("Index list out of range")
                        
    def plot_spec(self,no=None,ind=None,mask=None,var=False,ax=None):
        """
        Plot a spectrum extracted from the datacube.
        @param no: lenslet number in the datacube
        @param ind: index in the data array
        @param mask: optional array having the same shape than the data field of the cube. If given, the
            spectrum returned is the sum of the data array multiplied by the mask.
        @param var: Variance flag. If set to True, the variance spectrum is ploted.
        @param ax: pylab axes on which the spectrum will be ploted. If set to None, a new axes is created.
        """
        if ax is None:
            pylab.plot(self.lbda,self.spec(no=no,ind=ind,mask=mask,var=var))
        else:
            ax.plot(self.lbda,self.spec(no=no,ind=ind,mask=mask,var=var))
        pylab.show()

    def get_spec(self,no,num_array=True):
        """
        Extract the spectrum corresponding to lenslet no and return it as a pySNIFS.spectrum object
        @param no: lenslet number of the spectrum to be extracted
        @param num_array: Flag to chose between numarray or numeric array format
        """
        spec = spectrum(x=self.lbda,data=self.spec(no),num_array=num_array)
        if hasattr(self,'lstep'):
            spec.step = self.lstep
        if hasattr(self,'lstart'):
            spec.start = self.lstart

        return spec
    
    def disp_slice(self,n,coord='w',aspect='equal',vmin=None,vmax=None,cmap=pylab.cm.jet,var=False,contour=False,ima=True,nx=15,ny=15):
        """
        Display a 2D slice.
        @param n: If n is a list of 2 values [n1,n2], the function returns the sum of the slices between n1
            and n2 if it is an integer n , it returns the slice n
        @param coord: The type of coordinates:
                      - 'w' -> wavelength coordinates
                      - 'p' -> pixel coordinates
        @param cmap: Colormap in pylab syntax
        @param aspect: aspect ratio in pylab syntax (auto, equal or a number)
        @param vmin: low cut in the image for the display (if None, it is 'smartly' computed)
        @param vmax: high cut in the image for the display (if None, it is 'smartly' computed)
        @param var: Variance flag. If set to True, the variance slice is displayed. 
        @param nx: dimension of the slice in x
        @param ny: dimension of the slice in y
        """
        slice = self.slice2d(n,coord,var=var,nx=nx,ny=ny)
        med = scipy.median(ravel(slice))
        disp = sqrt(scipy.median((ravel(slice)-med)**2))
        if vmin is None:
            vmin = float(med - 3*disp)
        if vmax is None or vmax < vmin:
            vmax = float(med + 10*disp)
        fig = pylab.gcf()
        #fig.clf()
        extent = [-1./2.,ny-1/2.,-1/2.,nx-1/2.]
        if ima:
            pylab.imshow(slice,interpolation='nearest',aspect='equal',vmin=vmin,vmax=vmax,cmap=cmap,\
                         origin='lower',extent=extent)
        if contour:
            levels = vmin + arange(10)*(vmax-vmin)/10.
            pylab.contour(slice,levels,cmap=pylab.cm.gray)
            
    def disp_data(self,vmin=None,vmax=None,cmap=pylab.cm.hot,var=False):
        """
        Display the datacube as the stack of all its spectra
        @param vmin: low cut in the image for the display (if None, it is 'smartly' computed)
        @param vmax: high cut in the image for the display (if None, it is 'smartly' computed)
        @param var: Variance flag. If set to True, the variance slice is displayed. 
        @param cmap: Colormap in pylab syntax  
        """
        if var:
            data = self.var
        else:
            data = self.data
        med = float(scipy.median(ravel(data)))
        disp = float(sqrt(scipy.median((ravel(data)-med)**2)))
        if vmin is None:
            vmin = med - 3*disp
        if vmax is None:
            vmax = med + 10*disp
        #pylab.imshow(numarray.transpose(self.data),interpolation='nearest',aspect='auto')
        pylab.imshow(numarray.transpose(self.data),interpolation='nearest',aspect='auto',vmin=vmin,vmax=vmax)     
            
    def get_no(self,i,j):
        """
        Get the lenslet number from its coordinates i,j 
        """
        if i>max(self.i) or i<min(self.i) or j>max(self.j) or j<min(self.j):
            raise ValueError("Index out of range.")
        no = self.no[argmax((self.i == i)*(self.j == j),axis=-1)]
        return(no)

    def get_ij(self,no):
        """
        Get the lenslet coordinates i,j from its number no
        """
        if no>max(self.no) or no<min(self.no):
            raise ValueError("Lens number out of range.")
        i = self.i[argmax(self.no == no,axis=-1)]
        j = self.j[argmax(self.no == no,axis=-1)]
        return((i,j))

    def get_lindex(self,val):
        """
        Return the index of the spec (i,j) or no in the stacked array data
        @param val: tuple (i,j) or integer no defining the lenslet
        """
        if isinstance(val,tuple):
            if val[0]>max(self.i) or val[0]<min(self.i) or val[1]>max(self.j) or val[1]<min(self.j):
                raise ValueError("Index out of range.")
            ind = argmax((self.i == val[0])*(self.j == val[1]),axis=-1)
        else:
            ind = argmax(self.no == val,axis=-1)

        return(ind)

    def WR_e3d_file(self,fits_file):
        """
        Write the datacube as a euro3d fits file.
        @param fits_file: Name of the output file
        """
        if not self.from_e3d_file:
            raise Error("Writing e3d file from scratch not yet implemented")
        data_list = (numarray.transpose(self.data)).tolist()
        start_list = [self.lstart for i in arange(self.nlens)]
        no_list = self.no.tolist()
        xpos_list = self.x.tolist()
        ypos_list = self.y.tolist()
        WR_e3d_file(data_list,None,no_list,start_list,self.lstep,xpos_list,ypos_list,\
                    fits_file,self.e3d_data_header,self.e3d_grp_hdu,self.e3d_extra_hdu_list)
        

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
        @param path_Snifs: Path of the Snifs software
        @param path_mask: Path where to find the mask fits file
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

        mask_tbl = pyfits.open('tmp_mask.fits')
        tmp_tbl = pyfits.open('tmp_tbl.fits')
        self.no = mask_tbl[1].data.field('no').tolist()
        self.lbda_list = []
        i = 1
        while tmp_tbl[1].header.has_key('LBDA%d'%i):
            self.lbda_list.append(tmp_tbl[1].header.get('LBDA%d'%i))
            i = i+1

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
    
    def plot(self,no_list=None,interpolate=False,lbda=None,symbol='k-'):
        """
        Plot the spectra positions in CCD pixels
        @param no_list: lens number list. If only one number is given, the spectrum position corresponding
            to this lens number is ploted interpolate: Used if lbda==None: if interpolate==False, the point
            will be ploted for wavelenghtes in lbda_list. Otherwise, 100 points per spectra will be ploted
            interpolated from lbda_list.
        @param lbda: wavelength at which we plot the spectra.           
        """
        if no_list is None:
            no_list = self.no
        elif isinstance(no_list,int):
            no_list = [no_list]
        for no in no_list:
            if not no in self.no:
                raise ValueError('lens #%d not present in mask'%no)
        for no in no_list:
            if lbda is None:
                x,y = self.get_spec_lens(no)
                if interpolate:
                    y = y[0] + arange(100)*(y[len(y)-1]-y[0]-1)/100. 
                    x = self.interpolate(no,y)
            else:
                x,y = self.get_coord_lens(no,lbda)
                xx = [x,x]
                yy = [y,y]
                x = xx
                y = yy
            pylab.plot(x,y,symbol)

     
    def interpolate(self,no,yy):
        """
        Interpolate the spectra x position for spectrum #no at position yy
        @param no: lens number
        @param yy: y position where to compute the x value
        """
        if not no in self.no:
            raise ValueError('lens #%d not present in mask'%no)
        #y = numarray.array(sort(self.pts[no][1]))
        #x = numarray.array(self.pts[no][0])[numarray.argsort(self.pts[no][1])]
        x,y = self.get_spec_lens(no)
        x = numarray.array(x)[numarray.argsort(y)]
        y = numarray.array(sort(y))
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
    tck = scipy.interpolate.splrep(table[1].data.field(colx),\
                                   table[1].data.field(coly),s=0)
    tab = scipy.interpolate.splev(ref_pos,tck)
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
    print Min,Max
    if bin is None:
        bin = (Max-Min)/nbin
        
    bin_array = arange(nbin)*bin + Min
    n = searchsorted(sort(data), bin_array)
    n = concatenate([n, [len(data)]])
    data = n[1:]-n[:-1]
    x = bin_array
    hist = spectrum(data=data,x=x)
    
    hist.len = len(bin_array)
    if cumul:
        hist.data = numarray.array([float(sum(hist.data[0:i+1])) for i in arange(hist.len)])/float(sum(hist.data))
    return hist

def common_bounds_cube(cube_list):
    """
    Computes the common bounds of a list of pySNIFS cubes
    @param cube_list: Input list of spectra
    @return : imin,imax: lists of the indexes of the lower/upper common bound for each cube
    """
    if False in [hasattr(cube,'lstep') for cube in cube_list]:
        raise ValueError("Missing attribute lstep in datacubes.")
    else:
        if L.mlab.std([cube.lstep for cube in cube_list]) != 0:
            raise ValueError("All cubes should have the same step.")
        xinf = max([min(cube.lbda) for cube in cube_list])
        xsup = min([max(cube.lbda) for cube in cube_list])
        imin = [argmin(abs(cube.lbda-xinf),axis=-1) for cube in cube_list]
        imax = [argmin(abs(cube.lbda-xsup),axis=-1) for cube in cube_list]

    return imin,imax

def common_bounds_spec(spec_list):
    """
    Computes the common bounds of a list of pySNIFS spectra
    @param spec_list: Input list of spectra
    @return: imin,imax: lists of the indexes of the lower/upper common bound for each spectrum
    """
    if False in [hasattr(spec,'step') for spec in spec_list]:
        raise ValueError("Missing attribute lstep in spectra.")
    else:
        if L.mlab.std([spec.step for spec in spec_list]) != 0:
            raise ValueError("All spectra should have the same step.")
        xinf = max([min(spec.x) for spec in spec_list])
        xsup = min([max(spec.x) for spec in spec_list])
        imin = [argmin(abs(spec.x-xinf),axis=-1) for spec in spec_list]
        imax = [argmin(abs(spec.x-xsup),axis=-1) for spec in spec_list]

    return imin,imax

def common_lens(cube_list):
    """
    Give the common lenses of a list of pySNIFS cubes
    @param cube_list: input list of datacubes
    @return: inters: list of the lenses common to all the cubes of the list
    """
    inters = cube_list[0].no
    for i in arange(len(cube_list)-1)+1:
        inters = filter(lambda x:x in cube_list[i].no,inters) 
    return inters

def fit_poly(y,n,deg,x=None):
    """
    Fit a polynomial whith median sigma clipping on an array y
    @param y: Input array giving the ordinates of the points to be fitted
    @param n: rejection threshold
    @param deg: Degree of the polunomial
    @param x: Optional input array giving the abscissae of the points to be fitted. If not given
       the abscissae are taken as an array [1:len(y)]
    """
    if x is None:
        x = arange(len(y))
    old_l = 0
    l = len(y)
    while l != old_l:
        old_l = len(y)
        p = scipy.poly1d(matplotlib.mlab.polyfit(x,y,deg))
        sigma = sqrt(L.mlab.median((p(x) - y)**2))
        y1 = compress(y<p(x)+n*sigma,y)
        x1 = compress(y<p(x)+n*sigma,x)
        y = compress(y1>p(x1)-n*sigma,y1)
        x = compress(y1>p(x1)-n*sigma,x1)
        l = len(y)      
    return p

def WR_e3d_file(data_list,var_list,no_list,start_list,step,xpos_list,ypos_list,fits_file,data_header,grp_hdu,extra_hdu_list):
    """
    Write a data cube as a euro3d file on disk.
       @param data_list: List of the spectra of the datacube
       @param var_list: List of the variance spectra of the data_cube
       @param no_list: List of the spaxel ident number of the datacube
       @param start_list: List of the start wavelength of each spectrum of the datacube
       @param step: Wavelength step of the datacube
       @param xpos_list: List of the x position of each spaxel of the datacube
       @param ypos_list: List of the x position of each spaxel of the datacube
       @param fits_file: Output fits file to be written on disk
       @param data_header: data header of the new e3d file. All the standards e3d
           headers containing information on the data themselves will be overwritten. Only the non standard
           will be copied from this parameter.
       @param grp_hdu: pyfits HDU describing the spaxel groups.
       @param extra_hdu_list: pyfits HDU containing non e3d-mandatory data.
    """
    start = max(start_list)
    spec_sta = [int((s - start)/step+0.5*scipy.sign(s-start)) for s in start_list]
    spec_len = [len(s) for s in data_list]
    selected = [0 for i in arange(len(data_list))]
    group_n = [1 for i in arange(len(data_list))]
    nspax = [1 for i in arange(len(data_list))]
    spax_id = [' ' for i in arange(len(data_list))]
    #for i,no in enumerate(no_list):
    col_list = []
    col_list.append(pyfits.Column(name='SPEC_ID',format='J',array=no_list))
    col_list.append(pyfits.Column(name='SELECTED',format='J',array=selected))
    col_list.append(pyfits.Column(name='NSPAX',format='J',array=nspax))
    col_list.append(pyfits.Column(name='SPEC_LEN',format='J',array=spec_len))
    col_list.append(pyfits.Column(name='SPEC_STA',format='J',array=spec_sta))
    col_list.append(pyfits.Column(name='XPOS',format='E',array=xpos_list))
    col_list.append(pyfits.Column(name='YPOS',format='E',array=ypos_list))
    col_list.append(pyfits.Column(name='GROUP_N',format='J',array=group_n))
    col_list.append(pyfits.Column(name='SPAX_ID',format='1A1',array=spax_id))
    col_list.append(pyfits.Column(name='DATA_SPE',format='PE()',array=data_list))
    col_list.append(pyfits.Column(name='QUAL_SPE',format='PJ()',array=[[0 for i in d] for d in data_list]))
        
    tb_hdu = pyfits.new_table(col_list)
    tb_hdu.header.update('CTYPES',' ')
    tb_hdu.header.update('CRVALS',start)
    tb_hdu.header.update('CDELTS',step)
    tb_hdu.header.update('CRPIXS',1)
    tb_hdu.header.update('EXTNAME','E3D_DATA')


    for desc in data_header:
        if desc[0]!='XTENSION' and desc[0]!='BITPIX' and desc[0][0:5]!='NAXIS' and desc[0]!='GCOUNT' and \
               desc[0]!='PCOUNT' and desc[0]!='TFIELDS' and desc[0]!='EXTNAME' and desc[0][0:5]!='TTYPE' and \
               desc[0]!='TFORM' and desc[0]!='TUNIT' and desc[0]!='TDISP' and desc[0]!='CTYPES' and \
               desc[0]!='CRVALS' and desc[0]!='CDELTS' and desc[0]!='CRPIXS':
            tb_hdu.header.update(desc[0],desc[1])

    pri_hdu = pyfits.PrimaryHDU()
    pri_hdu.header.update('EURO3D',pyfits.TRUE)
    pri_hdu.header.update('E3D_ADC',pyfits.FALSE)
    pri_hdu.header.update('E3D_VERS',1.0)
    hdu_list = pyfits.HDUList([pri_hdu,tb_hdu,grp_hdu]+extra_hdu_list)
    hdu_list.writeto(fits_file, clobber=True) # Overwrite
        
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
        x = arange(ima_shape)-xc
        gaus = I*exp(-0.5*(x/s)**2)
    else:
        if pa is None:
            raise ValueError('Position angle must be supplied by the user')
        pa = pa*pi/180.
        nx = ima_shape[0]
        ny = ima_shape[1]
        xc = center[0]
        yc = center[1]
        sx = sigma[0]
        sy = sigma[1]
        x = reshape(repeat(arange(nx),ny),(nx,ny))
        y = transpose(reshape(repeat(arange(ny),nx),(ny,nx)))
        xr = (x-xc)*cos(pa) - (y-yc)*sin(pa)
        yr = (x-xc)*sin(pa) + (y-yc)*cos(pa)
        val = (xr/sx)**2 + (yr/sy)**2
        gaus = I*exp(-val/2)

    return gaus
