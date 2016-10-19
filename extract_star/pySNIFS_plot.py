######################################################################
## Filename:      pySNIFS_plot.py
## Version:       $Revision$
## Description:   
## Author:        Emmanuel Pecontal
## Author:        $Author$
## $Id$
######################################################################

from pySNIFS import *
import pylab

__author__ = '$Author$'
__version__ = '$Id$'

# spectrum methods ==============================

def spectrum_plot(self,intervals=None,var=False,line='-',color='b'):
    """
    Create a new pylab figure and plot the spectrum.

    @param intervals: A list of 2 elements tuples defining the intervals
        in x to be plotted
    @param var: Flag to determine if we plot the variance instead of the
        data
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
            pylab.plot(self.x[ind_interval[0]:ind_interval[1]],
                       data[ind_interval[0]:ind_interval[1]],line)
        else:
            # handling of different line style for each interval not yet
            # implemented
            ## pylab.plot(self.x[ind_interval[0]:ind_interval[1]],
            ##            data[ind_interval[0]:ind_interval[1]],
            ##            ind_interval[2])
            raise NotImplementedError()

def spectrum_overplot(self,intervals=None,var=False,line='-',color='b',ax=None):
    # modif du 11-10-05 (GR)      
    """
    Plot the spectrum in the current pylab figure.
    @param intervals: A list of 2 elements tuples defining the intervals in x to be plotted
    @param var: Flag to determine if we plot the variance instead of the data
    @param line: line type in pylab syntax
    @param color: line color in pylab syntax
    @param ax: pylab axes on which the spectrum will be ploted. If set to None, a new axes is created.
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
            if ax is None:
                pylab.plot(self.x[ind_interval[0]:ind_interval[1]],
                           data[ind_interval[0]:ind_interval[1]],line)
            else:
                ax.plot(self.x[ind_interval[0]:ind_interval[1]],
                        data[ind_interval[0]:ind_interval[1]],line)

def spectrum_cursor_on(self,print_flag=True):
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

def spectrum_cursor_off(self):
    """
    Turn off cursor binding and reset the field curs_val to an empty list.
    """
    if self.cid is not None:
        pylab.disconnect(self.cid)
        self.curs_val = []
        self.cid = None

spectrum.plot = spectrum_plot
spectrum.overplot = spectrum_overplot
spectrum.cursor_on = spectrum_cursor_on
spectrum.cursor_off = spectrum_cursor_off

# image_array methods ==============================

def imarray_display(self,cmap=pylab.cm.jet,aspect='equal',vmin=None,vmax=None,
                    subima=None,ima=True,alpha=1,contour=False,var=False,
                    linewidth=None,line_cmap=pylab.cm.gray,ncontour=10):
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
    if var:
        if self.var == None:
            raise ValueError("No variance array in this image.")
        data = self.var
    else:
        data = self.data
    if vmin is not None: self.vmin = vmin
    if vmax is not None: self.vmax = vmax
    if subima is not None:
        if isinstance(subima,list):
            if isinstance(subima[0],list) and isinstance(subima[1],list):
                ii = [int((subima[1][0]-self.startx)/self.stepx),\
                        int((subima[1][1]-self.startx)/self.stepx)]
                jj = [int((subima[0][0]-self.starty)/self.stepy),\
                        int((subima[0][1]-self.starty)/self.stepy)]
                extent = [subima[0][0],subima[0][1],
                          subima[1][0],subima[1][1]]
            elif isinstance(subima[0],tuple) and \
                     isinstance(subima[1],tuple):
                ii = subima[0]
                jj = subima[1]
                extent = [subima[1][0]*self.stepx+self.startx,
                          subima[1][1]*self.stepx+self.startx,
                          subima[0][0]*self.stepy+self.starty,
                          subima[0][1]*self.stepy+self.starty]
            else:
                raise TypeError("Subima must a list of 2 tuples or two lists")
    else:
        extent = [self.startx-self.stepx/2.,self.endx+self.stepx/2.,
                  self.starty-self.stepy/2.,self.endy+self.stepy/2.]
        ii = [0,num.shape(data)[0]]
        jj = [0,num.shape(data)[1]]

    if ima:
        pylab.imshow(data[ii[0]:ii[1],jj[0]:jj[1]],
                     interpolation='nearest',aspect=aspect,cmap=cmap,
                     vmin=self.vmin,vmax=self.vmax,extent=extent,
                     origin='lower',alpha=alpha)
    if contour:
        levels = self.vmin + \
                 num.arange(ncontour)*(self.vmax-self.vmin)/ncontour
        pylab.contour(data[ii[0]:ii[1],jj[0]:jj[1]],levels,
                      extent=extent,cmap=line_cmap,linewidth=linewidth)
    if self.labx is not None:
        pylab.xlabel(self.labx)
    if self.laby is not None:
        pylab.ylabel(self.laby)

image_array.display = imarray_display

# SNIFS_cube methods ==============================

def cube_plot_spec(self,no=None,ind=None,mask=None,var=False,
                   intervals=None,color='b',ax=None):
    """
    Plot a spectrum extracted from the datacube.
    @param no: lenslet number in the datacube
    @param ind: index in the data array
    @param mask: optional array having the same shape than the data field of the cube. If given, the
        spectrum returned is the sum of the data array multiplied by the mask.
    @param var: Variance flag. If set to True, the variance spectrum is ploted.
    @param intervals: A list of 2 elements tuples defining the intervals in wavelength to be plotted
    @param color: line color in pylab syntax
    @param ax: pylab axes on which the spectrum will be ploted. If set to None, a new axes is created.
    """
    spec = self.get_spec(no)
    spec.overplot(intervals=intervals,color=color,ax=ax,var=var)
    pylab.draw()

def cube_disp_slice(self,n=None,coord='w',weight=None,aspect='equal',
                    scale='lin',vmin=None,vmax=None,cmap=pylab.cm.jet,
                    var=False,contour=False,ima=True,nx=15,ny=15,NAN=True):
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
    @param weight: Spectrum giving the weights to be applied to each slice before lambda integration
    @param NAN: Flag to set non existing spaxel value to nan. if NAN=False, non existing spaxels will be set to 0
    """
    slice = self.slice2d(n,coord,var=var,nx=nx,ny=ny,weight=weight,NAN=NAN)
    if vmin != None and vmax != None:
        if vmin > vmax:
            raise ValueError("vmin must be lower than vmax.")
    if scale=='log':
        slice = num.log(slice)
        if vmin!=None:
            if vmin<=0:
                raise ValueError("In log scale vmin and vmax must be positive")
            vmin = float(num.log(vmin))
        if vmax!=None:
            if vmax<=0:
                raise ValueError("In log scale vmin and vmax must be positive")
            vmax = float(num.log(vmax))

    if vmin is None:
        vmin = float(min(num.ravel(slice)))
    if vmax is None:
        vmax = float(max(num.ravel(slice)))

    fig = pylab.gcf()
    extent = [-1./2.,ny-1/2.,-1/2.,nx-1/2.]

    if ima:
        pylab.imshow(slice,interpolation='nearest',aspect=aspect, \
                     vmin=vmin,vmax=vmax,cmap=cmap, \
                     origin='lower',extent=extent)
    if contour:
        levels = vmin + num.arange(10)*(vmax-vmin)/10.
        pylab.contour(slice,levels,cmap=pylab.cm.gray)

def cube_disp_data(self,vmin=None,vmax=None,cmap=pylab.cm.hot,var=False):
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
    med = float(num.median(num.ravel(data)))
    disp = float(num.sqrt(num.median((num.ravel(data)-med)**2)))
    if vmin is None:
        vmin = med - 3*disp
    if vmax is None:
        vmax = med + 10*disp
    extent = [self.lstart,self.lend,-1./2.,self.nlens-1./2.]
    pylab.imshow(num.transpose(self.data),vmin=vmin,vmax=vmax,extent=extent,
                 interpolation='nearest',aspect='auto')

SNIFS_cube.plot_spec = cube_plot_spec
SNIFS_cube.disp_slice = cube_disp_slice
SNIFS_cube.disp_data = cube_disp_data

# SNIFS_mask methods ==============================

def mask_plot(self,no_list=None,interpolate=False,lbda=None,symbol='k-'):
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
        if no not in self.no:
            raise ValueError('lens #%d not present in mask'%no)
    for no in no_list:
        if lbda is None:
            x,y = self.get_spec_lens(no)
            if interpolate:
                y = y[0] + num.arange(100)*(y[len(y)-1]-y[0]-1)/100. 
                x = self.interpolate(no,y)
        else:
            x,y = self.get_coord_lens(no,lbda)
            xx = [x,x]
            yy = [y,y]
            x = xx
            y = yy
        pylab.plot(x,y,symbol)

SNIFS_mask.plot = mask_plot
