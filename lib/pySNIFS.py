import os
import pylab
import pyfits
import numarray
from numarray import convolve
from numarray import linear_algebra as L
import scipy
import matplotlib
from scipy import optimize,size,special,Float64
from scipy.special import *
from scipy import interpolate as I



########################   I/O functions   #########################

class spectrum:
    """
    1D spectrum class.
    """
    def __init__(self,data_file=None,var_file=None,no=None,num_array=True,x=None,data=None,var=None,start=None,step=None,nx=None):
        """
        initiating the class.
        data_file: fits file from which the data are read. It can be a 1D fits image or a euro3d
              datacube. In the euro3d case, the variance spectrum is read from this cube if present.
              In the 1D fits image file case, if this file contains an image extension, the variance 
              spectrum is read from it.
        data_var: fits file from which the variance is read if it is not present in the data file. 
              It must be a 1D fits image.
        no: number of the spaxel in the datacube. Needed only if the input file is a euro3d cube.
        
        The following parameters are used when the spectrum is not read from a data file.
        x: array containing the x coordinates in the spectrum. It is used only if the spectrum
           is not regularly sampled
        data: array containing the data of the spectrum
        var: array containing the variance of the spectrum
        start: coordinate in user coordinates of the first point of the data array.
        step: step in user coordinates of the data.
        nx: number of data point. It is usefull when the user want to create an array of zero
            data. otherwise, this value is the size of the data array.
        num_array: flag to say if the data are stored in numarray or in numeric.
        
        """
        self.file = file
        if data_file != None:
            data_fits = pyfits.open(data_file)
            if data_fits[0].header.has_key('EURO3D'):
                # Case where the spectrum belongs to an e3d datacube
                if no == None:
                    raise ValueError, 'The user must provide the spectrum number in the datacube'
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
                elif var_file != None:
                    # The variance is read from a fits file 
                    var_fits = pyfits.open(var_file)
                    self.var = var_fits[0].data
                else:
                    self.var = None
                if isinstance(self.var,type(None)):
                    if len(self.var) != len(self.data):
                        raise ValueError, 'Data and variance spectra must have the same length'
                self.len = data_fits[0].header.get('naxis1')
                self.x = data_fits[0].header.get('crval1') + arange(data_fits[0].header.get('naxis1')) * data_fits[0].header.get('cdelt1')
                self.step = data_fits[0].header.get('cdelt1')
                self.start = data_fits[0].header.get('crval1')
        else:
            if x == None:
                # Case for a regularly sampled spectrum
                if start == None or step == None or nx == None:
                    if data == None:
                        self.data = None
                        self.var = None
                        self.len = None
                        self.start = None
                        self.step = None
                        self.x = None
                    else:
                        self.data = data
                        self.var = var
                        self.len = len(data)
                        if not isinstance(var,type(None)):
                            if len(var) != len(data):
                                raise ValueError, 'data and variance array must have the same length'  
                        self.x = arange(len(data))
                        self.start = 0
                        self.step = 1
                else:
                    self.start = start
                    self.step = step
                    self.len = len(x)
                    self.x = start + arange(nx)*step
                    if data == None:
                        self.data = zeros(nx)
                        self.var = zeros(nx)
                        self.x = start + arange(nx)*step
                    else:
                        if len(data) != len(x):
                            raise ValueError, 'data and x arrays must have the same length'
                        self.data = data
                        self.var = var
                        if not isinstance(var,type(None)):
                            if len(var) != len(data):
                                raise ValueError, 'data and variance array must have the same length'  
            else:
                # Case for a not regularly sampled spectrum
                self.start = None
                self.step = None
                self.len = len(x)
                if data == None:
                        self.data = zeros(len(x))
                        self.var = zeros(len(x))
                        self.x = x
                else:
                    if len(data) != len(x):
                        raise ValueError, "x and data arrays must have the same size"
                    if not isinstance(var,type(None)):
                        if len(var) != len(data):
                            raise ValueError, "data and var arrays must have the same size"
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
            if not isinstance(self.var,type(None)):
                self.var = scipy.array(self.var)
              
        if not isinstance(self.var,type(None)):
            self.has_var = True
        
        if self.len != None:
            self.index_list = arange(self.len).tolist()
            self.intervals = [(1,self.len)]
         
        self.curs_val = []
        self.cid = None       
        
    def plot(self,intervals=None,var=None,line='-',color='b'):
        line = line+color
        if not isinstance(var,type(None)):
            data = self.var
        else:
            data = self.data
        pylab.figure()  #modif du 11-10-05 (GR)
        if intervals == None:
            ind_intervals = self.intervals
        else:
            self.subset(intervals=intervals)
                                   
        for ind_interval in ind_intervals:
            if len(ind_interval) == 2:
                pylab.plot(self.x[ind_interval[0]:ind_interval[1]],data[ind_interval[0]:ind_interval[1]],line)
            else:
                pylab.plot(self.x[ind_interval[0]:ind_interval[1]],data[ind_interval[0]:ind_interval[1]],ind_interval[2])

    def overplot(self,intervals=None,var=None,line='-',color='b'): # modif du 11-10-05 (GR)
        line = line+color
        if not isinstance(var,type(None)):
            data = self.var
        else:
            data = self.data
        if intervals == None:      
            ind_intervals = self.intervals
        else:
            self.subset(intervals=intervals)
                                   
        for ind_interval in ind_intervals:
            if len(ind_interval) == 2:
                pylab.plot(self.x[ind_interval[0]:ind_interval[1]],data[ind_interval[0]:ind_interval[1]],line)
            else:
                pylab.plot(self.x[ind_interval[0]:ind_interval[1]],data[ind_interval[0]:ind_interval[1]],ind_interval[2])
                
  
    def subset(self,intervals=None,reject=None):
        self.index_list = []
        self.intervals = []
        if intervals == None:
            return
        if not isinstance(intervals,list):
            raise ValueError, "Interval must be provided as a list of 2 elements tuples"
        else:
            for interval in intervals:
                if not isinstance(interval,tuple):
                    raise ValueError, "Intervals must be provided as a list of 2 elements tuples"
                else:
                    if len(interval) == 2:
                        if not isscalar(interval[0]) or interval[0] < min(self.x) or interval[1] > max(self.x):
                            raise ValueError, "Interval bounds must be numbers between xmin and xmax."
                        else:
                            ind_min = self.index(interval[0])
                            ind_max = self.index(interval[1])
                            #self.index_list.append((arange(ind_max-ind_min+1)+ind_min).tolist())
                            self.index_list = self.index_list + (arange(ind_max-ind_min+1)+ind_min).tolist()
                            self.intervals.append((ind_min,ind_max))
                    else:
                        raise ValueError, "Interval must be provided as a list of 2 elements tuples"
        self.len = len(self.index_list)
     
    def cursor_on(self,print_flag=True):
        def click(event):
            if event.button == 2:
                self.curs_val.append([event.xdata,event.ydata])
                if print_flag:
                    print 'x = %e  y = %e'%(event.xdata,event.ydata)
        if self.cid != None:
            self.cursor_off()
            
        self.cid = pylab.connect('button_press_event',click)
    
    def cursor_off(self):
        if self.cid != None:
            pylab.disconnect(self.cid)
            self.curs_val = []
            self.cid = None
            
    def reset(self):
        self.index_list = arange(self.len).tolist()
        self.len = len(self.data)
        self.intervals = [(1,self.len)]

    def zeros_from(self,spec):
        self.x = spec.x
        self.data = spec.data * 0.
        self.len = spec.len

    def ones_from(self,spec):
        self.x = spec.x
        self.data = spec.data * 0. + 1.
        self.len = spec.len

    def index(self,lbda):
        return argmin((self.x - lbda)**2)

                        
    def WR_fits_file(self,fits_file,header_list=None):
        hdu = pyfits.PrimaryHDU()
        hdu.data = numarray.array(self.data)
        hdu.header.update('NAXIS', 1)
        hdu.header.update('NAXIS1', self.len,after='NAXIS')
        hdu.header.update('CRVAL1', self.start)
        hdu.header.update('CDELT1', self.step)
        if header_list != None:
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
        hdulist.append(hdu_var)
        if os.path.isfile(fits_file):
            os.system('rm %s'%fits_file)
        hdulist.writeto(fits_file)

########################## Image #######################################

class image_array:
    """
    2D image class
    """
    def __init__(self,data=None,var=None,startx=0,stepx=1,starty=0,stepy=1,endx=None,endy=None,labx=None,laby=None):
        if data == None:
            raise ValueError, 'The user must provide a data array.'
        elif len(shape(data)) != 2:
            raise ValueError, 'The user must provide a two dimensions data array.'
        
        self.data = data
        self.var = var
        self.startx = startx
        self.starty = starty
        self.stepx = stepx
        self.stepy = stepy
        if endx == None:
            self.endx = startx + len(data[0])*stepx
        else:
            self.endx = endx
            self.stepx = (endx-startx)/(shape(data)[0]-1)
        if endy == None:
            self.endy = starty + len(data[:,0])*stepy
        else:
            self.endy = endy
            self.stepy = (endy-starty)/(shape(data)[1]-1)
            
        self.labx = labx
        self.laby = laby
        self.vmin = None
        self.vmax = None
        self.nx = shape(data)[0]
        self.ny = shape(data)[1]

        
    def display(self,cmap=pylab.cm.hot,aspect='free',vmin=None,vmax=None,subima=None):
        if vmin != None: self.vmin = vmin
        if vmax != None: self.vmax = vmax
        if subima != None:
            if isinstance(subima,list):
                if isinstance(subima[0],list) and isinstance(subima[1],list):
                    indy = [int((subima[0][0]-self.startx)/self.stepx),\
                            int((subima[0][1]-self.startx)/self.stepx)]
                    indx = [int((subima[1][0]-self.starty)/self.stepy),\
                            int((subima[1][1]-self.starty)/self.stepy)]
                    extent = [subima[0][0],subima[0][1],subima[1][0],subima[1][1]]
                elif isinstance(subima[0],tuple) and isinstance(subima[1],tuple):
                    indx = subima[0]
                    indy = subima[1]
                    extent = [subima[0][0]*self.stepx+self.startx,subima[0][1]*self.stepx+self.startx,\
                              subima[1][0]*self.stepy+self.starty,subima[1][1]*self.stepy+self.starty]
                else:
                    raise TypeError, "Subima must a list of 2 tuples or two lists"
        else:
            extent = [self.startx,self.endx,self.starty,self.endy]
            indx = [0,shape(self.data)[0]]
            indy = [0,shape(self.data)[1]]
            
        pylab.imshow(self.data[indx[0]:indx[1],indy[0]:indy[1]],interpolation='nearest',aspect=aspect,cmap=cmap,\
                     vmin=self.vmin,vmax=self.vmax,extent=extent,origin='lower')
        pylab.xlabel(self.labx)
        pylab.ylabel(self.laby)

    def WR_fits_file(self,nom_ima,mode='w+'):
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
        hdu.data = numarray.array(self.data)
        hdulist.append(hdu)
        if mode == 'w+':
            if os.path.isfile(nom_ima):
                os.system('rm %s'%nom_ima)
        hdulist.writeto(nom_ima)


########################## Cube ########################################

class SNIFS_cube:

    def __init__(self,e3d_file_tmp=None,l=None,num_array=True,lbda=None,s=False,threshold=1e20):
        
        if e3d_file_tmp != None:
            e3d_file = pyfits.open(e3d_file_tmp)
            if not ('EURO3D','T') in e3d_file[0].header.items():
                raise ValueError, "This is not a e3d file"
            self.from_e3d_file = True
            self.e3d_file = e3d_file_tmp
            # The header of the data extension is stored in a field of the class
            self.e3d_data_header = e3d_file[1].header.items()
            # The group definition HDU and the optional extensions HDU are stored in fields of the class
            self.e3d_grp_hdu = e3d_file[2]
            self.e3d_extra_hdu_list = [e3d_file[i] for i in arange(len(e3d_file)-3)+3] 
            ref_start = e3d_file[1].header['CRVALS']
            step      = e3d_file[1].header['CDELTS']
            self.lstep = step
            if 'STAT_SPE' in e3d_file[1].columns.names:
                var  = e3d_file[1].data.field('STAT_SPE')
            else:
                var = None
            data  = e3d_file[1].data.field('DATA_SPE')
            spec_sta  = e3d_file[1].data.field('SPEC_STA')
            spec_len  = e3d_file[1].data.field('SPEC_LEN')
            spec_end  = spec_len + spec_sta
            npts      = e3d_file[1].data.getshape()[0]
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

            

            if l != None:
                if l[0] != None and l[1] != None:
                    lmin,lmax = max([common_lstart,l[0]]),min([common_lend,l[1]])       
                    if len(l) == 3:
                        lstep = l[2]
                    else:
                        lstep=1
                else:
                    if len(l) == 3:
                        lstep = l[2]
                    else:
                        lstep=1
            else:
                lmin,lmax = common_lstart,common_lend
               
            if lmin < 0:
                raise ValueError, "A slice index cannot be negative."
            
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
            self.lstart = self.lbda[0]

            self.x = e3d_file[1].data.field('XPOS')
            self.y = e3d_file[1].data.field('YPOS')
            self.no = e3d_file[1].data.field('spec_id')
            # We read in the extension table the I and J index of the lenslets. As the extension table may have
            # a different number of lenslets than the data table, we first search the index of the common
            # lenslets.
            ind = [e3d_file[3].data.field('NO').tolist().index(i) for i in e3d_file[1].data.field('spec_id')]
            self.i = e3d_file[3].data.field('I')[ind]+7
            self.j = e3d_file[3].data.field('J')[ind]+7
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
            self.from_e3d_file = False
            self.nlens = 225
            if lbda == None:
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
          
    def slice2d(self,n,coord='w',var=False):
        if isinstance(n,list):
            if coord == 'p':
                n1 = n[0]
                n2 = n[1]
            elif coord == 'w':
                n1 = argmin((self.lbda-n[0])**2)
                n2 = argmin((self.lbda-n[1])**2)
            else:
                raise ValueError, "Coordinates flag should be either \'p\' or \'w\'"
            if n1 == n2:n2=n2+1
        else:
            if coord == 'p':
                n1 = n
                n2 = n+1
            elif coord == 'w':
                n1 = argmin((self.lbda-n)**2)
                n2 = n1+1
            else:
                raise ValueError, "Coordinates flag should be either \'p\' or \'w\'"

        if n1 >= 0 and n2 <= numarray.shape(self.data)[0]:
            slice_2D = numarray.zeros((15,15),Float32) * nan
            i = numarray.array(self.i)
            j = numarray.array(self.j)
            if var:
                slice_2D[i,j] = sum(self.var[n1:n2])
            else:
                slice_2D[i,j] = sum(self.data[n1:n2])
            return(slice_2D)
        else:
            raise IndexError, "no slice #%d" % n 

    def spec(self,no=None,ind=None,mask=None,var=False):
        if var:
            data = self.var
        else:
            data = self.data
            
        if mask != None:
            data = data * mask
            return sum(data,1)
        
        if (no != None and ind != None) or (no == None and ind == None):
            raise TypeError, "lens number (no) OR spec index (ind) should be given."
        else:
            if (no != None):
                if no in self.no.tolist():
                    return data[:,argmax(self.no == no)]
                else:
                    raise IndexError, "no lens #%d" % no
            else:
                if not isinstance(ind,list):
                    if 0 <= ind < numarray.shape(data)[1]:
                        return data[:,ind]
                    else:
                        raise IndexError, "no index #%d" % ind
                else:
                    if 0 <= ind[0] and ind[1] < numarray.shape(data)[1]:
                        if ind[0] == ind[1]: ind[1] = ind[1]+1
                        return sum(data[:,ind[0]:ind[1]],1)
                    else:
                        raise IndexError, "Index list out of range"
                        
    def plot_spec(self,no=None,ind=None,mask=None,ax=None,var=False):
        if ax == None:
            pylab.plot(self.lbda,self.spec(no=no,ind=ind,mask=mask,var=var))
        else:
            ax.plot(self.lbda,self.spec(no=no,ind=ind,mask=mask,var=var))
        pylab.show()

    def get_spec(self,no):
        spec = spectrum(x=self.lbda,data=self.spec(no))
        if hasattr(self,'lstep'):
            spec.step = self.lstep
        if hasattr(self,'lstart'):
            spec.start = self.lstart

        return spec
    
    def disp_slice(self,n,coord='w',vmin=None,vmax=None,cmap=pylab.cm.hot,var=False):
        slice = self.slice2d(n,coord,var=var)
        med = scipy.median(ravel(slice))
        disp = sqrt(scipy.median((ravel(slice)-med)**2))
        if vmin == None:
            vmin = med - 3*disp
        if vmax == None or vmax < vmin:
            vmax = med + 10*disp
        fig = pylab.gcf()
        fig.clf()
        q = fig.get_figheight()/fig.get_figwidth()
        fig.add_axes((0.5-q*0.65/2.,0.3,0.65*q,0.65))
        pylab.imshow(slice,interpolation='nearest',aspect='preserve',vmin=vmin,vmax=vmax,cmap=cmap)
        fig.add_axes((0.5-q*0.65/2.,0.05,0.65*q,0.2))
        dum0,dum1,dum2 = pylab.hist(slice,arange(vmin,vmax,(vmax-vmin)/100))

    def disp_data(self,vmin=None,vmax=None,var=False):
        if var:
            data = self.var
        else:
            data = self.data
        med = scipy.median(ravel(data))
        disp = sqrt(scipy.median((ravel(data)-med)**2))
        if vmin == None:
            vmin = med - 3*disp
        if vmax == None:
            vmax = med + 10*disp
        pylab.imshow(self.data,interpolation='nearest',vmin=vmin,vmax=vmax)    
            
    def zeros_from(self,SNIFS_cube):
        self.x = SNIFS_cube.x
        self.y = SNIFS_cube.y
        self.i = SNIFS_cube.i
        self.j = SNIFS_cube.j
        self.no = SNIFS_cube.no
        self.lbda = SNIFS_cube.lbda
        self.nlens = SNIFS_cube.nlens
        self.nslice = SNIFS_cube.nslice
        self.data = SNIFS_cube.data * 0.
        self.var = SNIFS_cube.data * 0.

    def ones_from(self,SNIFS_cube):
        self.x = SNIFS_cube.x
        self.y = SNIFS_cube.y
        self.i = SNIFS_cube.i
        self.j = SNIFS_cube.j
        self.no = SNIFS_cube.no
        self.lbda = SNIFS_cube.lbda
        self.nlens = SNIFS_cube.nlens
        self.nslice = SNIFS_cube.nslice
        self.data = SNIFS_cube.data * 0. + 1
        self.var = SNIFS_cube.data * 0.

    def get_no(self,i,j):
        if i>max(self.i) or i<min(self.i) or j>max(self.j) or j<min(self.j):
            raise ValueError, "Index out of range."
        no = self.no[argmax((self.i == i)*(self.j == j))]
        return(no)

    def get_ij(self,no):
        if no>max(self.no) or no<min(self.no):
            raise ValueError, "Lens number out of range."
        i = self.i[argmax(self.no == no)]
        j = self.j[argmax(self.no == no)]
        return((i,j))

    def get_lindex(self,val):
        if isinstance(val,tuple):
            if val[0]>max(self.i) or val[0]<min(self.i) or val[1]>max(self.j) or val[1]<min(self.j):
                raise ValueError, "Index out of range."
            ind = argmax((self.i == val[0])*(self.j == val[1]))
        else:
            ind = argmax(self.no == val)

        return(ind)

    def WR_e3d_file(self,fits_file):
        if not self.from_e3d_file:
            raise ValueError,"Writing e3d file from scratch not yet implemented"
        data_list = transpose(self.data).tolist()
        start_list = [self.lstart for i in arange(self.nlens)]
        no_list = self.no.tolist()
        xpos_list = self.x.tolist()
        ypos_list = self.y.tolist()
        WR_e3d_file(data_list,None,no_list,start_list,self.lstep,xpos_list,ypos_list,\
                    fits_file,self.e3d_data_header,self.e3d_grp_hdu,self.e3d_extra_hdu_list)
        

#####################     SNIFS masks     ########################

class SNIFS_mask:
    def __init__(self,mask,offx=0,offy=0,step=200,path=None,order=1):
        if path == None:
            if not 'SNIFS_PATH' in os.environ.keys():
                raise ValueError, 'The user must provide the path where the mask can be found or set an environment variable giving the path of\
                the Snifs software'
            else:
                maskdir = os.environ['SNIFS_PATH']+'/pkg/pipeline/data/'
                bindir = os.environ['SNIFS_PATH']+'/user/bin/'
                mask = maskdir+mask
        
        mask_tbl = pyfits.open(mask) 
        os.system('%s/plot_optics -mask %s -offset %f,%f -orders %d,%d -step %d -local -table ./tmp_tbl.fits -inputformat euro3d -outputformat euro3d > ./tmp_optics'%(bindir,mask,offx,offy,order,order,step))
        print '%s/plot_optics -mask %s -offset %f,%f -orders %d,%d -step %d -local -table ./tmp_tbl.fits -inputformat euro3d -outputformat euro3d > ./tmp_optics'%(bindir,mask,offx,offy,order,order,step)
        tmp_tbl = pyfits.open('tmp_tbl.fits')
        i = 1
        self.no = mask_tbl[1].data.field('no').tolist()
        self.lbda_list = []
        while tmp_tbl[1].header.has_key('lbda%d'%i):
            self.lbda_list.append(tmp_tbl[1].header.get('lbda%d'%i))
            i = i+1

        self.x = {}
        self.y = {}
        for i,l in enumerate(self.lbda_list):
            self.x[l] = tmp_tbl[1].data.field('XLBDA%d'%(i+1))
            self.y[l] = tmp_tbl[1].data.field('YLBDA%d'%(i+1))

    def get_spec_lens(self,no):
        i = self.no.index(no)
        x = [self.x[l][i] for l in self.lbda_list]
        y = [self.y[l][i] for l in self.lbda_list]
        return x,y

    def get_coord_lens(self,no,lbda):
        x,y = self.get_spec_lens(no)
        tckx = I.splrep(self.lbda_list,x,s=0)
        tcky = I.splrep(self.lbda_list,y,s=0)
        x = I.splev(lbda,tckx)
        y = I.splev(lbda,tcky)
        return x,y
    
    def plot(self,no_list=None,interpolate=False,symbol='k-',lbda=None):
        if no_list == None:
            no_list = self.no
        elif isinstance(no_list,int):
            no_list = [no_list]
        for no in no_list:
            if not no in self.no:
                raise ValueError, 'lens #%d not present in mask'%no
        for no in no_list:
            if lbda == None:
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
        if not no in self.no:
            raise ValueError, 'lens #%d not present in mask'%no
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
    table: Input table
    colx: x column in the table
    coly: y column in the table
    ref_pos: array giving the x positions where to compute the interpolated values
    """
    tck = scipy.interpolate.splrep(table[1].data.field(colx),\
                                   table[1].data.field(coly),s=0)
    tab = scipy.interpolate.splev(ref_pos,tck)
    return tab

def histogram(data,nbin=None,Min=None,Max=None,bin=None,cumul=False):
    """
    Compute the histogram of an array
    data: Input array
    nbin: number of bins in the histogram
    Min,Max: Interval of values between which the histogram is computed
    bin: Size of the bins. If not given it is computed from the number of bins requested.
    cumul: If True, compute a cumulative histogram
    """
    if Min == None:
        Min = min(data)
    if Max == None:
        Max = max(data)
    if bin == None:
        bin = (Max-Min)/nbin
        
    bin_array = arange(nbin)*bin + Min
    n = searchsorted(sort(data), bin_array)
    n = concatenate([n, [len(data)]])
    hist = spectrum()
    hist.data = n[1:]-n[:-1]
    hist.x = bin_array
    hist.len = len(bin_array)
    if cumul:
        hist.data = numarray.array([float(sum(hist.data[0:i+1])) for i in arange(hist.len)])/float(sum(hist.data))
    return hist

def common_bounds_cube(cube_list):
    """
    Computes the common bounds of a list of pySNIFS cubes
    cube_list: Input list of spectra
    returns:
    imin: list of the indexes of the lower common bound for each cube
    imax: list of the indexes of the upper common bound for each cube
    """
    if False in [hasattr(cube,'lstep') for cube in cube_list]:
        raise ValueError, "Missing attribute lstep in datacubes."
    else:
        if L.mlab.std([cube.lstep for cube in cube_list]) != 0:
            raise ValueError, "All cubes should have the same step."
        xinf = max([min(cube.lbda) for cube in cube_list])
        xsup = min([max(cube.lbda) for cube in cube_list])
        imin = [argmin(abs(cube.lbda-xinf)) for cube in cube_list]
        imax = [argmin(abs(cube.lbda-xsup)) for cube in cube_list]

    return imin,imax

def common_bounds_spec(spec_list):
    """
    Computes the common bounds of a list of pySNIFS spectra
    spec_list: Input list of spectra
    returns:
    imin: list of the indexes of the lower common bound for each spectrum
    imax: list of the indexes of the upper common bound for each spectrum
    """
    if False in [hasattr(spec,'step') for spec in spec_list]:
        raise ValueError, "Missing attribute lstep in spectra."
    else:
        if L.mlab.std([spec.step for spec in spec_list]) != 0:
            raise ValueError, "All spectra should have the same step."
        xinf = max([min(spec.x) for spec in spec_list])
        xsup = min([max(spec.x) for spec in spec_list])
        imin = [argmin(abs(spec.x-xinf)) for spec in spec_list]
        imax = [argmin(abs(spec.x-xsup)) for spec in spec_list]

    return imin,imax

def common_lens(cube_list):
    """
    Give the common lenses of a list of pySNIFS cubes
    cube_list: input list of datacubes
    returns:
    inters: list of the lenses common to all the cubes of the list
    """
    inters = cube_list[0].no
    for i in arange(len(cube_list)-1)+1:
        inters = filter(lambda x:x in cube_list[i].no,inters) 
    return inters

def fit_poly(y,n,deg,x=None):
    """
    Fit a polynomial whith median sigma clipping on an array y
    y: Input array giving the ordinates of the points to be fitted
    n: rejection threshold
    deg: Degree of the polunomial
    x: Optional input array giving the abscissae of the points to be fitted. If not given
       the abscissae are taken as an array [1:len(y)]
    """
    if x == None:
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
       data_list: List of the spectra of the datacube
       var_list: List of the variance spectra of the data_cube
       no_list: List of the spaxel ident number of the datacube
       start_list: List of the start wavelength of each spectrum of the datacube
       step: Wavelength step of the datacube
       xpos_list: List of the x position of each spaxel of the datacube
       ypos_list: List of the x position of each spaxel of the datacube
       fits_file: Output fits file to be written on disk
       data_header: data header of the new e3d file. All the standards e3d
                    headers containing information on the data themselves
                    will be overwritten. Only the non standard will be copied
                    from this parameter.
       grp_hdu: pyfits HDU describing the spaxel groups.
       extra_hdu_list: pyfits HDU containing non e3d-mandatory data.
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
    if os.path.isfile(fits_file):
        os.system('rm %s'%fits_file)
    hdu_list.writeto(fits_file)
        


            

    

    
    
