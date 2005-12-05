import pylab
import pyfits
import numarray
import scipy
from scipy import optimize,size,special,Float64
from scipy.special import *



########################   I/O functions   #########################

class spectrum:

    def __init__(self,file=None,l=None,num_array=True,x=None,data=None):
        if file != None:
            tmp = pyfits.open(file)
            self.data = tmp[0].data
            self.len = tmp[0].header.get('naxis1')
            self.x = tmp[0].header.get('crval1') + arange(tmp[0].header.get('naxis1')) * tmp[0].header.get('cdelt1')
        else:
            if x == None:
                self.data = None
                self.x = None
                self.len = None
            else:
                self.len = len(x)
                if data == None:
                    if num_array:
                        self.data = numarray.zeros(len(x))
                        self.x = numarray.array(x)
                    else:
                        self.data = scipy.array(len(x))
                        self.x = scipy.array(x)
                else:
                    if len(data) != len(x):
                        raise ValueError, "x and data arrays must have the same size"
                    if num_array:
                        self.data = numarray.array(data)
                        self.x = numarray.array(x)
                    else:
                        self.data = scipy.array(data)
                        self.x = scipy.array(x)
                    
    def plot(self,intervals=None):
	pylab.figure()  #modif du 11-10-05
        if intervals == None:
            pylab.plot(self.x,self.data)
        else:
            ind_intervals = []
            if not isinstance(intervals,list):
                raise ValueError, "Interval must be provided as a list of 2 elements tuples"
            else:
                for interval in intervals:
                    if not isinstance(interval,tuple):
                        raise ValueError, "Interval must be provided as a list of 2 elements tuples"
                    else:
                        if len(interval) == 2 or len(interval) == 3:
                            if not isscalar(interval[0]) or interval[0] < min(self.x) or interval[1] > max(self.x):
                                raise ValueError, "Interval bounds must be numbers between xmin and xmax."
                            else:
                                ind_min = argmin(abs(self.x - interval[0]))
                                ind_max = argmin(abs(self.x - interval[1]))
                                ind_intervals.append((ind_min,ind_max))
                            if len(interval) == 3:
                                ltype = interval[2]
                                ind_intervals[len(ind_intervals)-1] = ind_intervals[len(ind_intervals)-1] + (ltype,)
                        else:
                            raise ValueError, "Interval must be provided as a list of 2 or 3 elements tuples"
                        
                for ind_interval in ind_intervals:
                    if len(ind_interval) == 2:
                        pylab.plot(self.x[ind_interval[0]:ind_interval[1]],self.data[ind_interval[0]:ind_interval[1]],'k-')
                    else:
                        pylab.plot(self.x[ind_interval[0]:ind_interval[1]],self.data[ind_interval[0]:ind_interval[1]],ind_interval[2])

    def overplot(self,intervals=None): # modif du 11-10-05
        if intervals == None:
            pylab.plot(self.x,self.data)
        else:
            ind_intervals = []
            if not isinstance(intervals,list):
                raise ValueError, "Interval must be provided as a list of 2 elements tuples"
            else:
                for interval in intervals:
                    if not isinstance(interval,tuple):
                        raise ValueError, "Interval must be provided as a list of 2 elements tuples"
                    else:
                        if len(interval) == 2 or len(interval) == 3:
                            if not isscalar(interval[0]) or interval[0] < min(self.x) or interval[1] > max(self.x):
                                raise ValueError, "Interval bounds must be numbers between xmin and xmax."
                            else:
                                ind_min = argmin(abs(self.x - interval[0]))
                                ind_max = argmin(abs(self.x - interval[1]))
                                ind_intervals.append((ind_min,ind_max))
                            if len(interval) == 3:
                                ltype = interval[2]
                                ind_intervals[len(ind_intervals)-1] = ind_intervals[len(ind_intervals)-1] + (ltype,)
                        else:
                            raise ValueError, "Interval must be provided as a list of 2 or 3 elements tuples"
                        
                for ind_interval in ind_intervals:
                    if len(ind_interval) == 2:
                        pylab.plot(self.x[ind_interval[0]:ind_interval[1]],self.data[ind_interval[0]:ind_interval[1]],'k-')
                    else:
                        pylab.plot(self.x[ind_interval[0]:ind_interval[1]],self.data[ind_interval[0]:ind_interval[1]],ind_interval[2])

    def zeros_from(self,spec):
        self.x = spec.x
        self.data = spec.data * 0.
        self.len = spec.len

    def ones_from(self,spec):
        self.x = spec.x
        self.data = spec.data * 0. + 1.
        self.len = spec.len

########################## Cube ########################################

class SNIFS_cube:

    def __init__(self,e3d_file_tmp=None,l=None,noise=False,num_array=True,lbda=None,s=False):
        if e3d_file_tmp != None:
	    e3d_file = pyfits.open(e3d_file_tmp)
            ref_start = e3d_file[1].header['CRVALS']
            step      = e3d_file[1].header['CDELTS']
            if noise:
                data  = e3d_file[1].data.field('STAT_SPE')
            else:
                data  = e3d_file[1].data.field('DATA_SPE')
            spec_sta  = e3d_file[1].data.field('SPEC_STA')
            spec_len  = e3d_file[1].data.field('SPEC_LEN')
            spec_end  = spec_len + spec_sta
            npts      = e3d_file[1].data.getshape()[0]
            common_lstart,common_lend,lstep = max(spec_sta),min(spec_end),1
            common_start = ref_start + common_lstart * step

            tdata = numarray.transpose(numarray.array([data[i][common_lstart-spec_sta[i]:common_lend-spec_sta[i]] \
                                                       for i in range(len(data))]))
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
                self.lbda = lbda[lmin - common_lstart:lmax - common_lstart:lstep]
            else:
                self.data = numarray.convolve.boxcar(tdata,(lstep,1))\
                            [lmin - common_lstart + lstep/2:lmax - common_lstart+lstep/2:lstep]
                self.lbda = lbda[lmin - common_lstart+lstep/2:lmax - common_lstart+lstep/2:lstep]
            self.x = e3d_file[1].data.field('XPOS')
            self.y = e3d_file[1].data.field('YPOS')
            self.i = e3d_file[3].data.field('I')+7
            self.j = e3d_file[3].data.field('J')+7
            self.no = e3d_file[3].data.field('NO')
            if not num_array:
                self.data = scipy.array(self.data)
                self.lbda = scipy.array(self.lbda)
                self.x = scipy.array(self.x)
                self.y = scipy.array(self.y)
                self.i = scipy.array(self.i)
                self.j = scipy.array(self.j)
                self.no = scipy.array(self.no)
                
            self.nslice = len(self.lbda)
            self.nlens = len(self.x)
        else:
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
                    
    def slice2d(self,n,coord='w'):
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
            slice_2D[i,j] = sum(self.data[n1:n2])
            return(slice_2D)
        else:
            raise IndexError, "no slice #%d" % n 

    def spec(self,no=None,ind=None,mask=None):
        if mask != None:
            data = self.data * mask
            return sum(data,1)
        
        if (no != None and ind != None) or (no == None and ind == None):
            raise TypeError, "lens number (no) OR spec index (ind) should be given."
        else:
            if (no != None):
                if no in self.no.tolist():
                    return self.data[:,argmax(self.no == no)]
                else:
                    raise IndexError, "no lens #%d" % no
            else:
                if not isinstance(ind,list):
                    if 0 <= ind < numarray.shape(self.data)[1]:
                        return self.data[:,ind]
                    else:
                        raise IndexError, "no index #%d" % ind
                else:
                    if 0 <= ind[0] and ind[1] < numarray.shape(self.data)[1]:
                        if ind[0] == ind[1]: ind[1] = ind[1]+1
                        return sum(self.data[:,ind[0]:ind[1]],1)
                    else:
                        raise IndexError, "Index list out of range"
                        
    def plot_spec(self,no=None,ind=None,mask=None,ax=None):
        if ax == None:
            pylab.plot(self.lbda,self.spec(no=no,ind=ind,mask=mask))
        else:
            ax.plot(self.lbda,self.spec(no=no,ind=ind,mask=mask))
        pylab.show()
        
    def disp_slice(self,n,coord='w',vmin=None,vmax=None,cmap=pylab.cm.hot):
        slice = self.slice2d(n,coord)
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

    def disp_data(self,vmin=None,vmax=None):
        if vmin == None:
            med = scipy.median(ravel(self.data))
            disp = sqrt(scipy.median((ravel(self.data)-med)**2))
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


#####################  Utility functions  ########################
       
def convert_tab(table,colx,coly,ref_pos):
    tck = scipy.interpolate.splrep(table[1].data.field(colx),\
                                   table[1].data.field(coly),s=0)
    tab = scipy.interpolate.splev(ref_pos,tck)
    return tab

def histogram(data,nbin=None,Min=None,Max=None,bin=None,cumul=False):
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
