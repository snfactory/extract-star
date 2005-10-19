#!/usr/bin/python
import os
import sys
import pylab
import numarray
from numarray import copy
from numarray import linear_algebra as L
import pyfits
import scipy
from scipy import optimize,size,special,Float64
from scipy.special import *



#####################  Data cube explorer  ########################

class graph_interact:

    def __init__(self,fig,cube):
        self.cube = cube
        self.cube_med = scipy.median(ravel(self.cube.data))
        self.cube_mdisp = sqrt(scipy.median((ravel(self.cube.data)-self.cube_med)**2))
        self.fig = fig
        self.axes = fig.get_axes()
        ax0 = self.axes[0]
        ax1 = self.axes[1]
        ax2 = self.axes[2]
        ax3 = self.axes[3]
        
	print('etape 1')
        # Prepare text strings
        self.text = {}
        self.text['nbspec'] = ax3.text(0.02,0.85,'',fontname='Times')
        self.text['sl_stat'] = ax3.text(0.02,0.7,'Current slice statistics')
        self.text['sl_stat'].set_fontweight('Bold')
        self.text['mean'] = ax3.text(0.04,0.6,'')
        self.text['mean'].set_text(    'Mean ....... ')
	self.text['mean1'] = ax3.text(0.25,0.6,'')
        self.text['sigma'] = ax3.text(0.04,0.51,'')
        self.text['sigma'].set_text(   'RMS ........ ')
        self.text['sigma1'] = ax3.text(0.25,0.51,'')
        self.text['median'] = ax3.text(0.04,0.41,'')
        self.text['median'].set_text(  'Median ..... ')
        self.text['median1'] = ax3.text(0.25,0.41,'')
        self.text['med_disp'] = ax3.text(0.04,0.295,'')
        self.text['med_disp'].set_text('Med. disp .. ')
        self.text['med_disp1'] = ax3.text(0.25,0.295,'')
        self.text['wav_range'] = ax3.text(0.04,0.2,'')
        self.text['wav_range'].set_text('Wav. range:  ')
        self.text['wav_range1'] = ax3.text(0.25,0.2,'')
        self.text['current_int'] = ax3.text(0.04,0.1,'')
        self.text['current_int'].set_text('Current int: ')
        self.text['current_int1'] = ax3.text(0.25,0.1,'')
        self.text['sp_stat'] = ax3.text(0.52,0.7,'Current spectrum statistics')
        self.text['sp_stat'].set_fontweight('Bold')
        self.text['mean_sp'] = ax3.text(0.54,0.6,'')
        self.text['mean_sp'].set_text(    'Mean ....... ')
        self.text['mean_sp1'] = ax3.text(0.75,0.6,'')
        self.text['sigma_sp'] = ax3.text(0.54,0.51,'')
        self.text['sigma_sp'].set_text(   'RMS ........ ')
        self.text['sigma_sp1'] = ax3.text(0.75,0.51,'')
        self.text['median_sp'] = ax3.text(0.54,0.41,'')
        self.text['median_sp'].set_text(  'Median ..... ')
        self.text['median_sp1'] = ax3.text(0.75,0.41,'')
        self.text['med_disp_sp'] = ax3.text(0.54,0.295,'')
        self.text['med_disp_sp'].set_text('Med. disp .. ')
        self.text['med_disp_sp1'] = ax3.text(0.75,0.295,'')
	self.text['numslice'] = ax3.text(0.54,0.18,'')
	self.text['numslice'].set_text('Slice :')
	self.text['numslice1'] = ax3.text(0.65,0.17,'')
	self.text['numlens'] = ax3.text(0.54,0.08,'')
	self.text['numlens'].set_text('Lens :')
	self.text['numlens1'] = ax3.text(0.65,0.07,'')

        self.lcut0=-3
        self.hcut0=10
        self.lcut1=-3
        self.hcut1=10

	print('etape 2')
        # Display reconstructed image in the whole spectral range
        self.slice = transpose(cube.slice2d([0,cube.nslice-1],coord='p'))
        self.sl_med = scipy.median(ravel(self.slice))
        self.sl_mdisp = sqrt(scipy.median((ravel(self.slice)-self.sl_med)**2))
        self.sl_mean = scipy.mean(numarray.compress(1-isnan(self.slice),self.slice))
        self.sl_disp = scipy.std(numarray.compress(1-isnan(self.slice),self.slice))
        vmin = self.sl_med + self.lcut0*self.sl_disp
        vmax = self.sl_med + self.hcut0*self.sl_disp
        ax0.imshow(self.slice,interpolation='nearest',aspect='preserve',vmin=vmin,vmax=vmax,cmap=pylab.cm.hot,origin='lower')

	print('etape 3')
        # Display statistics of the reconstructed image
        ind = [0,self.cube.nslice-1]
        self.update_sl_stat(ind)
#        self.update_current_int()
        
	cursor = Cursor(ax1, useblit=False, color='blue', linewidth=1)


	print('etape 4')
        # Display the stacked spectra image
        vmin = self.cube_med + self.lcut1*self.cube_mdisp
        vmax = self.cube_med + self.hcut1*self.cube_mdisp 
        ax1.imshow(transpose(cube.data),interpolation='nearest',vmin=vmin,vmax=vmax,cmap=pylab.cm.hot,origin='lower')

	print('etape 5')
        # Store the x/y limits of both images
        ax0_xlim = ax0.get_xlim()
        ax0_ylim = ax0.get_ylim()
        ax1_xlim = ax1.get_xlim()
        ax1_ylim = ax1.get_ylim() 
        self.minx0,self.maxx0 = ax0.get_xlim()
        self.miny0,self.maxy0 = ax0.get_ylim()
        self.minx1,self.maxx1 = ax1.get_xlim()
        self.miny1,self.maxy1 = ax1.get_ylim()

	print('etape 8')
        # Initialize the dot on the slice window showing the selected spectrum
        self.dot, = ax0.plot((self.minx1,self.minx1),(self.miny1,self.miny1),'wo')
	
	print('etape 9')
        # Initialize the transparent circle showing the aperture selection
        self.center = [0,0]
        self.radius = 0
        ax0.add_patch(pylab.Circle(self.center,radius=1,alpha=0.5,fc='w'))
        self.circle = ax0.patches[0]
        self.unit_circ_verts = []
        self.unit_verts = [v for v in self.circle.verts]
                      
	print('etape 10')
        # Dirty trick necessary, otherwise, the axes limits are changed I don't know why...
#        ax0_xlim = ax0.get_xlim()
#        ax0_ylim = ax0.get_ylim()
#	self.minx0,self.maxx0 = ax0.get_xlim()
#        self.miny0,self.maxy0 = ax0.get_ylim()
        ax0.set_xlim(ax0_xlim)
        ax0.set_ylim(ax0_ylim)
        ax1.set_xlim(ax1_xlim)
        ax1.set_ylim(ax1_ylim)
        self.dot.set_visible(False)
        self.circle.set_visible(False)

	print('etape 11')
        # Initialize the begin selection flags. Any 1/2 button press will set the corresponding window flag      
        # to false and any 1/2 button release will reset it to True
        self.begin_sel_ax0 = True
        self.begin_sel_ax1 = True
        self.erase_flag = True
	
	pylab.draw()


#	def onselectvert(ymin, ymax):
#		ym = int(ymin)
#		yM = int(ymax)
#		self.text['numlens1'].set_text('[%3.0d:%3.0d]'%(ym,yM))
#		#self.lx.set_data( (self.minx1, self.maxx1), (ym,ym) )
#                #pylab.draw()
#
#        
#	span = VerticalSpanSelector(ax1, onselectvert, useblit=False,rectprops=dict(alpha=0.5, facecolor='green') )
#	pylab.show


	def onselecthori(xmin, xmax):
		xm = int(xmin)
		xM = int(xmax)
		self.text['numslice1'].set_text('[%4.0d:%4.0d]'%(xm,xM))
		pylab.draw()

	span1 = HorizontalSpanSelector(ax1, onselecthori, useblit=False,rectprops=dict(alpha=0.5, facecolor='blue') )
	pylab.show
	
    def update_slice(self,ind=None):
        if ind != None:
            self.slice = transpose(self.cube.slice2d(ind,coord='p'))
            self.sl_med = scipy.median(ravel(self.slice))
            self.sl_mdisp = sqrt(scipy.median((ravel(self.slice)-self.sl_med)**2))
            self.sl_mean = scipy.mean(numarray.compress(1-isnan(self.slice),self.slice))
            self.sl_disp = scipy.std(numarray.compress(1-isnan(self.slice),self.slice))
        if self.hcut0 < self.lcut0: self.lcut0 = self.hcut0
        vmin = self.sl_med + self.lcut0*self.sl_mdisp
        vmax = self.sl_med + self.hcut0*self.sl_mdisp
        self.axes[0].images[0].set_array(self.slice)
        self.axes[0].images[0].set_clim(vmin,vmax)
        self.update_sl_stat(ind)
        pylab.show()

    def update_stacked_spec(self):
        if self.hcut1 < self.lcut1: self.lcut1 = self.hcut1
        print self.lcut1,self.hcut1
        vmin = self.cube_med + self.lcut1*self.cube_mdisp
        vmax = self.cube_med + self.hcut1*self.cube_mdisp
        self.axes[1].images[0].set_clim(vmin,vmax)
        pylab.show()
        
    def update_spec(self,ind):
        if self.erase_flag:
            self.axes[2].cla()
        self.cube.plot_spec(ind=ind,ax=self.axes[2])
        self.erase_flag = True
        self.update_sp_stat()

    def comp_new_verts(self,center,radius):
        self.circle.verts = [((v[0]-center[0])*radius+center[0],\
                              (v[1]-center[1])*radius+center[1]) for v in self.unit_verts]
        
    def integ_aperture(self):
        if self.erase_flag:
            self.axes[2].cla()
        r = sqrt((self.cube.i-self.center[0]+0.5)**2 + (self.cube.j-self.center[1]+0.5)**2)
        self.cube.plot_spec(mask=(r<=self.radius),ax=self.axes[2])
        self.erase_flag = True
        self.update_sp_stat()
        
    def print_nbspec(self,n=None):
        if n == None:
            r = sqrt((self.cube.i-self.center[0]+0.5)**2 + (self.cube.j-self.center[1]+0.5)**2)
            nbspec = len(compress(r<=self.radius,r))
        else:
            nbspec = n
        if nbspec <= 1: 
            self.text['nbspec'].set_text('%i spectrum selected'%nbspec)
        else:
            self.text['nbspec'].set_text('%i spectra selected'%nbspec)
        pylab.show()

    def update_sl_stat(self,ind):
        lbda = numarray.array(self.cube.lbda)[ind]
        self.text['mean1'].set_text('%8.2f'%self.sl_mean)
        self.text['sigma1'].set_text('%8.2f'%self.sl_disp)
        self.text['median1'].set_text('%8.2f'%self.sl_med)
        self.text['med_disp1'].set_text('%8.2f'%self.sl_mdisp)
        self.text['wav_range1'].set_text('[%6.2f:%6.2f]'%(lbda[0],lbda[1]))

    def update_sp_stat(self):
        n = len(self.axes[2].lines)-1
        data = self.axes[2].lines[n].get_ydata()
        med = scipy.median(data)
        Mean = scipy.mean(data)
        sigma = scipy.std(data)
        disp = sqrt(scipy.median((data-med)**2))
        self.text['mean_sp1'].set_text('%8.2f'%Mean)
        self.text['sigma_sp1'].set_text('%8.2f'%sigma)
        self.text['median_sp1'].set_text('%8.2f'%med)
        self.text['med_disp_sp1'].set_text('%8.2f'%disp)
        
    def update_current_int(self,ind=None):
        if ind != None:
            self.text['current_int1'].set_text('%8.2f'%(self.slice[ind[1],ind[0]]))
        pylab.show()

    def reset_draw(self):
	self.axes[1].patches[0].xy = [(self.minx1,self.minx1),(self.minx1,self.minx1),(self.minx1,self.minx1),(self.minx1,self.minx1)]
	self.unit_verts = [(v[0] - self.center[0],v[1] - self.center[1]) for v in self.unit_verts]
	self.circle.set_visible(False)

   
class explore_cube:

    def __init__(self,cube,cmap=pylab.cm.hot):
        fig = pylab.figure(figsize=(12,9))
        q = fig.get_figheight()/fig.get_figwidth()
        ax0 = fig.add_axes((0.05,0.3,0.65*q,0.65))
        ax1 = fig.add_axes((0.05+0.7*q,0.3,0.9-0.65*q,0.65))
        ax2 = fig.add_axes((0.05+0.7*q,0.05,0.9-0.65*q,0.2))
        ax3 = fig.add_axes((0.05,0.05,0.65*q,0.2))
        ax3.set_xticks([])
        ax3.set_yticks([])
#        ax3.set_axis_bgcolor('#e7e0c8')
	
        print('juste avant graph')
        graph = graph_interact(fig,cube)
        #pylab.connect('button_press_event', graph.on_click)
        #pylab.connect('motion_notify_event', graph.on_move)
        #pylab.connect('button_release_event', graph.on_release)
        #self.spec_axe = ax2

    def get_spec(self,n):
        l = self.spec_axe.lines[n]
        spec = spectrum(x=l.get_xdata(),data=l.get_ydata())
        return spec



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
