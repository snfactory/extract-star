######################################################################
## Filename:      pySNIFS_fit.py
## Version:       $Revision$
## Description:   
## Author:        Emmanuel Pecontal
## Author:        $Author$
## $Id$
######################################################################

from pySNIFS import *
#import numarray
import scipy
import numpy
import copy
from numpy import *

__author__ = '$Author$'
__version__ = '$Revision$'
__docformat__ = "epytext en"

#####################  Fitting Package  ##########################


#######################################################
#                   1D Functions                      #
#######################################################

class gaus1D:
    """
    1D gaussian function used by the L{model} class
    """
    def __init__(self,cube=None):
        """
        Initiating the class.
        @param cube: Input cube. This is a L{SNIFS_cube} object with only one spaxel. It is built from a
            L{spectrum} object by the L{model} class.
        """
        self.npar_ind = 0
        self.npar_cor = 3
        self.npar = self.npar_ind*cube.nslice + self.npar_cor
        #self.l = zeros(shape(transpose(cube.data)),Float64)
        self.name = 'gaus1D'
        #self.l[:][:] = cube.lbda
        self.l = reshape(cube.lbda,shape(cube.data))
            
    def comp(self,param):
        """
        Compute the gaussian function.
        @param param: Input parameters of the gaussian. A list of three number: [xc,sigma,I] 
        """
        self.param = param
        val = param[2] * scipy.exp(-0.5*(((self.l-param[0])/param[1])**2))
        return val

    def deriv(self,param):
        """
        Compute the derivative of the gaussian function with respect to the parameters.
        @param param: Input parameters of the gaussian. A list of three number: [xc,sigma,I] 
        """
        grad = scipy.zeros((self.npar_cor+self.npar_ind,)+shape(self.l),'d')
        expo = scipy.exp(-0.5*(((self.l-param[0])/param[1])**2))
        val = expo * param[2] * (self.l-param[0])/param[1]**2
        grad[0] = val
        val = expo * param[2] * (self.l-param[0])**2 / param[1]**3
        grad[1] = val
        grad[2] = expo
        return grad
    
class poly1D:
    """
    1D polynomial used by the L{model} class
    """
    def __init__(self,deg=None,cube=None):
        """
        Initiating the class.
        @param deg: Degree of the polynomial
        @param cube: Input cube. This is a L{SNIFS_cube} object with only one spaxel. It is built from a
            L{spectrum} object by the L{model} class.
        """
        self.deg = deg
        self.npar_ind = 0
        self.npar_cor = deg+1
        self.npar = self.npar_ind*cube.nslice + self.npar_cor
        self.name = 'poly1D'
        self.l = reshape(cube.lbda,shape(cube.data))
        
    def comp(self,param):
        """
        Compute the polynomial.
        @param param: Input parameters of the polynomial. A list of deg+1 numbers.
        """
        val = scipy.poly1d(param[::-1])(self.l)
        return val
    
    def deriv(self,param):
        """
        Compute the derivative of the polynomial with respect to its parameters.
        @param param: Input parameters of the polynomial. A list of deg+1 numbers.
        """
        grad = [(self.l)**i for i in arange(self.npar_cor)]
        grad = scipy.array(grad)
        #grad = scipy.zeros((self.npar_cor+self.npar_ind,)+shape(self.l),'d')
        #for i in arange(self.npar_cor):
        #    grad[i] = l**i
        return grad

#######################################################
#                   2D Functions                      #
#######################################################

class gaus2D:
    """
    2D gaussian function used by the L{model} class
    """
    def __init__(self,cube=None):
        """
        Initiating the class.
        @param cube: Input cube. This is a L{SNIFS_cube} object with only one wavelength. It is built from a
            spectrum object by the L{model} class.
        """ 
        self.nslice = cube.nslice
        self.npar_ind = 5
        self.npar_cor = 0
        self.npar = self.npar_ind*self.nslice + self.npar_cor
        self.name = 'gaus2D'
        self.x = zeros(shape(cube.data),'d')
        self.y = zeros(shape(cube.data),'d')
        self.x[:][:] = cube.x
        self.y[:][:] = cube.y

    def comp(self,param):
        """
        Compute the gaussian function.
        @param param: Input parameters of the gaussian. A list of five number: [xc,yc,sx,sy,I] 
        """
        self.param = param
        tab_param_ind = transpose(reshape(param[self.npar_cor:],(self.npar_ind,self.nslice)))
        val = tab_param_ind[:,4:5] * scipy.exp(-0.5*(((self.x-tab_param_ind[:,0:1])/tab_param_ind[:,2:3])**2 + ((self.y-tab_param_ind[:,1:2])/tab_param_ind[:,3:4])**2))
        return val
    
    def deriv(self,param):
        """
        Compute the derivative of the gaussian function with respect to the parameters.
        @param param: Input parameters of the gaussian. A list of five number: [xc,yc,sx,sy,I]
        """
        self.param = param
        tab_param_ind = transpose(reshape(param,(self.npar_ind,self.nslice)))
        grad = scipy.ones((self.npar_ind,)+shape((self.x)),'d')

        expo = scipy.exp(-0.5*(((self.x-tab_param_ind[:,0:1])/tab_param_ind[:,2:3])**2 + ((self.y-tab_param_ind[:,1:2])/tab_param_ind[:,3:4])**2))
        val = expo * tab_param_ind[:,4:5] * (self.x-tab_param_ind[:,0:1])/(tab_param_ind[:,2:3])**2
        grad[0] = val
        val = expo * tab_param_ind[:,4:5] * (self.y-tab_param_ind[:,1:2])/(tab_param_ind[:,3:4])**2
        grad[1] = val
        val = expo * tab_param_ind[:,4:5] * (self.x-tab_param_ind[:,0:1])**2 / (tab_param_ind[:,2:3])**3
        grad[2] = val
        val = expo * tab_param_ind[:,4:5] * (self.y-tab_param_ind[:,1:2])**2 / (tab_param_ind[:,3:4])**3
        grad[3] = val
        grad[4] = expo
        return grad


class gaus2D_integ:
    """
    2D gaussian function with integration in the pixel, used by the L{model} class
    """
    def __init__(self,pix=None,cube=None):
        """
        Initiating the class.
        @param pix: Size of the pixel in spatial units of the cube.
        @param cube: Input cube. This is a L{SNIFS_cube} object with only one wavelength. It is built from an
            L{image_array} object by the class model.
        """ 
        self.nslice = cube.nslice
        self.pix = pix
        self.npar_ind = 5
        self.npar_cor = 0
        self.npar = self.npar_ind*self.nslice + self.npar_cor
        self.name = 'gauss2D_integ'
        self.x = zeros(shape(cube.data),'d')
        self.y = zeros(shape(cube.data),'d')
        self.x[:][:] = cube.x
        self.y[:][:] = cube.y
        
    def comp(self,param):
        """
        Compute the gaussian function.
        @param param: Input parameters of the gaussian. A list of five number: [xc,yc,sx,sy,I] 
        """
        self.param = param
        tab_param_ind = transpose(reshape(param,(self.npar_ind,self.nslice)))
        sq2 = sqrt(2)
        xpcn = (self.x + self.pix/2 - tab_param_ind[:,0:1])/(sq2*tab_param_ind[:,2:3])
        xmcn = (self.x - self.pix/2 - tab_param_ind[:,0:1])/(sq2*tab_param_ind[:,2:3])
        ypcn = (self.y + self.pix/2 - tab_param_ind[:,1:2])/(sq2*tab_param_ind[:,3:4])
        ymcn = (self.y - self.pix/2 - tab_param_ind[:,1:2])/(sq2*tab_param_ind[:,3:4])
        val = pi*tab_param_ind[:,2:3]*tab_param_ind[:,3:4]*tab_param_ind[:,4:5]*(erf(xpcn)-erf(xmcn))*(erf(ypcn)-erf(ymcn))/2
        return val
    
    def deriv(self,param):
        """
        Compute the derivative of the gaussian function with respect to the parameters.
        @param param: Input parameters of the gaussian. A list of five number: [xc,yc,sx,sy,I]
        """
        self.param = param
        tab_param_ind = transpose(reshape(param,(self.npar_ind,self.nslice)))
        grad = scipy.ones((self.npar_ind,)+shape((self.x)),'d')
        sqpi_2 = sqrt(pi/2.)
        sq2 = sqrt(2.)
        xpc = (self.x + self.pix/2 - tab_param_ind[:,0:1])
        xmc = (self.x - self.pix/2 - tab_param_ind[:,0:1])
        ypc = (self.y + self.pix/2 - tab_param_ind[:,1:2])
        ymc = (self.y - self.pix/2 - tab_param_ind[:,1:2])
        xpcn = xpc/(sq2*tab_param_ind[:,2:3])
        xmcn = xmc/(sq2*tab_param_ind[:,2:3])
        ypcn = ypc/(sq2*tab_param_ind[:,3:4])
        ymcn = ymc/(sq2*tab_param_ind[:,3:4])
        gxp = scipy.exp(-xpcn**2)
        gxm = scipy.exp(-xmcn**2)
        gyp = scipy.exp(-ypcn**2)
        gym = scipy.exp(-ymcn**2)
        Ix = sqpi_2*tab_param_ind[:,2:3]*(erf(xpcn)-erf(xmcn))
        Iy = sqpi_2*tab_param_ind[:,3:4]*(erf(ypcn)-erf(ymcn))
        grad[0] = tab_param_ind[:,4:5]*Iy*(gxm-gxp)
        grad[1] = tab_param_ind[:,4:5]*Ix*(gym-gyp)
        grad[2] = tab_param_ind[:,4:5]*Iy*(Ix-(xpc*gxp-xmc*gxm))/tab_param_ind[:,2:3]
        grad[3] = tab_param_ind[:,4:5]*Ix*(Iy-(ypc*gyp-ymc*gym))/tab_param_ind[:,3:4]
        grad[4] = Ix*Iy
        return grad

class poly2D:
    """
    2D polynomial used by the L{model} class
    """
    def __init__(self,deg=None,cube=None):
        """
        Initiating the class.
        @param deg: Degree of the polynomial
        @param cube: Input cube. This is a L{SNIFS_cube} object with only one wavelength. It is built from an
            L{image_array} object by the L{model} class.
        """
        self.nslice = cube.nslice
        self.deg = deg
        self.npar_ind = int((deg+1)*(deg+2)/2)
        self.npar_cor = 0
        self.npar = self.npar_ind*self.nslice + self.npar_cor
        self.name = 'poly2D'
        self.x = zeros(shape(cube.data),'d')
        self.y = zeros(shape(cube.data),'d')
        self.x[:][:] = cube.x
        self.y[:][:] = cube.y
        
    def comp(self,param):
        """
        Compute the polynomial.
        @param param: Input parameters of the polynomial. A list of (deg+1)*(deg+2)/2 numbers.
        """
        self.param = param
        tab_param_ind = transpose(reshape(param,(self.npar_ind,self.nslice)))
        n = 0
        val = scipy.ones(shape(self.x),'d') * tab_param_ind[:,0:1]
        #print str(self.param[0])
        for d in arange(self.deg)+1:
            for j in arange(d+1):
                #print str(self.param[n+1])+'x^'+str(d-j)+'y^'+str(j)
                val = val + tab_param_ind[:,n+1:n+2] * self.x**(d-j) * self.y**(j)
                n=n+1
        return val
    
    def deriv(self,param):
        """
        Compute the derivative of the polynomial with respect to its parameters.
        @param param: Input parameters of the polynomial. A list of (deg+1)*(deg+2)/2 numbers.
        """
        self.param = param
        tab_param_ind = transpose(reshape(param,(self.npar_ind,self.nslice)))
        n = 0
        grad = scipy.ones((self.npar_ind,)+shape((self.x)),'d')
        #print str(self.param[0])

        for d in arange(self.deg)+1:
            for j in arange(d+1):
                #print str(self.param[n+1])+'x^'+str(d-j)+'y^'+str(j)
                grad[n+1] = self.x**(d-j) * self.y**(j)
                n=n+1
        return grad
    
#######################################################
#                   3D Functions                      #
#######################################################

class SNIFS_psf_3D:
    """
    SNIFS PSF 3D function used by the L{model} class.
    """
    def __init__(self,intpar=[None,None],cube=None):
        """
        Initiating the class.
        @param intpar: Internal parameters (pixel size in cube spatial unit and reference wavelength). A
            list of two numbers.
        @param cube: Input cube. This is a L{SNIFS_cube} object.
        """
        #self.cube = cube
        self.pix = intpar[0]
        self.lbda_ref = intpar[1]
        self.npar_ind = 1
        self.npar_cor = 11
        self.npar = self.npar_ind*cube.nslice + self.npar_cor
        self.name = 'SNIFS_psf_3D'
        self.x = zeros((cube.nslice,cube.nlens),'d')
        self.y = zeros((cube.nslice,cube.nlens),'d')
        self.l = zeros(shape(transpose(cube.data)),'d')
        self.x[:][:] = cube.x
        self.y[:][:] = cube.y
        self.l[:][:] = cube.lbda
        self.l = transpose(self.l)
        self.n_ref = 1e-6*(64.328 + 29498.1/(146.-1./(self.lbda_ref*1e-4)**2) + 255.4/(41.-1./(self.lbda_ref*1e-4)**2)) + 1.
        self.ADR_coef = 206265*(1e-6*(64.328 + 29498.1/(146.-1./(self.l*1e-4)**2) + 255.4/(41.-1./(self.l*1e-4)**2)) + 1. - self.n_ref)
        
    def comp(self,param):
        """
        Compute the function.
        @param param: Input parameters of the polynomial. A list of numbers:
                - C{param[0:10]}: The 11 parameters of the PSF shape
                     - C{param[0]}: Atmospheric dispersion power
                     - C{param[1]}: Atmospheric dispersion position angle
                     - C{param[2]}: X center at the reference wevelength
                     - C{param[3]}: Y center at the reference wevelength
                     - C{param[4]}: PSF core dispersion at the reference wevelength
                     - C{param[5]}: PSF wing dispersion at the reference wevelength
                     - C{param[6]}: exponent of the PSF width vs wavelength relation
                     - C{param[7]}: Ratio between the core and the wing gaussians
                     - C{param[8]}: Bluring gaussian kernel dispersion
                     - C{param[9]}: Bluring gaussian kernel axis ratio
                     - C{param[10]}: Bluring gaussian kernel position angle
                - C{param[11:]} : The intensity parameters (one for each slice in the cube.
        """
        self.param = param
        x0 = self.param[0]*self.ADR_coef*cos(self.param[1]) + self.param[2]
        y0 = self.param[0]*self.ADR_coef*sin(self.param[1]) + self.param[3]
        #x0 = self.param[0]*self.l + self.param[1]
        #y0 = self.param[2]*self.l + self.param[3]
        sigma_c = self.param[4]*(self.l/self.lbda_ref)**self.param[5]
        sigma_w = self.param[6]*sigma_c
        eps = self.param[7]
        sigma_k = self.param[8]
        q_k = self.param[9]
        theta = self.param[10]
        sig_c_x = sqrt(sigma_c**2 + sigma_k**2)
        sig_c_y = sqrt(sigma_c**2 + (sigma_k*q_k)**2)
        sig_w_x = sqrt(sigma_w**2 + sigma_k**2)
        sig_w_y = sqrt(sigma_w**2 + (sigma_k*q_k)**2)
        sq2 = sqrt(2.)
        cost = cos(theta)
        sint = sin(theta)
        xr = (self.x-x0)*cost - (self.y-y0)*sint
        yr = (self.x-x0)*sint + (self.y-y0)*cost
        xrcp = (xr + self.pix/2)/(sq2*sig_c_x)
        xrcm = (xr - self.pix/2)/(sq2*sig_c_x)
        yrcp = (yr + self.pix/2)/(sq2*sig_c_y)
        yrcm = (yr - self.pix/2)/(sq2*sig_c_y)
        Icx = (erf(xrcp)-erf(xrcm))/2.
        Icy = (erf(yrcp)-erf(yrcm))/2.
        xrwp = (xr + self.pix/2)/(sq2*sig_w_x)
        xrwm = (xr - self.pix/2)/(sq2*sig_w_x)
        yrwp = (yr + self.pix/2)/(sq2*sig_w_y)
        yrwm = (yr - self.pix/2)/(sq2*sig_w_y)
        Iwx = (erf(xrwp)-erf(xrwm))/2.
        Iwy = (erf(yrwp)-erf(yrwm))/2.
        #return sigma_c
        return reshape(param[11:],(len(param[11:]),1))*(Icx*Icy+eps*Iwx*Iwy)
    
    def deriv(self,param):
        """
        Compute the derivative of the function with respect to its parameters.
        @param param: Input parameters of the polynomial. A list numbers (see L{SNIFS_psf_3D.comp}).
        """
        self.param = param    
        grad = scipy.zeros((self.npar_cor+self.npar_ind,)+shape(self.x),'d')
        
        costp = cos(self.param[1])
        sintp = sin(self.param[1])
        lbda_rel = self.l / self.lbda_ref
        x0 = self.param[0]*self.ADR_coef*costp + self.param[2]
        y0 = self.param[0]*self.ADR_coef*sintp + self.param[3]
        #x0 = self.param[0]*self.l + self.param[1]
        #y0 = self.param[2]*self.l + self.param[3]
        gamma = self.param[5]
        lbda_rel_exp_gamma = lbda_rel ** gamma
        sigma_c = self.param[4]*lbda_rel_exp_gamma
        q = self.param[6]
        sigma_w = self.param[6]*sigma_c
        eps = self.param[7]
        sigma_k = self.param[8]
        q_k = self.param[9]
        theta = self.param[10]
        sig_cx = sqrt(sigma_c**2 + sigma_k**2)
        sig_cy = sqrt(sigma_c**2 + (sigma_k*q_k)**2)
        sig_wx = sqrt(sigma_w**2 + sigma_k**2)
        sig_wy = sqrt(sigma_w**2 + (sigma_k*q_k)**2)
        sqpi_2 = sqrt(pi/2.)
        sq2 = sqrt(2.)
        cost = cos(theta)
        sint = sin(theta)
        xr = (self.x-x0)*cost - (self.y-y0)*sint
        yr = (self.x-x0)*sint + (self.y-y0)*cost
        xrcp = (xr + self.pix/2)/(sq2*sig_cx)
        xrcm = (xr - self.pix/2)/(sq2*sig_cx)
        yrcp = (yr + self.pix/2)/(sq2*sig_cy)
        yrcm = (yr - self.pix/2)/(sq2*sig_cy)
        xrwp = (xr + self.pix/2)/(sq2*sig_wx)
        xrwm = (xr - self.pix/2)/(sq2*sig_wx)
        yrwp = (yr + self.pix/2)/(sq2*sig_wy)
        yrwm = (yr - self.pix/2)/(sq2*sig_wy)

    
        gxrcp = scipy.exp(-xrcp**2)
        gxrcm = scipy.exp(-xrcm**2)
        gyrcp = scipy.exp(-yrcp**2)
        gyrcm = scipy.exp(-yrcm**2)
        gxrwp = scipy.exp(-xrwp**2)
        gxrwm = scipy.exp(-xrwm**2)
        gyrwp = scipy.exp(-yrwp**2)
        gyrwm = scipy.exp(-yrwm**2)
        Icx = (erf(xrcp)-erf(xrcm))/2.
        Icy = (erf(yrcp)-erf(yrcm))/2.
        Iwx = (erf(xrwp)-erf(xrwm))/2.
        Iwy = (erf(yrwp)-erf(yrwm))/2.
        
        grad[2] = Icy*cost*(gxrcm-gxrcp)/sqrt(2*pi)/sig_cx + Icx*sint*(gyrcm-gyrcp)/sqrt(2*pi)/sig_cy + \
                  eps*(Iwy*cost*(gxrwm-gxrwp)/sqrt(2*pi)/sig_wx + Iwx*sint*(gyrwm-gyrwp)/sqrt(2*pi)/sig_wy)
        grad[3] = Icy*sint*(gxrcp-gxrcm)/sqrt(2*pi)/sig_cx + Icx*cost*(gyrcm-gyrcp)/sqrt(2*pi)/sig_cy + \
                  eps*(Iwy*sint*(gxrwp-gxrwm)/sqrt(2*pi)/sig_wx + Iwx*cost*(gyrwm-gyrwp)/sqrt(2*pi)/sig_wy)
        grad[0] = self.ADR_coef*(costp*grad[2] + sintp*grad[3])
        grad[1] = self.param[0]*self.ADR_coef*(costp*grad[3]-sintp*grad[2])
        der_cx = Icy*(gxrcm*xrcm-gxrcp*xrcp)/sqrt(pi)/sig_cx**2
        der_cy = Icx*(gyrcm*yrcm-gyrcp*yrcp)/sqrt(pi)/sig_cy**2
        der_wx = Iwy*(gxrwm*xrwm-gxrwp*xrwp)/sqrt(pi)/sig_wx**2
        der_wy = Iwx*(gyrwm*yrwm-gyrwp*yrwp)/sqrt(pi)/sig_wy**2
        der_sigc = sigma_c*(der_cx + der_cy + eps * q**2 * (der_wx + der_wy))
        grad[4] = lbda_rel_exp_gamma * der_sigc
        grad[5] = grad[4] * self.param[4] * log(lbda_rel)
        grad[6] = eps*sigma_c**2*q*(der_wx + der_wy)
        grad[7] = Iwx*Iwy
        grad[8] = sigma_k*(der_cx+q_k**2*der_cy \
                                            + eps*(der_wx+q_k**2*der_wy))
        grad[9] = q_k*sigma_k**2 *(der_cy+eps*der_wy)
        grad[10] = (Icy*yr/sqrt(2*pi)/sig_cx)*(gxrcm-gxrcp) + (Icx*xr/sqrt(2*pi)/sig_cy)*(gyrcp-gyrcm)\
                  + eps*((Iwy*yr/sqrt(2*pi)/sig_wx)*(gxrwm-gxrwp) + (Iwx*xr/sqrt(2*pi)/sig_wy)*(gyrwp-gyrwm))
        grad[11] = Icx*Icy + eps*Iwx*Iwy
        grad[0:11] = grad[0:11] * reshape(param[11:],(1,len(param[11:]),1))
        
        return grad

  
#######################################################
#                  Model fitting                      #
#######################################################	 
 
class model:
    """
    Model fiting class
    """
    def __init__(self,func=['gaus2D'],data=None,param=None,bounds=None,myfunc=None):
        self.fitpar = None
        
        if param == None:
            raise ValueError, "A set of model parameters values must be provided."
            
        if (not isinstance(data,SNIFS_cube)) and (not isinstance(data,spectrum)) and (not isinstance(data,image_array)):
            raise TypeError, "data array must be a SNIFS_cube, image_array or spectrum object."
        if isinstance(data,SNIFS_cube):
            self.model_1D = False
            self.model_2D = False
            self.model_3D = True
            self.data = data
        elif isinstance(data,image_array):
            self.model_1D = False
            self.model_2D = True
            self.model_3D = False
            self.data = SNIFS_cube()
            self.data.data = scipy.reshape(scipy.ravel(data.data),(1,len(scipy.ravel(data.data))))
            self.data.lbda = scipy.array([0])
            self.data.nslice = 1
            self.data.nlens = data.nx * data.ny
            self.data.i = scipy.ravel(indices((data.nx,data.ny))[0]) 
            self.data.j = scipy.ravel(indices((data.nx,data.ny))[1]) 
            self.data.x = self.data.i*data.stepx+data.startx
            self.data.y = self.data.j*data.stepy+data.starty
        else:
            self.model_1D = True
            self.model_2D = False
            self.model_3D = False
            self.data = SNIFS_cube()
            self.data.data = scipy.reshape(data.data[data.index_list],(len(data.index_list),1))
            self.data.lbda = data.x[data.index_list]
            self.data.nslice = len(data.index_list)
            self.data.nlens = 1
            self.data.x = None
            self.data.y = None
            self.data.i = None
            self.data.j = None
            self.data.no = None
        
        func_dict = {}
        func_dict['gaus1D']=gaus1D
        func_dict['poly1D']=poly1D
        func_dict['gaus2D']=gaus2D
        func_dict['poly2D']=poly2D
        func_dict['gaus2D_integ']=gaus2D_integ
        func_dict['SNIFS_psf_3D']=SNIFS_psf_3D
        
        # We add in the available function dictionary the user's ones
        if (myfunc) != None:
            if not isinstance(myfunc,dict):
                raise ValueError,'The users functions must be provided as a dictionary {\'name1\':func1,\
                \'name2\':func2...}'
            func_dict.update(myfunc)

        if self.model_1D:
            avail_func = ['gaus1D','poly1D']
            if myfunc != None:
                avail_func = avail_func+myfunc.keys()
                
        else:
            avail_func = ['SNIFS_psf_3D','gaus2D','gaus2D_integ','poly2D']
            if myfunc != None:
                avail_func = avail_func+myfunc.keys()
                
        if data.var == None:
            #self.weight = SNIFS_cube()
            self.weight = scipy.ones(shape(self.data.data),'d')
            #self.weight.ones_from(self.data)
        else:
            if self.model_3D:
                #self.weight = SNIFS_cube()
                #self.weight.data = scipy.array(where(data.var!=0,1./abs(data.var),0.),'d')
                self.weight = scipy.array(where(data.var!=0,1./abs(data.var),0.),'d')
            elif self.model_2D:# TODO: Implement the variance
                self.data.var = scipy.reshape(scipy.ravel(data.var),(1,len(scipy.ravel(data.var))))
                self.weight = scipy.array(where(self.data.var!=0,1./abs(self.data.var),0.),'d')
                
            elif self.model_1D:
                #self.weight = SNIFS_cube()
                weight_val = scipy.array(where(data.var[data.index_list]!=0,1./abs(data.var[data.index_list]),0.),'d')
                self.weight = scipy.reshape(weight_val,(len(data.index_list),1))
                #self.weight.lbda = data.x
                #self.weight.nslice = data.len
                #self.weight.nlens = 1
                #self.weight.x = None
                #self.weight.y = None
                #self.weight.i = None
                #self.weight.j = None
                #self.weight.no = None
                
        self.khi2 = None

        ######### Function list analysis #########
        if not isinstance(func,list):
            raise TypeError, "Parameter func should be a list."

        self.func = []
        for f in func:
            fstring = f.split(';')[0]
            if fstring not in avail_func:
                if self.model_1D:
                    raise ValueError, "Function %s not available for 1D models." % fstring
                else:
                    raise ValueError, "Function %s not available for 2D/3D models." % fstring
            #Check if the function has internal parameters
            if len(f.split(';')) == 1:
                self.func.append(func_dict[fstring](self.data))
            else:
                inpar_string = f.split(';')[1]
                internal_param = [float(inpar_string.split(',')[i]) for i in arange(len(inpar_string.split(',')))]
                if len(internal_param) == 1:
                    internal_param = internal_param[0]
                self.func.append(func_dict[fstring](internal_param,self.data))

        ######### Parameter list analysis #########
        if not isinstance(param,list):
            raise TypeError, "param must be a list of arrays."
        if len(param) != len(func):
            raise ValueError, "param list and func list must have the same length."
        for i,f in enumerate(self.func):
            if f.npar != len(param[i]):
                raise ValueError, "Function %s must have %d parameters, %d given." % \
                      (f.name, f.npar, len(param[i]))
        self.param = param
        nparam = [len(param[i]) for i in arange(len(param))]
        #self.flatparam = numarray.zeros(sum(nparam),'d')]
        self.flatparam = scipy.zeros(sum(nparam),'d')
        self.nparam = sum(nparam)
        n = 0
        for i in arange(len(param)):
            self.flatparam[n:n+nparam[i]] = param[i]
            n = n + nparam[i]

        ######### Bounds list analysis #########
        if bounds == None:
            self.bounds = None
        else:
            if len(param) != len(bounds):
                raise ValueError, "There must be one bound pairs for each variable."
            self.bounds = []
            n = 0
            for i in arange(len(param)):
                if len(param[i]) != len(bounds[i]):
                    raise ValueError, "Function #%d has not the same bound pairs and variables number." % i
                for j in arange(len(param[i])):
                    self.bounds.append(bounds[i][j])
                n = n + nparam[i]
 
    def new_param(self,param=None):
        """ Store new parameters for the model evaluation. """
        nparam = [len(param[i]) for i in arange(len(param))]
        if not isinstance(param,list):
            raise TypeError, "param must be a list of arrays."
        if scipy.size(param) != scipy.size(self.param) or len(param) != len(self.param):
            raise ValueError, "param has not the correct size."
        self.param = param
        self.flatparam = scipy.zeros(sum(nparam),'d')
        n = 0
        for i in range(len(param)):
            self.flatparam[n:n+nparam[i]] = param[i]
            n = n + nparam[i]
   
    def eval(self):
        """ Evaluate the model at the current parameters stored in the field flatparam."""
        val = scipy.zeros((self.data.nslice,self.data.nlens),'d')
        i = 0
        for f in self.func:
            val = val + f.comp(self.flatparam[i:i+f.npar])
            i=i+f.npar
        return val

    def evalfit(self):
        """ Evaluate the model at the last fitted parameters stored in the field fitpar."""
        if self.fitpar == None:
            raise ValueError, "No fit parameter to evaluate model."
        val = scipy.zeros((self.data.nslice,self.data.nlens),'d')
        i = 0
        for f in self.func:
            val = val + f.comp(self.fitpar[i:i+f.npar])
            i=i+f.npar
        return val

    def res_eval(self,param=None):
        """ Evaluate the residuals at the current parameters stored in the field flatparam."""
        if param == None:
            #param = numarray.copy.deepcopy(self.flatparam)
            param = numpy.copy.deepcopy(self.flatparam)
        val = scipy.zeros((self.data.nslice,self.data.nlens),'d')
        i = 0
        for f in self.func:
            val = val + f.comp(param[i:i+f.npar])
            i = i + f.npar
        val = self.data.data - val
        return val

    def res_evalfit(self):
        """ Evaluate the residuals at the last fitted parameters stored in the field fitpar."""
        if self.fitpar == None:
            raise ValueError, "No parameter to estimate function value."
        val = scipy.zeros((self.data.nslice,self.data.nlens),'d')
        i = 0
        for f in self.func: 
            val =val+ f.comp(self.fitpar[i:i+f.npar])
            i=i+f.npar
        val = self.data.data - val
        return val

    def objfun(self,param=None):
        """ Compute the objective function to be minimized: Sum(weight*(data-model)^2) at the given parameters values."""
        #val = sum(self.res_eval(param=param)**2 * self.weight.data,1)
        val = sum(self.res_eval(param=param)**2 * self.weight,1)
        val = sum(val)
        return val
        
    def objgrad(self,param=None):
        """ Compute the gradient of the objective function at the given parameters value."""
        if param == None:
            param = self.flatparam
        #val1 = self.res_eval(param=param) * self.weight.data
        val1 = self.res_eval(param=param) * self.weight
        val2 = scipy.zeros(scipy.size(param),'d')
        i = 0
        for f in self.func:
            deriv = -2*val1 * f.deriv(param[i:i+f.npar])
            if f.npar_cor != 0:
                temp = sum(deriv[0:f.npar_cor],2)
                val2[i:i+f.npar_cor] = sum(temp,1)
            for n in arange(f.npar_ind):
                val2[i+f.npar_cor+n*self.data.nslice:
                     i+f.npar_cor+(n+1)*self.data.nslice]=sum(deriv[n+f.npar_cor],1)
            i=i+f.npar
        return val2
   

    def check_grad(self,eps=1e-3):
        """ Check the gradient of the objective function at the current parameters stored in the field flatparam."""
        print "%20s %20s %20s" % ("Finite difference","Objgrad","Rel. diff.")
        approx_grad = scipy.optimize.approx_fprime(self.flatparam, self.objfun, eps)
        comp_grad = self.objgrad()
        for n in arange(scipy.size(self.flatparam)):
            print "%20.6f %20.6f %20.6f" % (approx_grad[n],comp_grad[n],
                                            abs(approx_grad[n]-comp_grad[n]) / \
                                            max([abs(comp_grad[n]),1e-10]))
    
    def save_fit(self):
        """ Save the last fit parameters (fitpar) into the current parameters (flatparam and param)."""
        self.flatparam = scipy.array(self.fitpar,'d')
        self.param = self.unflat_param(self.fitpar)
    
    def fit(self,disp=False,save=False,deriv=True,maxfun=1000,msge=0,scale=None):
        
        """ Perform the model fitting by minimizing the objective function objfun."""
        x0 = scipy.zeros(scipy.size(self.flatparam),'d')
        for i in arange(scipy.size(self.flatparam)): x0[i] = self.flatparam[i]

        if deriv: xopt = scipy.optimize.fmin_tnc(self.objfun, self.flatparam.tolist(),
                                                 fprime=self.objgrad, 
                                                 approx_grad=False, bounds=self.bounds,
                                                 messages=msge,maxfun=maxfun,scale=scale)
        else:     xopt = scipy.optimize.fmin_tnc(self.objfun, self.flatparam.tolist(),
                                                 approx_grad=True, bounds=self.bounds,
                                                 messages=msge,maxfun=maxfun)

        res = scipy.array(xopt[2])
        self.fitpar = res
        
        if disp:
            return res
        if save:
            self.flatparam = scipy.array(self.fitpar,'d')
        self.khi2 = self.objfun(param=res) / \
                    (self.data.nlens*self.data.nslice + scipy.size(self.flatparam))

        
    def param_error(self,param=None):
        if param == None:
            param = self.fitpar
        jac = self.objgrad
        print 'jac: ',shape(jac(param))
        hess = lambda param: approx_deriv(jac,param,order=2)
        print 'hess: ',shape(hess(param))
        try:
            cov = scipy.linalg.inv(hess(param))
            return cov
        except:
            return numpy.zeros((len(param),len(param)),'d')
 
    def flat_param(self,param=None):
        param = scipy.array(scipy.cast['f'](param),shape=(scipy.size(param))).tolist()
        return param

    def unflat_param(self,param):
        if scipy.size(param) != scipy.size(self.param):
            raise ValueError, "Parameter list does not have the right size."
        newparam = []
        i = 0
        for f in self.func:    
            newparam.append(param[i:i+f.npar])
            i=i+f.npar
        return newparam

    def save_fit_as_SNIFS_cube(self):
        fit_cube = copy.deepcopy(self.data)
        fit_cube.data = self.evalfit()
        return fit_cube

    def save_guess_as_SNIFS_cube(self):
        guess_cube = copy.deepcopy(self.data)
        guess_cube.data = self.eval()
        return guess_cube
        
#######################################################
#                   Fit auxilliary functions          #
#######################################################

def approx_deriv(func, pars, dpars=None, order=3, eps=1e-6, args=()):
    """Let's assume len(pars)=N and func returns a array of shape S.

    S.optimize.approx_fprime corresponds to approx_deriv(order=2).

    S.derivative only works with univariate function. One could use
    scipy.optimize.approx_fprime (and associated check_grad) instead, but it
    only works with scalar function (e.g. chi2), and it cannot therefore be
    used to check model derivatives or hessian.
    """

    if order == 2:                      # Simplest crudest differentiation
        weights = scipy.array([-1,1])
    elif order == 3:                    # Simplest symmetric diff.
        weights = scipy.array([-1,0,1])/2.
    elif order == 5:
        weights = scipy.array([1,-8,0,8,-1])/12.
    elif order == 7:
        weights = scipy.array([-1,9,-45,0,45,-9,1])/60.
    elif order == 9:
        weights = scipy.array([3,-32,168,-672,0,672,-168,32,-3])/840.
    else:
        raise NotImplementedError

    if dpars is None:
        dpars = scipy.ones(len(pars))*eps  # N
    mat = scipy.diag(dpars)                 # NxN

    delta = scipy.arange(len(weights)) - (len(weights)-1)//2
    df = 0
    for w,d in zip(weights,delta):
        if w:
            df += w*scipy.array([ func(pars+d*dpi, *args) for dpi in mat]) # NxS

    f = func(pars, *args)               # S
    if f.ndim==0:                       # func returns a scalar S=()
        der = df/dpars                  # N
    else:                               # func returns an array of shape S
        der = df/dpars[...,scipy.newaxis]   # NxS

    return der

def fit_spectrum(spec,func='gaus1D',param=None,bounds=None,abs=False):
    if param==None:
        param=init_param(func)
    if not isinstance(spec,spectrum):
        raise TypeError, 'spec must be a pySNIFS.spectrum'
    # copying the input spectrum into a temporary one
    spec2 = copy.deepcopy(spec)
    spec2.data = spec2.data/scipy.mean(spec2.data) # Normalisation of the data
    if spec2.var != None:
        spec2.var = spec2.var/scipy.mean(spec2.var) # Normalisation of the data
    mod_spec = model(data=spec2,func=func,param=param,bounds=bounds)
    mod_spec.fit()
    return mod_spec.fitpar



