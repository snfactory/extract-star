######################################################################
## Filename:      pySNIFS_fit.py
## Version:       $Revision$
## Description:   
## Author:        Emmanuel Pecontal
## Author:        $Author$
## $Id$
######################################################################

from pySNIFS import *
import scipy as S
from scipy import optimize
import numpy
import numpy.linalg.linalg as la 
from copy import deepcopy

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
        @param cube: Input cube. This is a L{SNIFS_cube} object with
                     only one spaxel. It is built from a
                     L{spectrum} object by the L{model} class.
        """
        self.npar_ind = 0
        self.npar_cor = 3
        self.npar = self.npar_ind*cube.nslice + self.npar_cor
        #self.l = zeros(shape(transpose(cube.data)),Float64)
        self.name = 'gaus1D'
        #self.l[:][:] = cube.lbda
        self.l = S.reshape(cube.lbda,S.shape(cube.data))
            
    def comp(self,param):
        """
        Compute the gaussian function.
        @param param: Input parameters of the gaussian. A list of three
                      number: [xc,sigma,I]
        """
        self.param = param
        val = param[2] * S.exp(-0.5*(((self.l-param[0])/param[1])**2))
        return val

    def deriv(self,param):
        """
        Compute the derivative of the gaussian function with respect to the
        parameters.
        @param param: Input parameters of the gaussian. A list of three
                      number: [xc,sigma,I]
        """
        grad = S.zeros((self.npar_cor+self.npar_ind,)+S.shape(self.l),'d')
        expo = S.exp(-0.5*(((self.l-param[0])/param[1])**2))
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
        @param cube: Input cube. This is a L{SNIFS_cube} object with only one
                     spaxel. It is built from a L{spectrum} object by the
                     L{model} class.
        """
        self.deg = int(deg)
        self.npar_ind = 0
        self.npar_cor = self.deg + 1
        self.npar = self.npar_ind*cube.nslice + self.npar_cor
        self.name = 'poly1D'
        self.l = S.reshape(cube.lbda,S.shape(cube.data))
        
    def comp(self,param):
        """
        Compute the polynomial.
        @param param: Input parameters of the polynomial. A list of deg+1
                      numbers.
        """
        val = S.poly1d(param[::-1])(self.l)
        return val
    
    def deriv(self,param):
        """
        Compute the derivative of the polynomial with respect to its parameters.
        @param param: Input parameters of the polynomial. A list of deg+1
                      numbers.
        """
        grad = [ self.l**i for i in range(self.npar_cor) ]
        grad = S.array(grad)
        #grad = S.zeros((self.npar_cor+self.npar_ind,)+shape(self.l),'d')
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
        @param cube: Input cube. This is a L{SNIFS_cube} object with only one
                     wavelength. It is built from a spectrum object by the
                     L{model} class.
        """ 
        self.nslice = cube.nslice
        self.npar_ind = 5
        self.npar_cor = 0
        self.npar = self.npar_ind*self.nslice + self.npar_cor
        self.name = 'gaus2D'
        self.x = S.zeros(S.shape(cube.data),'d')
        self.y = S.zeros(S.shape(cube.data),'d')
        self.x[:][:] = cube.x
        self.y[:][:] = cube.y

    def comp(self,param):
        """
        Compute the gaussian function.
        @param param: Input parameters of the gaussian. A list of five number:
                      [xc,yc,sx,sy,I]
        """
        self.param = param
        tab_param_ind = S.transpose(S.reshape(param[self.npar_cor:],(self.npar_ind,self.nslice)))
        val = tab_param_ind[:,4:5] * S.exp(-0.5*(((self.x-tab_param_ind[:,0:1])/tab_param_ind[:,2:3])**2 + ((self.y-tab_param_ind[:,1:2])/tab_param_ind[:,3:4])**2))
        return val
    
    def deriv(self,param):
        """
        Compute the derivative of the gaussian function with respect to the
        parameters.
        @param param: Input parameters of the gaussian. A list of five number:
                      [xc,yc,sx,sy,I]
        """
        self.param = param
        tab_param_ind = S.transpose(S.reshape(param,(self.npar_ind,self.nslice)))
        grad = S.ones((self.npar_ind,)+S.shape(self.x),'d')

        expo = S.exp(-0.5*(((self.x-tab_param_ind[:,0:1])/tab_param_ind[:,2:3])**2 + ((self.y-tab_param_ind[:,1:2])/tab_param_ind[:,3:4])**2))
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
        @param cube: Input cube. This is a L{SNIFS_cube} object with only one
                     wavelength. It is built from an L{image_array} object by
                     the class model.
        """ 
        self.nslice = cube.nslice
        self.pix = pix
        self.npar_ind = 5
        self.npar_cor = 0
        self.npar = self.npar_ind*self.nslice + self.npar_cor
        self.name = 'gauss2D_integ'
        self.x = S.zeros(S.shape(cube.data),'d')
        self.y = S.zeros(S.shape(cube.data),'d')
        self.x[:][:] = cube.x
        self.y[:][:] = cube.y
        
    def comp(self,param):
        """
        Compute the gaussian function.
        @param param: Input parameters of the gaussian. A list of five number:
                      [xc,yc,sx,sy,I]
        """
        self.param = param
        tab_param_ind = S.transpose(S.reshape(param,(self.npar_ind,self.nslice)))
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
        tab_param_ind = S.transpose(S.reshape(param,(self.npar_ind,self.nslice)))
        grad = S.ones((self.npar_ind,)+S.shape(self.x),'d')
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
        gxp = S.exp(-xpcn**2)
        gxm = S.exp(-xmcn**2)
        gyp = S.exp(-ypcn**2)
        gym = S.exp(-ymcn**2)
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
        self.deg = int(deg)
        self.npar_ind = int((deg+1)*(deg+2)/2)
        self.npar_cor = 0
        self.npar = self.npar_ind*self.nslice + self.npar_cor
        self.name = 'poly2D'
        self.x = S.zeros(S.shape(cube.data),'d')
        self.y = S.zeros(S.shape(cube.data),'d')
        self.x[:][:] = cube.x
        self.y[:][:] = cube.y
        
    def comp(self,param):
        """
        Compute the polynomial.
        @param param: Input parameters of the polynomial. A list of (deg+1)*(deg+2)/2 numbers.
        """
        self.param = param
        tab_param_ind = S.transpose(S.reshape(S.transpose(param),
                                              (self.npar_ind,self.nslice)))
        n = 0
        val = S.ones(S.shape(self.x),'d') * tab_param_ind[:,0:1]
        #print str(self.param[0])
        for d in S.arange(self.deg)+1:
            for j in range(d+1):
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
        tab_param_ind = S.transpose(S.reshape(S.transpose(param),
                                              (self.npar_ind,self.nslice)))
        n = 0
        grad = S.ones((self.npar_ind,)+S.shape((self.x)),'d')
        #print str(self.param[0])

        for d in S.arange(self.deg)+1:
            for j in range(d+1):
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
        self.x = S.zeros((cube.nslice,cube.nlens),'d')
        self.y = S.zeros((cube.nslice,cube.nlens),'d')
        self.l = S.zeros(S.shape(S.transpose(cube.data)),'d')
        self.x[:][:] = cube.x
        self.y[:][:] = cube.y
        self.l[:][:] = cube.lbda
        self.l = transpose(self.l)
        self.n_ref = 1e-6*(64.328 + \
                           29498.1/(146.-1./(self.lbda_ref*1e-4)**2) + \
                           255.4/(41.-1./(self.lbda_ref*1e-4)**2)) + 1.
        self.ADR_coef = 206265*(1e-6*(64.328 + \
                                      29498.1/(146.-1./(self.l*1e-4)**2) + \
                                      255.4/(41.-1./(self.l*1e-4)**2)) + 1. - \
                                self.n_ref)
        
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
        return S.reshape(param[11:],(len(param[11:]),1))*(Icx*Icy+eps*Iwx*Iwy)
    
    def deriv(self,param):
        """
        Compute the derivative of the function with respect to its parameters.
        @param param: Input parameters of the polynomial. A list numbers (see L{SNIFS_psf_3D.comp}).
        """
        self.param = param    
        grad = S.zeros((self.npar_cor+self.npar_ind,)+S.shape(self.x),'d')
        
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

    
        gxrcp = S.exp(-xrcp**2)
        gxrcm = S.exp(-xrcm**2)
        gyrcp = S.exp(-yrcp**2)
        gyrcm = S.exp(-yrcm**2)
        gxrwp = S.exp(-xrwp**2)
        gxrwm = S.exp(-xrwm**2)
        gyrwp = S.exp(-yrwp**2)
        gyrwm = S.exp(-yrwm**2)
        Icx = (erf(xrcp)-erf(xrcm))/2.
        Icy = (erf(yrcp)-erf(yrcm))/2.
        Iwx = (erf(xrwp)-erf(xrwm))/2.
        Iwy = (erf(yrwp)-erf(yrwm))/2.
        
        grad[2] = Icy*cost*(gxrcm-gxrcp)/sqrt(2*pi)/sig_cx + \
                  Icx*sint*(gyrcm-gyrcp)/sqrt(2*pi)/sig_cy + \
                  eps*(Iwy*cost*(gxrwm-gxrwp)/sqrt(2*pi)/sig_wx + \
                       Iwx*sint*(gyrwm-gyrwp)/sqrt(2*pi)/sig_wy)
        grad[3] = Icy*sint*(gxrcp-gxrcm)/sqrt(2*pi)/sig_cx + \
                  Icx*cost*(gyrcm-gyrcp)/sqrt(2*pi)/sig_cy + \
                  eps*(Iwy*sint*(gxrwp-gxrwm)/sqrt(2*pi)/sig_wx + \
                       Iwx*cost*(gyrwm-gyrwp)/sqrt(2*pi)/sig_wy)
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
        grad[8] = sigma_k*(der_cx+q_k**2*der_cy + eps*(der_wx+q_k**2*der_wy))
        grad[9] = q_k*sigma_k**2 *(der_cy+eps*der_wy)
        grad[10] = (Icy*yr/sqrt(2*pi)/sig_cx)*(gxrcm-gxrcp) + \
                   (Icx*xr/sqrt(2*pi)/sig_cy)*(gyrcp-gyrcm) + \
                   eps*((Iwy*yr/sqrt(2*pi)/sig_wx)*(gxrwm-gxrwp) + \
                        (Iwx*xr/sqrt(2*pi)/sig_wy)*(gyrwp-gyrwm))
        grad[11] = Icx*Icy + eps*Iwx*Iwy
        grad[0:11] = grad[0:11] * S.reshape(param[11:],(1,len(param[11:]),1))
        
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
        
        if param is None:
            raise ValueError("A set of model parameters values must be provided.")
            
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
            self.data.data = S.reshape(S.ravel(data.data),
                                       (1,len(S.ravel(data.data))))
            self.data.lbda = S.array([0])
            self.data.nslice = 1
            self.data.nlens = data.nx * data.ny
            self.data.i = S.ravel(numpy.indices((data.nx,data.ny))[0]) 
            self.data.j = S.ravel(numpy.indices((data.nx,data.ny))[1]) 
            self.data.x = self.data.i*data.stepx+data.startx
            self.data.y = self.data.j*data.stepy+data.starty
        elif isinstance(data,spectrum):
            self.model_1D = True
            self.model_2D = False
            self.model_3D = False
            self.data = SNIFS_cube()
            self.data.data = S.reshape(data.data[data.index_list],
                                       (len(data.index_list),1))
            self.data.lbda = data.x[data.index_list]
            self.data.nslice = len(data.index_list)
            self.data.nlens = 1
            self.data.x = None
            self.data.y = None
            self.data.i = None
            self.data.j = None
            self.data.no = None
        else:
            raise TypeError("data array must be a SNIFS_cube, image_array or spectrum object.")
            
        
        func_dict = {}
        func_dict['gaus1D']=gaus1D
        func_dict['poly1D']=poly1D
        func_dict['gaus2D']=gaus2D
        func_dict['poly2D']=poly2D
        func_dict['gaus2D_integ']=gaus2D_integ
        func_dict['SNIFS_psf_3D']=SNIFS_psf_3D
        
        # We add in the available function dictionary the user's ones
        if myfunc is not None:
            try:
                func_dict.update(myfunc)
            except:
                raise ValueError("User functions must be provided as dictionary {'name1':func1, 'name2':func2...}")

        if self.model_1D:
            avail_func = ['gaus1D','poly1D']
            if myfunc is not None:
                avail_func = avail_func+myfunc.keys()
                
        else:
            avail_func = ['SNIFS_psf_3D','gaus2D','gaus2D_integ','poly2D']
            if myfunc is not None:
                avail_func = avail_func+myfunc.keys()
                
        if data.var is None:
            #self.weight = SNIFS_cube()
            self.weight = S.ones(S.shape(self.data.data),'d')
            #self.weight.ones_from(self.data)
        else:
            if self.model_3D:
                #self.weight = SNIFS_cube()
                #self.weight.data = S.array(where(data.var!=0,1./abs(data.var),0.),'d')
                self.weight = S.where(data.var>0, 1/data.var, 0.).astype('d')
            elif self.model_2D:# TODO: Implement the variance
                self.data.var = S.array(data.var, dtype='d')
                self.data.var = self.data.var.reshape(1,self.data.var.size)
                self.weight = S.where(self.data.var>0, 1./self.data.var, 0.)
            elif self.model_1D:
                #self.weight = SNIFS_cube()
                weight_val = S.where(data.var[data.index_list]>0,
                                     1./data.var[data.index_list],
                                     0.).astype('d')
                self.weight = S.reshape(weight_val,(len(data.index_list),1))
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
                internal_param = [float(x) for x in inpar_string.split(',')]
                if len(internal_param) == 1:
                    internal_param = internal_param[0]
                self.func.append(func_dict[fstring](internal_param,self.data))

        ######### Parameter list analysis #########
        if len(param) != len(func):
            raise ValueError, "param list and func list must have the same length."
        for i,f in enumerate(self.func):
            if f.npar != S.size(param[i]):
                raise ValueError, "Function %s must have %d parameters, %d given." % \
                      (f.name, f.npar, len(param[i]))
        self.param = param
        nparam = [ len(p) for p in param ]
        self.nparam = S.sum(nparam)
        self.flatparam = S.concatenate(param).astype('d')

        ######### Bounds list analysis #########
        if bounds is None:
            self.bounds = None
        else:
            if len(param) != len(bounds):
                raise ValueError, "There must be one bound pairs for each variable."
            self.bounds = []
            n = 0
            for i in range(len(param)):
                if len(param[i]) != len(bounds[i]):
                    raise ValueError, "Function #%d has not the same bound pairs and variables number." % i
                for j in range(len(param[i])):
                    self.bounds.append(bounds[i][j])
                n += nparam[i]
 
    def new_param(self,param=None):
        """ Store new parameters for the model evaluation. """
        nparam = [ len(p) for p in param ]
        if S.size(param) != S.size(self.param) or \
               len(param) != len(self.param):
            raise ValueError, "param has not the correct size."
        self.param = param
        self.flatparam = S.concatenate(param).astype('d')
   
    def eval(self, param=None):
        """Evaluate model with current parameters stored in flatparam."""
        if param is None:
            param = self.flatparam
        val = S.zeros((self.data.nslice,self.data.nlens),'d')
        i = 0
        for f in self.func:
            val += f.comp(param[i:i+f.npar])
            i += f.npar
        return val

    def res_eval(self, param=None):
        """Evaluate model residuals with current parameters stored in
        param (or flatparam)."""
        return self.data.data - self.eval(param)

    def evalfit(self):
        """Evaluate model at fitted parameters stored in fitpar."""
        if self.fitpar is None:
            raise ValueError, "No fit parameters to evaluate model."
        return self.eval(param=self.fitpar)

    def res_evalfit(self):
        """Evaluate model residuals at fitted parameters stored fitpar."""
        return self.data.data - self.evalfit()

    def objfun(self, param=None):
        """ Compute the objective function to be minimized:
        Sum(weight*(data-model)^2) at the given parameters values."""
        return (self.res_eval(param=param)**2 * self.weight).sum()
        
    def objgrad(self, param=None):
        """ Compute the gradient of the objective function at the given
        parameters value."""
        if param is None:
            param = self.flatparam
        val1 = self.res_eval(param=param) * self.weight
        val2 = S.zeros(S.size(param),'d')
        i = 0
        for f in self.func:
            if hasattr(f,'deriv'):
                deriv = -2*val1 * f.deriv(param[i:i+f.npar])
                #try:
                #    assert S.allclose(approx_deriv(f.comp, param[i:i+f.npar]),
                #                      f.deriv(param[i:i+f.npar]))
                #except AssertionError:
                #    print "Deriv\n", f.deriv(param[i:i+f.npar])
                #    print "Approx\n", approx_deriv(f.comp, param[i:i+f.npar])
            else:
                #raise AttributeError,"%s has not 'deriv' method" % f.name
                #print "model.objgrad: using approx_deriv for", f.name
                deriv = -2*val1 * approx_deriv(f.comp, param[i:i+f.npar])
            #print "DEBUG OG: val1=%s, val2=%s" % \
            #    (S.shape(val1),S.shape(val2))
            #print "DEBUG OG: param=%s, param[]=%s" % \
            #    (S.shape(param),S.shape(param[i:i+f.npar]))
            #print "DEBUG OG: f.comp=%s, f.deriv=%s" % \
            #    (S.shape(f.comp(param[i:i+f.npar])),
            #     S.shape(f.deriv(param[i:i+f.npar])))
            #print "DEBUG OG: approx_deriv=%s, deriv=%s" % \
            #    (S.shape(approx_deriv(f.comp, param[i:i+f.npar])),
            #     S.shape(deriv))

            if f.npar_cor != 0:         # Sum over all axes except 0
                val2[i:i+f.npar_cor] = \
                     deriv[0:f.npar_cor].sum(axis=-1).sum(axis=-1)
            for n in range(f.npar_ind):
                val2[i+f.npar_cor+n*self.data.nslice:
                     i+f.npar_cor+(n+1)*self.data.nslice] = \
                     deriv[n+f.npar_cor].sum(axis=1)
            i += f.npar
        return val2

    def check_grad(self, param=None, eps=1e-6):
        """ Check the gradient of the objective function at the current
        parameters stored in the field flatparam."""
        if param is None:
            param = self.flatparam
        print "%20s %20s %20s" % ("Finite difference","Objgrad","Rel. diff.")
        approx_grad = S.optimize.approx_fprime(param, self.objfun, eps)
        comp_grad = self.objgrad(param)
        for n in range(S.size(param)):
            print "%20.6f %20.6f %20.6f" % (approx_grad[n],comp_grad[n],
                                            abs(approx_grad[n]-comp_grad[n]) / \
                                            max([abs(comp_grad[n]),1e-10]))
    
    def save_fit(self):
        """ Save the last fit parameters (fitpar) into the current parameters
        (flatparam and param)."""
        self.flatparam = self.fitpar.copy()
        self.param = self.unflat_param(self.fitpar)
    
    def fit(self, disp=False, save=False, deriv=True,
            maxfun=1000, msge=0, scale=None):
        """ Perform the model fitting by minimizing the objective function
        objfun."""

        # From scipy 0.6.0, fmin_tnc's output was inverted, and auto-offset is buggy
        if (S.__version__ >='0.6.0'): 
            x,nfeval,rc = S.optimize.fmin_tnc(self.objfun, self.flatparam.tolist(),
                                              fprime=deriv and self.objgrad or None, 
                                              approx_grad=not deriv,
                                              bounds=self.bounds, offset=[0]*self.nparam,
                                              messages=msge,maxfun=maxfun,
                                              scale=deriv and scale or None)
        else:
            rc,nfeval,x = S.optimize.fmin_tnc(self.objfun, self.flatparam.tolist(),
                                              fprime=deriv and self.objgrad or None, 
                                              approx_grad=not deriv,
                                              bounds=self.bounds,
                                              messages=msge,maxfun=maxfun,
                                              scale=deriv and scale or None)

        # See S.optimize.tnc.RCSTRINGS for status message
        self.status = (rc not in [0,1]) and rc or 0
        if msge>=1:
            print "fmin_tnc (%d funcalls): %s" % \
                (nfeval,S.optimize.tnc.RCSTRINGS[rc])
        self.fitpar = S.asarray(x,'d')

        # Reduced khi2 = khi2 / DoF
        self.dof = self.data.nlens*self.data.nslice - self.nparam
        self.khi2 = self.objfun(param=self.fitpar) / self.dof

        if disp:
            return self.fitpar
        if save:
            self.flatparam = self.fitpar.copy()
        
    def fit_bfgs(self, disp=False, save=False, deriv=True,
                 maxfun=1000, msge=0, scale=None):
        """Same as fit, but using S.optimize.fmin_l_bfgs_b instead of
        S.optimize.fmin_tnc."""

        x,f,d = S.optimize.fmin_l_bfgs_b(self.objfun, self.flatparam.tolist(),
                                         fprime=deriv and self.objgrad or None,
                                         approx_grad=not deriv,
                                         bounds=self.bounds,
                                         iprint= msge==0 and -1 or 0)
        self.status = d['warnflag']
        if msge>=1:
            print "fmin_l_bfgs_b [%d funcalls]: %s" % (d['funcalls'],d['task'])
        self.fitpar = S.asarray(x,'d')

        # Reduced khi2 = khi2 / DoF
        self.dof = self.data.nlens*self.data.nslice - self.nparam
        self.khi2 = f / self.dof

        if disp:
            return self.fitpar
        if save:
            self.flatparam = self.fitpar.copy()

    def param_error(self,param=None):
        """Actually returns covariance matrix computed from hessian."""
        
        if param is None:
            param = self.fitpar
        try:
            # Hessian is 2nd-order derivative matrix, numerically estimated
            # from 1st-order derivative vector. Non-fitted elements (lb=ub)
            # should be removed prior to inversion.
            hess = approx_deriv(self.objgrad,param)
            cov = 2 * S.linalg.inv(hess)
            return cov
        except:
            return numpy.zeros((len(param),len(param)),'d')
 
    def unflat_param(self,param):
        
        if S.size(param) != S.size(self.param):
            raise ValueError, "Parameter list does not have the right size."
        newparam = []
        i = 0
        for f in self.func:    
            newparam.append(param[i:i+f.npar])
            i += f.npar
        return newparam

    def save_fit_as_SNIFS_cube(self):
        fit_cube = deepcopy(self.data)
        fit_cube.data = self.evalfit()
        return fit_cube

    def save_guess_as_SNIFS_cube(self):
        guess_cube = deepcopy(self.data)
        guess_cube.data = self.eval()
        return guess_cube
        
#######################################################
#                   Fit auxilliary functions          #
#######################################################

def approx_deriv(func, pars, dpars=None, order=3, eps=1e-6, args=()):
    """Let's assume len(pars)=N and func returns a array of shape S.

    S.optimize.approx_fprime corresponds to approx_deriv(order=2).

    S.derivative only works with univariate function. One could use
    S.optimize.approx_fprime (and associated check_grad) instead, but it
    only works with scalar function (e.g. chi2), and it cannot therefore be
    used to check model derivatives or hessian.
    """

    if order == 2:                      # Simplest crudest differentiation
        weights = S.array([-1,1])
    elif order == 3:                    # Simplest symmetric diff.
        weights = S.array([-1,0,1])/2.
    elif order == 5:
        weights = S.array([1,-8,0,8,-1])/12.
    elif order == 7:
        weights = S.array([-1,9,-45,0,45,-9,1])/60.
    elif order == 9:
        weights = S.array([3,-32,168,-672,0,672,-168,32,-3])/840.
    else:
        raise NotImplementedError

    if dpars is None:
        dpars = S.ones(len(pars))*eps   # (N,)
    mat = S.diag(dpars)                 # (N,N)

    delta = S.arange(len(weights))-(len(weights)-1)//2 # [-order/2...+order/2]
    df = 0
    for w,d in zip(weights,delta):
        if w:
            df += w*S.array([ func(pars+d*dpi, *args) for dpi in mat ]) # (N,S)

    f = func(pars, *args)               # (S,)
    #if f.ndim==0:                       # func returns a scalar S=()
    #    der = df/dpars                  # (N,)
    #else:                               # func returns an array of shape S
    #    der = df/dpars[...,S.newaxis]   # (N,S)
    der = df / dpars.reshape((-1,)+(1,)*f.ndim)
    #print "DEBUG AD: f=%s, df=%s, pars=%s, dpars=%s, der=%s" % \
    #    (S.shape(f),S.shape(df),S.shape(pars),S.shape(dpars),S.shape(der))

    return der

def fit_spectrum(spec,func='gaus1D',param=None,bounds=None,abs=False):
    if param is None:
        param=init_param(func)
    if not isinstance(spec,spectrum):
        raise TypeError, 'spec must be a pySNIFS.spectrum'
    # copying the input spectrum into a temporary one
    spec2 = deepcopy(spec)
    spec2.data = spec2.data/S.mean(spec2.data) # Normalisation of the data
    if spec2.var is not None:
        spec2.var = spec2.var/S.mean(spec2.var) # Normalisation of the data
    mod_spec = model(data=spec2,func=func,param=param,bounds=bounds)
    mod_spec.fit()
    return mod_spec.fitpar


def fnnls(XtX, Xty, tol = 0) : 
    #FNNLS Non-negative least-squares. 
    # 
    # Adapted from NNLS of Mathworks, Inc. 
    #          [x,w] = nnls(X, y) 
    # 
    # x, w = fnnls(XtX,Xty) returns the vector X that solves x = pinv(XtX)*Xty 
    # in a least squares sense, subject to x >= 0. 
    # Differently stated it solves the problem min ||y - Xx|| if 
    # XtX = X'*X and Xty = X'*y. 
    # 
    # A default tolerance of TOL = MAX(SIZE(XtX)) * NORM(XtX,1) * EPS 
    # is used for deciding when elements of x are less than zero. 
    # This can be overridden with x = fnnls(XtX,Xty,TOL). 
    # 
    # [x,w] = fnnls(XtX,Xty) also returns dual vector w where 
    # w(i) < 0 where x(i) = 0 and w(i) = 0 where x(i) > 0. 
    # 
    # See also NNLS and FNNLSb 
     
    # L. Shure 5-8-87 
    # Revised, 12-15-88,8-31-89 LS. 
    # (Partly) Copyright (c) 1984-94 by The MathWorks, Inc. 
     
    # Modified by R. Bro 5-7-96 according to 
    #       Bro R., de Jong S., Journal of Chemometrics, 1997, 11, 393-401 
    # Corresponds to the FNNLSa algorithm in the paper 
    # 
    # Rasmus bro 
    # Chemometrics Group, Food Technology 
    # Dept. Dairy and Food Science 
    # Royal Vet. & Agricultural 
    # DK-1958 Frederiksberg C 
    # Denmark 
    # rb@... 
    # http://newton.foodsci.kvl.dk/users/rasmus.html 
    #  Reference: 
    #  Lawson and Hanson, "Solving Least Squares Problems", Prentice-Hall, 1974. 
    # 
     
    # initialize variables
    m,n = XtX.shape
 
    if tol == 0 : 
        eps = 2.2204e-16 
        tol = 10 * eps * la.norm(XtX,1)*max(m, n)
    #end 
 
    P = numpy.zeros(n, 'i') 
    P -= 1
    Z = numpy.arange(0,n) 
 
    z = numpy.zeros(m, 'd') 
    x = P.copy()
    ZZ = Z.copy()
 
    w = Xty - numpy.dot(XtX, x) 
 
    # set up iteration criterion 
    iter = 0 
    itmax = 30 * n 

    # outer loop to put variables into set to hold positive coefficients 

    def find(X): return numpy.where(X)[0]

    while Z.any() and (w[ZZ] > tol).any() : 
        wt = w[ZZ].max() 
        t = find(w[ZZ] == wt) 
        t = t[-1:][0] 
        t = ZZ[t] 
        P[t] = t 
        Z[t] = -1 
        PP = find(P != -1) 
        
        ZZ = find(Z != -1) 
        if len(PP) == 1 : 
            XtyPP = Xty[PP] 
            XtXPP = XtX[PP, PP] 
            z[PP] = XtyPP / XtXPP 
        else : 
            XtyPP = numpy.array(Xty[PP]) 
            XtXPP = numpy.array(XtX[PP, numpy.array(PP)[:, numpy.newaxis]]) 
            z[PP] = numpy.dot(XtyPP, la.pinv(XtXPP)) 
        #end 
        z[ZZ] = 0 

        # inner loop to remove elements from the positive set which no longer belong 
        while (z[PP] <= tol).any() and (iter < itmax) : 
            iter += 1
            iztol = find(z <= tol)
            ip = find(P[iztol] != -1)
            QQ = iztol[ip]

            if len(QQ) == 1 :
                alpha = x[QQ] / (x[QQ] - z[QQ]) 
            else : 
                x_xz = x[QQ] / (x[QQ] - z[QQ])
                alpha = x_xz.min()

            x += alpha * (z - x) 
            iabs = find(abs(x) < tol) 
            ip = find(P[iabs] != -1) 
            ij = iabs[ip] 

            Z[ij] = numpy.array(ij) 
            P[ij] = -1 
            PP = find(P != -1) 
            ZZ = find(Z != -1) 

            if len(PP) == 1 : 
                XtyPP = Xty[PP] 
                XtXPP = XtX[PP, PP] 
                z[PP] = XtyPP / XtXPP 
            else : 
                XtyPP = numpy.array(Xty[PP]) 
                XtXPP = numpy.array(XtX[PP, numpy.array(PP)[:, numpy.newaxis]])
                z[PP] = numpy.dot(XtyPP, la.pinv(XtXPP)) 
            #endif 
            z[ZZ] = 0 
        x = numpy.array(z) 
        w = Xty - numpy.dot(XtX, x) 
 
    return x, w 
