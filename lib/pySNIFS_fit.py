# -*- coding: utf-8 -*-
######################################################################
## Filename:      pySNIFS_fit.py
## Version:       $Revision$
## Description:
## Author:        Emmanuel Pecontal
## Author:        $Author$
## $Id$
######################################################################

from pySNIFS import *
import numpy as N
import scipy.special as SS
import scipy.optimize as SO
import scipy.linalg as SL
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
        self.name = 'gaus1D'
        self.l = N.reshape(cube.lbda, N.shape(cube.data))

    def comp(self,param):
        """
        Compute the gaussian function.
        @param param: Input parameters of the gaussian. A list of three
                      number: [xc,sigma,I]
        """

        self.param = param
        val = param[2] * N.exp(-0.5*(((self.l-param[0])/param[1])**2))

        return val

    def deriv(self,param):
        """
        Compute the derivative of the gaussian function with respect to the
        parameters.
        @param param: Input parameters of the gaussian. A list of three
                      number: [xc,sigma,I]
        """

        grad = N.zeros((self.npar_cor+self.npar_ind,) + N.shape(self.l),'d')
        expo = N.exp(-0.5*(((self.l-param[0])/param[1])**2))
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
        self.l = N.reshape(cube.lbda, N.shape(cube.data))

    def comp(self,param):
        """
        Compute the polynomial.
        @param param: Input parameters of the polynomial. A list of deg+1
                      numbers.
        """

        val = N.poly1d(param[::-1])(self.l)

        return val

    def deriv(self,param):
        """
        Compute the derivative of the polynomial with respect to its parameters.
        @param param: Input parameters of the polynomial. A list of deg+1
                      numbers.
        """

        grad = [ self.l**i for i in range(self.npar_cor) ]
        grad = N.array(grad)

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
        self.x = N.zeros_like(cube.data, dtype='d')
        self.y = N.zeros_like(cube.data, dtype='d')
        self.x[:][:] = cube.x
        self.y[:][:] = cube.y

    def comp(self,param):
        """
        Compute the gaussian function.
        @param param: Input parameters of the gaussian. A list of five number:
                      [xc,yc,sx,sy,I]
        """

        self.param = param
        tab_param_ind = N.reshape(param[self.npar_cor:],
                                  (self.npar_ind,self.nslice)).T
        val = tab_param_ind[:,4:5] * N.exp(-0.5*(
            ((self.x-tab_param_ind[:,0:1])/tab_param_ind[:,2:3])**2 +
            ((self.y-tab_param_ind[:,1:2])/tab_param_ind[:,3:4])**2))

        return val

    def deriv(self,param):
        """
        Compute the derivative of the gaussian function with respect to the
        parameters.
        @param param: Input parameters of the gaussian. A list of five number:
                      [xc,yc,sx,sy,I]
        """

        self.param = param
        tab_param_ind = N.reshape(param,(self.npar_ind,self.nslice)).T
        grad = N.ones((self.npar_ind,)+N.shape(self.x),'d')

        expo = N.exp(-0.5*(
            ((self.x-tab_param_ind[:,0:1])/tab_param_ind[:,2:3])**2 +
            ((self.y-tab_param_ind[:,1:2])/tab_param_ind[:,3:4])**2))
        val = expo * tab_param_ind[:,4:5] * \
              (self.x-tab_param_ind[:,0:1])/(tab_param_ind[:,2:3])**2
        grad[0] = val
        val = expo * tab_param_ind[:,4:5] * \
              (self.y-tab_param_ind[:,1:2])/(tab_param_ind[:,3:4])**2
        grad[1] = val
        val = expo * tab_param_ind[:,4:5] * \
              (self.x-tab_param_ind[:,0:1])**2 / (tab_param_ind[:,2:3])**3
        grad[2] = val
        val = expo * tab_param_ind[:,4:5] * \
              (self.y-tab_param_ind[:,1:2])**2 / (tab_param_ind[:,3:4])**3
        grad[3] = val
        grad[4] = expo

        return grad

class gaus2D_integ:
    """
    2D gaussian function with integration in the pixel, used by the
    L{model} class
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
        self.x = N.zeros_like(cube.data, dtype='d')
        self.y = N.zeros_like(cube.data, dtype='d')
        self.x[:][:] = cube.x
        self.y[:][:] = cube.y

    def comp(self,param):
        """
        Compute the gaussian function.
        @param param: Input parameters of the gaussian. A list of five number:
                      [xc,yc,sx,sy,I]
        """

        self.param = param
        tab_param_ind = N.reshape(param,(self.npar_ind,self.nslice)).T
        sq2 = 1.4142135623730951
        xpcn = (self.x + self.pix/2 - tab_param_ind[:,0:1]) / \
               (sq2*tab_param_ind[:,2:3])
        xmcn = (self.x - self.pix/2 - tab_param_ind[:,0:1]) / \
               (sq2*tab_param_ind[:,2:3])
        ypcn = (self.y + self.pix/2 - tab_param_ind[:,1:2]) / \
               (sq2*tab_param_ind[:,3:4])
        ymcn = (self.y - self.pix/2 - tab_param_ind[:,1:2]) / \
               (sq2*tab_param_ind[:,3:4])
        val = N.pi/2 * tab_param_ind[:,2:3] * tab_param_ind[:,3:4] * \
              tab_param_ind[:,4:5] * \
              (SS.erf(xpcn)-SS.erf(xmcn)) * (SS.erf(ypcn)-SS.erf(ymcn))

        return val

    def deriv(self,param):
        """
        Compute the derivative of the gaussian function with respect to the parameters.
        @param param: Input parameters of the gaussian. A list of five number: [xc,yc,sx,sy,I]
        """

        self.param = param
        tab_param_ind = N.reshape(param,(self.npar_ind,self.nslice)).T
        grad = N.ones((self.npar_ind,)+N.shape(self.x),'d')
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
        gxp = N.exp(-xpcn**2)
        gxm = N.exp(-xmcn**2)
        gyp = N.exp(-ypcn**2)
        gym = N.exp(-ymcn**2)
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
        self.x = N.zeros_like(cube.data, dtype='d')
        self.y = N.zeros_like(cube.data, dtype='d')
        self.x[:][:] = cube.x
        self.y[:][:] = cube.y

    def comp(self,param):
        """
        Compute the polynomial.
        @param param: Input parameters of the polynomial. A list of (deg+1)*(deg+2)/2 numbers.
        """
        self.param = param
        tab_param_ind = N.reshape(N.transpose(param),
                                  (self.npar_ind,self.nslice)).T
        n = 0
        val = N.ones_like(self.x, dtype='d') * tab_param_ind[:,0:1]
        #print str(self.param[0])
        for d in N.arange(self.deg)+1:
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
        tab_param_ind = N.reshape(N.transpose(param),
                                  (self.npar_ind,self.nslice)).T
        n = 0
        grad = N.ones((self.npar_ind,)+N.shape((self.x)),'d')
        #print str(self.param[0])

        for d in N.arange(self.deg)+1:
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
        self.x = N.zeros((cube.nslice,cube.nlens),'d')
        self.y = N.zeros((cube.nslice,cube.nlens),'d')
        self.l = N.zeros_like(cube.data.T, dtype='d')
        self.x[:][:] = cube.x
        self.y[:][:] = cube.y
        self.l[:][:] = cube.lbda
        self.l = self.l.T
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
        - C{param[11:]}: Intensities (one for each slice in the cube).
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
        return N.reshape(param[11:],(len(param[11:]),1))*(Icx*Icy+eps*Iwx*Iwy)

    def deriv(self,param):
        """
        Compute the derivative of the function with respect to its
        parameters.  @param param: Input parameters of the
        polynomial. A list numbers (see L{SNIFS_psf_3D.comp}).
        """
        self.param = param
        grad = N.zeros((self.npar_cor+self.npar_ind,)+N.shape(self.x),'d')

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

        gxrcp = N.exp(-xrcp**2)
        gxrcm = N.exp(-xrcm**2)
        gyrcp = N.exp(-yrcp**2)
        gyrcm = N.exp(-yrcm**2)
        gxrwp = N.exp(-xrwp**2)
        gxrwm = N.exp(-xrwm**2)
        gyrwp = N.exp(-yrwp**2)
        gyrwm = N.exp(-yrwm**2)
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
        grad[0:11] = grad[0:11] * N.reshape(param[11:],(1,len(param[11:]),1))

        return grad


class Hyper:
    """Example of Hyper class: (p-p̄)⋅Cov(p)⁻¹⋅(p-p̄)ᵀ"""

    def __init__(self, mean, cov, hparam=1.):

        self.hparam = hparam            # Hyper-parameter
        self.mean = N.asarray(mean)     # Mean parameters
        # Store inverse covariance (aka precision) matrix
        if cov.ndim==1:                 # Variance vector
            self.icov = N.diag(1/cov)
        else:                           # Full covariance
            self.icov = SL.pinvh(cov)

    def comp(self, params):
        """Hyper-term (p-p̄)⋅Cov(p)⁻¹⋅(p-p̄)ᵀ."""

        dpar = N.asarray(params) - self.mean        
        return self.hparam * dpar.dot(icov).dot(dpar) # Hyper-term ()

    def deriv(self, params):

        dpar = N.asarray(params) - self.mean
        return self.hparam * 2*dpar.dot(icov) # Hyper-jacobian (npar,)


#######################################################
#                  Model fitting                      #
#######################################################

class model:
    """
    Model fiting class
    """
    def __init__(self, func=['gaus2D'], data=None, param=None, bounds=None,
                 myfunc=None, hyper={}):
        """hyper={'fname':hyper term instance}."""
        
        self.fitpar = None

        if param is None:
            raise ValueError("A set of model param. values must be provided.")

        if isinstance(data,SNIFS_cube): # 3D-case
            self.model_1D = False
            self.model_2D = False
            self.model_3D = True
            self.data = data
        elif isinstance(data,image_array): # 2D-case
            self.model_1D = False
            self.model_2D = True
            self.model_3D = False
            self.data = SNIFS_cube()
            self.data.data = N.reshape(N.ravel(data.data),
                                       (1,len(N.ravel(data.data))))
            self.data.lbda = N.array([0])
            self.data.nslice = 1
            self.data.nlens = data.nx * data.ny
            self.data.i = N.ravel(N.indices((data.nx,data.ny))[0])
            self.data.j = N.ravel(N.indices((data.nx,data.ny))[1])
            self.data.x = self.data.i*data.stepx+data.startx
            self.data.y = self.data.j*data.stepy+data.starty
        elif isinstance(data,spectrum): # 1D-case
            self.model_1D = True
            self.model_2D = False
            self.model_3D = False
            self.data = SNIFS_cube()
            self.data.data = N.reshape(data.data[data.index_list],
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
            raise TypeError(
                "Unknown datatype (SNIFS_cube|image_array|spectrum)")

        # Predefined functions
        func_dict = {'gaus1D': gaus1D,
                     'poly1D': poly1D,
                     'gaus2D': gaus2D,
                     'poly2D': poly2D,
                     'gaus2D_integ': gaus2D_integ,
                     'SNIFS_psf_3D': SNIFS_psf_3D}

        # We add in the available function dictionary the user's ones
        if myfunc is not None:
            try:
                func_dict.update(myfunc)
            except:
                raise ValueError("User functions must be provided as "
                                 "dictionary {'name1':func1, 'name2':func2...}")

        if self.model_1D:
            avail_func = ['gaus1D','poly1D']
        else:
            avail_func = ['SNIFS_psf_3D','gaus2D','gaus2D_integ','poly2D']
        if myfunc is not None:          # Add user fns to available ones
            avail_func = avail_func + myfunc.keys()

        if data.var is None:            # Least-square
            self.chi2fit = False
            self.weight = N.ones_like(self.data.data, dtype='d')
        else:                           # Chi2
            self.chi2fit = True
            if self.model_3D:
                self.weight = N.where(data.var>0, 1/data.var, 0.).astype('d')
            elif self.model_2D: 
                self.data.var = N.array(data.var, dtype='d')
                self.data.var = self.data.var.reshape(1,self.data.var.size)
                self.weight = N.where(self.data.var>0, 1./self.data.var, 0.)
            elif self.model_1D:
                weight_val = N.where(data.var[data.index_list]>0,
                                     1./data.var[data.index_list],
                                     0.).astype('d')
                self.weight = N.reshape(weight_val,(len(data.index_list),1))

        self.khi2 = None                # Total chi2 (excluding hyper-term)

        ######### Function list analysis #########

        self.func = []
        for fstr in func:               # "name[;p1,p2...]"
            fname = fstr.split(';')[0]  # "name"
            if fname not in avail_func: # Unknown function
                if self.model_1D:
                    raise ValueError(
                        "Function %s not available for 1D models." % fname)
                else:
                    raise ValueError(
                        "Function %s not available for 2D/3D models." % fname)
            else:
                fn = func_dict[fname]   # Actual function (class)
            # Check if the function has internal parameters
            if len(fstr.split(';')) == 1:  # No internal parameters
                self.func.append(fn(self.data)) # Function instanciation
            else:                       # Presence of internal parameters
                fparams = fstr.split(';')[1] # "p1,p2,..."
                fparams = [ float(x) for x in fparams.split(',') ] # [p1,p2,...]
                if len(fparams) == 1:
                    fparams = fparams[0]
                self.func.append(fn(fparams, self.data)) # Fn instanciation

        ######### Parameter list analysis #########

        if len(param) != len(func):
            raise ValueError(
                "param list and func list must have the same length.")
        for i,f in enumerate(self.func):
            if f.npar != N.size(param[i]):
                raise ValueError(
                    "Function %s must have %d parameters, %d given." % 
                    (f.name, f.npar, len(param[i])))
        self.param = param
        nparam = [ len(p) for p in param ]
        self.nparam = N.sum(nparam)     # Total number of parameters
        self.flatparam = N.concatenate(param).astype('d')
        # Total degrees of freedom
        self.dof = self.data.nlens*self.data.nslice - self.nparam

        ######### Bounds list analysis #########

        if bounds is None:
            self.bounds = None
        else:
            if len(param) != len(bounds):
                raise ValueError(
                    "There must be one bound pairs for each variable.")
            self.bounds = []
            n = 0
            for i in range(len(param)):
                if len(param[i]) != len(bounds[i]):
                    raise ValueError(
                        "Function #%d has not the same " 
                        "bound pairs and variables number." % i)
                for j in range(len(param[i])):
                    self.bounds.append(bounds[i][j])
                n += nparam[i]

        ######### Hyper-terms #########

        self.hyper = hyper              # {'fname':hyper}


    def new_param(self,param=None):
        """ Store new parameters for the model evaluation. """

        if N.size(param) != N.size(self.param) or \
               len(param) != len(self.param):
            raise ValueError("param has not the correct size.")
        self.param = param
        self.flatparam = N.concatenate(param).astype('d')

    def eval(self, param=None):
        """Evaluate model with current parameters stored in flatparam."""
        
        if param is None:
            param = self.flatparam
        val = N.zeros((self.data.nslice,self.data.nlens),'d')
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
            raise ValueError("No fit parameters to evaluate model.")
        
        return self.eval(param=self.fitpar)

    def res_evalfit(self):
        """Evaluate model residuals at fitted parameters stored fitpar."""

        return self.data.data - self.evalfit()

    def eval_hyper(self, param=None):

        if param is None:
            param = self.flatparam
        i = 0
        hyper = 0.
        for f in self.func:             # Loop over functions
            pars = param[i:i+f.npar]    # Function parameters
            if f.name in self.hyper:    # Add hyper-term from this function
                hyper += self.hyper[f.name].comp(pars)
            i += f.npar

        return hyper                    # Total hyper-term ()

    def grad_hyper(self, param=None):

        if param is None:
            param = self.flatparam
        i = 0
        jac = N.zeros(N.size(param),'d') # (npars,)
        for f in self.func:             # Loop over functions
            pars = param[i:i+f.npar]    # Function parameters
            if f.name in self.hyper:    # Add hyper-term from this function
                hyper = self.hyper[f.name]
                if hasattr(hyper,'deriv'):
                    jac[i:i+f.npar] += hyper.deriv(pars)
                else:
                    jac[i:i+f.npar] += approx_deriv(hyper.comp, pars)
            i += f.npar

        return jac                      # Total hyper-jacobian (npars,)

    def objfun(self, param=None, hyper=True):
        """ Compute the objective function to be minimized:
        Sum(weight*(data-model)^2) at the given parameters
        values. Include hyper-term if needed."""
        
        chi2 = (self.res_eval(param=param)**2 * self.weight).sum()
        if self.hyper and hyper:        # Add hyper-terms
            chi2 += self.eval_hyper(param)

        return chi2

    def objgrad(self, param=None, hyper=True):
        """Compute the gradient of the objective function at the given
        parameters value."""

        if param is None:
            param = self.flatparam
        val = self.res_eval(param=param) * self.weight
        jac = N.zeros(N.size(param),'d') # (nparams,)
        i = 0
        for f in self.func:
            if hasattr(f,'deriv'):
                deriv = -2*val * f.deriv(param[i:i+f.npar])
            else:
                deriv = -2*val * approx_deriv(f.comp, param[i:i+f.npar])

            if f.npar_cor != 0:         # Sum over all axes except 0
                jac[i:i+f.npar_cor] = \
                    deriv[0:f.npar_cor].sum(axis=-1).sum(axis=-1)
            for n in range(f.npar_ind):
                jac[i+f.npar_cor+n*self.data.nslice:
                    i+f.npar_cor+(n+1)*self.data.nslice] = \
                    deriv[n+f.npar_cor].sum(axis=1)
            i += f.npar

        if self.hyper and hyper:        # Add hyper-jacobians
            jac += self.grad_hyper(param)
            
        return jac

    def check_grad(self, param=None, eps=1e-6):
        """ Check the gradient of the objective function at the current
        parameters stored in the field flatparam."""

        if param is None:
            param = self.flatparam
        print "%2s  %15s  %15s  %15s" % \
              ("##", "Finite diff.","Objgrad","Rel. diff.")
        grad = self.objgrad(param)
        approx = approx_deriv(self.objfun, param, order=3, eps=eps)
        for n in range(N.size(param)):
            print "%02d  %15.6f  %15.6f  %15.6f" % (
                n+1, approx[n],grad[n],
                abs(approx[n]-grad[n])/max(abs(grad[n]),1e-10))

    def save_fit(self):
        """ Save the last fit parameters (fitpar) into the current parameters
        (flatparam and param)."""

        self.flatparam = self.fitpar.copy()
        self.param = self.unflat_param(self.fitpar)

    def fit(self, disp=False, save=False, deriv=True,
            maxfun=1000, msge=0):
        """Perform the model fitting by minimizing the objective
        function objfun."""

        self.guessparam = self.flatparam.copy()

        # Help convergence by setting realistic objective: fmin=dof.
        # Use auto-offset and auto-scale
        x,nfeval,rc = SO.fmin_tnc(self.objfun, self.flatparam.tolist(),
                                  fprime=deriv and self.objgrad or None,
                                  approx_grad=not deriv,
                                  bounds=self.bounds,
                                  messages=msge, maxfun=maxfun,
                                  fmin=self.dof)

        # See SO.tnc.RCSTRINGS for status message
        self.status = rc>2 and rc or 0 # 0,1,2 means "fit has converged"
        if msge>=1:
            print "fmin_tnc (%d funcalls): %s" % \
                (nfeval, SO.tnc.RCSTRINGS[rc])
        self.fitpar = N.asarray(x,'d')

        # Reduced khi2 = khi2 / DoF (not including hyper term)
        self.khi2 = self.objfun(param=self.fitpar, hyper=False) / self.dof

        if disp:
            return self.fitpar
        
        if save:
            self.flatparam = self.fitpar.copy()

    def fit_bfgs(self, disp=False, save=False, deriv=True,
                 maxfun=1000, msge=0):
        """Same as fit, but using SO.fmin_l_bfgs_b instead of
        SO.fmin_tnc."""

        self.guessparam = self.flatparam.copy()

        x,f,d = SO.fmin_l_bfgs_b(self.objfun, self.flatparam.tolist(),
                                 fprime=deriv and self.objgrad or None,
                                 approx_grad=not deriv,
                                 bounds=self.bounds,
                                 iprint= msge==0 and -1 or 0)
        self.status = d['warnflag']
        if msge>=1:
            print "fmin_l_bfgs_b [%d funcalls]: %s" % (d['funcalls'],d['task'])
        self.fitpar = N.asarray(x,'d')

        # Reduced khi2 = khi2 / DoF
        self.khi2 = f / self.dof

        if disp:
            return self.fitpar
        
        if save:
            self.flatparam = self.fitpar.copy()

    def minimize(self, save=False, verbose=False,
                 method='TNC', tol=None, options=None):
        """Perform the model fitting by minimizing the objective
        function objfun. Similar to self.fit[_bfgs], but using unified
        minimizer SO.minimize."""

        self.guessparam = self.flatparam.copy()

        if options is None:
            options = {}
        if method.lower()=='tnc':
            options.setdefault('minfev', self.dof)

        self.res = SO.minimize(self.objfun, self.flatparam.tolist(),
                               method=method,
                               jac=self.objgrad,
                               bounds=self.bounds,
                               tol=tol, options=options)

        self.success = self.res.success # Boolean
        self.status = self.res.status   # Integer (can be >0 if successful)
        if verbose:
            print "minimize[%s]: %d funcalls, success=%s, status=%d: %s" % \
                  (method, self.res.nfev,
                   self.success, self.status, self.res.message)
        self.fitpar = self.res.x

        # Reduced khi2 = khi2 / DoF (not including hyper term)
        self.khi2 = self.objfun(param=self.fitpar, hyper=False) / self.dof

        if save and self.success:
            self.flatparam = self.fitpar.copy()

        return self.fitpar

    def facts(self, params=False, names=[]):

        from ToolBox.IO import str_magn

        if names:
            assert len(names)==self.nparam

        fns = [ (f.name,len(pars)) for f,pars in zip(self.func, self.param) ]
        s  = "Model: %s = %d parameters" % \
             (' + '.join( '%s[%d]' % fn for fn in fns ), self.nparam)
        if self.fitpar is not None:     # Minimization was performed
            s += "\n%s: status=%d, %s/Dof=%.2f/%d" % \
                 ('Chi2 minimization' if self.chi2fit else 'Least-squares',
                  self.status,
                  'Chi2' if self.chi2fit else 'RSS',
                  self.khi2*self.dof, self.dof)
            if params:
                s += "\n##  %s--Guess--  ---Fit---  --Error--  --Note--" % \
                     ('Name'.center(10,'-')+'  ' if names else '')
                covpar = self.param_error()
                dpars = N.sqrt(covpar.diagonal()) # StdErr on parameters
                for i in range(self.nparam):
                    ip = self.guessparam[i] # Initial guess
                    p = self.fitpar[i]      # Fitted parameters
                    dp = dpars[i]           # Error
                    pmin,pmax = self.bounds[i] # Bounds
                    if names:
                        name = '%10s  ' % names[i]
                    else:
                        name = ''
                    s += "\n%02d  %s%9s  %s" % \
                         (i+1,name, str_magn(ip, dp)[0],
                          '%9s  %9s' % str_magn(p, dp))
                    if not (pmin,pmax)==(None,None) and pmin==pmax:
                        s += '  fixed'                 # Fixed parameters
                    elif pmin is not None and p==pmin: # Hit minimal bound
                        s += '  <<<' 
                    elif pmax is not None and p==pmax: # Hit maximal bound
                        s += '  >>>'

        return s

    def param_cov(self, param=None, hyper=True):
        """Adjusted parameter covariance matrix computed from
        approximated hessian (including or not hyper-term if any).

        Hessian is 2nd-order derivative matrix, numerically estimated
        from 1st-order derivative vector. Non-fitted elements (lb=ub)
        should be removed prior to inversion.
        """

        if param is None:
            param = self.fitpar

        objgrad = lambda param: self.objgrad(param=param, hyper=hyper)
        hess = approx_deriv(objgrad, param) # Hessian approximation
        # Unfixed parameters
        free = ~N.array([ pmin==pmax!=None for pmin,pmax in self.bounds ])
        nfree = free.sum()
        selhess = hess[N.outer(free,free)].reshape(nfree,nfree)

        cov = N.zeros((len(param),len(param)),'d')
        try:
            selcov = 2 * SL.pinvh(selhess)
            cov[N.outer(free,free)] = selcov.ravel()
        except SL.LinAlgError:
            print "WARNING: cannot invert (selected) hessian approximation."
        
        return cov

    def param_error(self, param=None):
        """DEPRECATED"""

        return self.param_cov(param=param)

    def unflat_param(self,param):

        if N.size(param) != N.size(self.flatparam):
            raise ValueError("Parameter list does not have the right size.")
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

# 1st derivative central finite difference coefficients
# Ref: http://en.wikipedia.org/wiki/Finite_difference_coefficients
_CFDcoeffs = ((1/2.,),                               # 2nd-order
              (2/3.,-1/12.,),                        # 4th-order
              (3/4.,-3/20.,1/60.,),                  # 6th-order
              (4/5.,-1/5.,4/105.,-1/280.,),          # 8th-order
              )

def approx_deriv(func, pars, dpars=None, eps=1e-6, order=3, args=()):
    """Function 1st derivative approximation using central finite
    differences.

    Hereafter, m=len(pars) and func returns a array of shape
    S=[m×[m×]]n.

    .. Notes::

       * `scipy.derivative` only works with univariate function.
       * `scipy.optimize.approx_fprime` corresponds to
         `approx_deriv(order=2)`.
       * `scipy.optimize.approx_fprime` (and associated `check_grad`)
         only works with scalar function (e.g. chi2), and it cannot
         therefore be used to check model derivatives or hessian.

    .. Todo:: implement higher derivatives
    """

    horder = order // 2                 # Half-order

    if horder <= 4:
        coeffs = _CFDcoeffs[horder-1]
    else:
        raise NotImplementedError("approx_deriv supports order up to 8/9")

    if dpars is None:
        dpars = N.zeros(len(pars))+eps  # m
    mat = N.diag(dpars)                 # m×m diagonal matrix

    der = 0                             # Finite differences
    for i,c in enumerate(coeffs):       # Faster than N.sum(axis=0)
        der += c*N.array([ (func(pars+(i+1)*dpi, *args) -
                            func(pars-(i+1)*dpi, *args))
                           for dpi in mat ]) # m×S
    ## der = N.sum([ [ c*(func(pars+(i+1)*dpi,*args)-func(pars-(i+1)*dpi,*args))
    ##                 for dpi in mat ]
    ##               for i,c in enumerate(coeffs) ], axis=0)

    if der.ndim==1:                     # func actually returns a scalar (n=0)
        der /= dpars                    # m×0 / m = m
    else:                               # func returns an array of shape S
        der /= dpars[...,N.newaxis]     # m×S / m×1 = S

    return der                          # S


def fit_spectrum(spec,func='gaus1D',param=None,bounds=None,abs=False):
    if param is None:
        param=init_param(func)
    if not isinstance(spec,spectrum):
        raise TypeError('spec must be a pySNIFS.spectrum')
    # copying the input spectrum into a temporary one
    spec2 = deepcopy(spec)
    spec2.data = spec2.data/N.mean(spec2.data) # Normalisation of the data
    if spec2.var is not None:
        spec2.var = spec2.var/N.mean(spec2.var) # Normalisation of the data
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
        tol = 10 * eps * SL.norm(XtX,1)*max(m, n)
    #end

    P = N.zeros(n, 'i')
    P -= 1
    Z = N.arange(0,n)

    z = N.zeros(m, 'd')
    x = P.copy()
    ZZ = Z.copy()

    w = Xty - N.dot(XtX, x)

    # set up iteration criterion
    iter = 0
    itmax = 30 * n

    # outer loop to put variables into set to hold positive coefficients

    def find(X): return N.where(X)[0]

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
            XtyPP = N.array(Xty[PP])
            XtXPP = N.array(XtX[PP, N.array(PP)[:, N.newaxis]])
            z[PP] = N.dot(XtyPP, SL.pinv(XtXPP))
        #end
        z[ZZ] = 0

        # inner loop to remove elements from the positive set which no
        # longer belong
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

            Z[ij] = N.array(ij)
            P[ij] = -1
            PP = find(P != -1)
            ZZ = find(Z != -1)

            if len(PP) == 1 :
                XtyPP = Xty[PP]
                XtXPP = XtX[PP, PP]
                z[PP] = XtyPP / XtXPP
            else :
                XtyPP = N.array(Xty[PP])
                XtXPP = N.array(XtX[PP, N.array(PP)[:, N.newaxis]])
                z[PP] = N.dot(XtyPP, SL.pinv(XtXPP))
            #endif
            z[ZZ] = 0
        x = N.array(z)
        w = Xty - N.dot(XtX, x)

    return x, w
