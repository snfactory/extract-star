#!/usr/bin/python
import os
import sys
import pySNIFS
import pySNIFS_fit
from optparse import OptionParser
import pyfits
import pylab
import matplotlib
import numarray
from numarray import NewAxis,sqrt,sum,where,arange,compress
from numarray import linear_algebra as L
from numarray import nd_image as N
import scipy
from scipy import pi

def print_msg(str,v,v_print):
    if v > v_print:
        print str
                  
def plot_non_chromatic_param(ax,par_vec,lbda,guess_par,fitpar,str_par):
    pylab.plot(lbda,par_vec,'bo')
    pylab.axhline(guess_par,linestyle='--')
    pylab.axhline(fitpar,linestyle='-')
    pylab.ylabel(r'$%s$'%(str_par),fontsize=15)
    pylab.text(0.05,0.7,r'$\rm{Guess:}\hspace{0.5} %s = %4.2f \hspace{1} \rm{Fit:}\hspace{0.5} %s = %4.2f$'%(str_par,guess_par,str_par,fitpar),transform=ax.transAxes,fontsize=10)
    #pylab.legend(('2D PSF','3D PSF Guess','3D PSF Fit'))
    pylab.xticks(fontsize=8)
    pylab.yticks(fontsize=8)

def comp_spec(cube,psf_param,intpar=[None,None]):
    signal = pySNIFS.SNIFS_cube(cube,num_array=True)
    var = pySNIFS.SNIFS_cube(cube,noise=True,num_array=True)
    # DIRTY PATCH TO REMOVE BAD SPECTRA FROM THEIR VARIANCE
    var.data = numarray.where(var.data>1e20,0.,var.data)
    model = pySNIFS_fit.SNIFS_psf_3D(intpar,signal)
    # The PSF parameters are only the shape parameters. We set the intensity of each slice to 1.
    param = psf_param+[1. for i in arange(signal.nslice)]
    

    # Linear khi2 minimization
    #alpha = sum(model.comp(param)**2/var.data,1)
    #beta = sum(model.comp(param)/var.data,1)
    #gamma = sum(model.comp(param)*signal.data/var.data,1)
    #delta = sum(1./var.data,1)
    #epsilon = sum(signal.data/var.data,1)
    #obj = (beta*epsilon-delta*gamma)/(beta**2 - alpha*delta)
    #sky = (beta*gamma-alpha*epsilon)/(beta**2 - alpha*delta)
    #spec = scipy.zeros((3,signal.nslice),'d')
    #spec[0,:] = signal.lbda
    #spec[1,:] = obj
    #spec[2,:] = sky

    # Rejection of bad points
    lapl = N.filters.laplace(numarray.array(signal.data)/N.mean(signal.data))
    fdata = N.filters.median_filter(numarray.array(signal.data),size=[1,3])
    hist = pySNIFS.histogram(numarray.ravel(abs(lapl)),nbin=100,Max=100,cumul=True)
    threshold = hist.x[numarray.argmax(numarray.where(hist.data<0.9999,0,1))]
    mask = numarray.where(abs(lapl)>threshold,0.,1.)
    #signal.data = signal.data * mask + fdata*(1.-mask)
    var.data = var.data*mask
    weight = numarray.where(var.data!=0,1./var.data,0)
    # Fit on masked data
    
    psf = numarray.array(model.comp(param),'d')
    alpha = sum(weight*psf**2,1)
    beta = sum(weight*psf,1)
    gamma = sum(psf*signal.data*weight,1)
    delta = sum(weight,1)
    epsilon = sum(signal.data*weight,1)
    obj = (beta*epsilon-delta*gamma)/(beta**2 - alpha*delta)
    sky = (beta*gamma-alpha*epsilon)/(beta**2 - alpha*delta)
    spec = numarray.zeros((3,signal.nslice),'d')
    spec[0,:] = signal.lbda
    spec[1,:] = obj
    spec[2,:] = sky
  
    return spec
    
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--in", type="string",dest="incube",help="Input datacube")
    parser.add_option("-o", "--out", type="string",dest="outspec",help="Output star spectrum")
    parser.add_option("-s", "--sky", type="string",dest="outsky",help="Output sky spectrum")
    parser.add_option("-g", "--gentools", type="float",dest="gentools_vers",default=6.3,help="IFU_gentools version")
    parser.add_option("-p", "--plot", action="store_true",dest="isplot",help="Plot flag")
    parser.add_option("-v", "--verbose", type="int", dest="verbosity_level",default=0,help="Verbosity level")

    opts,pars = parser.parse_args()

    
    #print_msg("Converting input datacube in e3d format...",opts.verbosity_level,0) 
    #path = "$IFU_PATH/IFU_gentools-%3.1f/user/bin"%opts.gentools_vers
    #os.system('%s/convert_file -in %s -out tempo2_e3d.fits -inputformat tiger+fits -outputformat euro3d -quiet -noask'%(path,opts.incube))
    print_msg("Opening the datacube...",opts.verbosity_level,0)
    #cube = pyfits.open("tempo2_e3d.fits")
    cube = pyfits.open(opts.incube)
    print_msg("Extracting slices...",opts.verbosity_level,0)
    if cube[1].header.get('channel')[0] == 'B':
        slices=[10,900,20]
        first_slice = 10
        lstep = 20
    elif cube[1].header.get('channel')[0] == 'R':
        slices=[10,1500,30]
        first_slice = 10
        lstep = 30

    signal = pySNIFS.SNIFS_cube(opts.incube,l=slices,num_array=False,s=True)
    var = pySNIFS.SNIFS_cube(opts.incube,l=slices,num_array=False,noise=True,s=True)
    # Normalisation of the signal and variance in order to avoid numerical problems withh too small numbers
    norm = scipy.mean(scipy.mean(signal.data))
    signal.data = signal.data / norm
    var.data = var.data / (norm**2)
    
    # DIRTY PATCH TO REMOVE BAD SPECTRA FROM THEIR VARIANCE
    # var.data = scipy.where(var.data>1e20,1.,var.data)

    
    # Rejection of bad points
    print_msg("Rejection of slices with bad values...",opts.verbosity_level,0)
    max_spec = L.mlab.max(signal.data,1)
    med = scipy.median(N.filters.median_filter(max_spec,5) - max_spec)
    tmp1 = numarray.where(abs(max_spec-med)<5*sqrt(scipy.median((max_spec - med)**2)),0,1)
    indice = (compress(arange(len(tmp1))*(1-tmp1)!=0,arange(len(tmp1)))).tolist()
    tmp_signal = numarray.array(signal.data)[indice]
    tmp_var = numarray.array(var.data)[indice]
    tmp_lbda = numarray.array(signal.lbda)[indice]
    signal.data = scipy.array(tmp_signal)
    signal.lbda = scipy.array(tmp_lbda)
    signal.nslice = len(signal.lbda)
    var.data = scipy.array(tmp_var)
    var.lbda = scipy.array(tmp_lbda)
    var.nslice = len(signal.lbda)

    ##### Computing guess parameters from slice by slice 2D fitting #####
    print_msg("Slice by slice 2D fitting...",opts.verbosity_level,0)
    
    nslice = signal.nslice
    xc_vec = scipy.zeros(signal.nslice,'d')
    yc_vec = scipy.zeros(signal.nslice,'d')
    sigc_vec = scipy.zeros(signal.nslice,'d')
    q_vec = scipy.zeros(signal.nslice,'d')
    eps_vec = scipy.zeros(signal.nslice,'d')
    sigk_vec = scipy.zeros(signal.nslice,'d')
    qk_vec = scipy.zeros(signal.nslice,'d')
    theta_vec = scipy.zeros(signal.nslice,'d')
    int_vec = scipy.zeros(signal.nslice,'d')
    sky_vec = scipy.zeros(signal.nslice,'d')
    for i in arange(signal.nslice):
        print_msg("2D Fitting of slice %d"%(i),opts.verbosity_level,2)
        signal2 = pySNIFS.SNIFS_cube()
        var2 = pySNIFS.SNIFS_cube()
        signal2.nslice = 1
        signal2.nlens = signal.nlens
        data = numarray.array(signal.data)[i,NewAxis]
        signal2.data = scipy.array(data)
        signal2.x = signal.x
        signal2.y = signal.y
        signal2.lbda = scipy.array([signal.lbda[i]])
        var2.nslice = 1
        var2.nlens = var.nlens
        data = numarray.array(var.data)[i,NewAxis]
        var2.data = scipy.array(data)
        var2.x = var.x
        var2.y = var.y
        var2.lbda = scipy.array([var.lbda[i]])

        signal2.data = signal2.data
        var2.data = var2.data 
        
        # Evaluation of the background for the current slice
        sky = min(signal2.data[0]+5*var2.data[0])

        # Evaluation of the centroid of the current slice
        sl_int = N.filters.median_filter(signal2.data[0],3)
        xc = sum(signal2.x*(sl_int-sky))/sum(sl_int-sky)
        yc = sum(signal2.y*(sl_int-sky))/sum(sl_int-sky)

        # Evaluation of the intensity of the current slice
        imax = max(sl_int)

        # Filling of the guess parameters arrays and of the bounds arrays
        p1 = [0,0,xc,yc,0.3,-0.2,2.2,0.1,0.2,1.,0.,imax]
        p2 = [sky]
        b1 = [None]*(11+signal2.nslice)
        b1[0:11] = [[None,None],[-pi,pi],[None, None],[None,None],[0.01,None],[-5.,0],[1.,None],\
                    [0,None],[0.01,None],[1.,None],[0.,pi]]
        b1[11:11+signal2.nslice] = [[0,None]]
        b2 = [[0.,None]]
        p = [p1,p2]
        b = [b1,b2]

        # Instanciating of a model class
        lbda_ref = signal2.lbda[0]
        f = ['SNIFS_psf_3D;0.42,%f'%(lbda_ref),'poly2D;0']
        sl_model = pySNIFS_fit.model(data=signal2,var=var2,param=p,func=f,bounds=b)

        # Fit of the current slice
        if opts.verbosity_level == 2:
            sl_model.fit(maxfun=200,msge=1)
        else:
            sl_model.fit(maxfun=200)

        # Storing the result of the current slice parameters
        xc_vec[i] = sl_model.fitpar[2]
        yc_vec[i] = sl_model.fitpar[3]
        sigc_vec[i] = sl_model.fitpar[4]
        q_vec[i] = sl_model.fitpar[6]
        eps_vec[i] = sl_model.fitpar[7]
        sigk_vec[i] = sl_model.fitpar[8]
        qk_vec[i] = sl_model.fitpar[9]
        theta_vec[i] = sl_model.fitpar[10]
        int_vec[i] = sl_model.fitpar[11]
        sky_vec[i] = sl_model.fitpar[12]


    ##### 3D model fitting #####
        
    # Computing the initial guess for the 3D fitting from the results of the slic by slice 2D fit
    lbda_ref = numarray.array(signal.lbda).mean()
    #1) Position parameters:
    #   the xc,yc vectors obtained from 2D fit are smoothed, then the position corresponding to the reference wavelength is read in the filtered
    #   vectors. Finally, the parameters theta and alpha are determined from the xc,yc vectors.
    xc_vec = N.filters.median_filter(xc_vec,5)
    yc_vec = N.filters.median_filter(yc_vec,5)
    x0 = xc_vec[numarray.argmin(numarray.abs(lbda_ref - signal.lbda))]
    y0 = yc_vec[numarray.argmin(numarray.abs(lbda_ref - signal.lbda))]
    n_ref = 1e-6*(64.328 + 29498.1/(146.-1./(lbda_ref*1e-4)**2) + 255.4/(41.-1./(lbda_ref*1e-4)**2)) + 1.
    ADR_coef = numarray.array(206265*(1e-6*(64.328 + 29498.1/(146.-1./(signal.lbda*1e-4)**2) + 255.4/(41.-1./(signal.lbda*1e-4)**2)) + 1. - n_ref))
    theta = scipy.arctan(scipy.median(numarray.compress(numarray.logical_not(scipy.isnan((yc_vec-y0)/(xc_vec-x0))),(yc_vec-y0)/(xc_vec-x0))))

    alpha_x_vec = compress(ADR_coef!=0,(xc_vec-x0)/(numarray.cos(theta)*ADR_coef))
    alpha_y_vec = compress(ADR_coef!=0,(yc_vec-y0)/(numarray.sin(theta)*ADR_coef))
    if theta == 0:
        alpha = scipy.median(alpha_x_vec)
    elif theta == pi/2.:
        alpha = scipy.median(alpha_y_vec)
    else:
        alpha_x = scipy.median(alpha_x_vec)
        alpha_y = scipy.median(alpha_y_vec)
        alpha = scipy.mean([alpha_x,alpha_y])

    #2) Other parameters:
    sigc = scipy.median(sigc_vec*(signal.lbda/lbda_ref)*0.2)
    q = scipy.median(q_vec)
    qk = scipy.median(qk_vec)
    eps = scipy.median(eps_vec)
    sigk = scipy.median(sigk_vec)
    theta_k = scipy.median(theta_vec)

    # Filling the guess solution and bounds arrays
    p1 = [None]*(11+signal.nslice)
    b1 = [None]*(11+signal.nslice)
    
    p1[0:11] = [alpha,theta,x0,y0,sigc,-0.2,q,eps,sigk,qk,theta_k]
    b1[0:11] = [[None,None],[-pi,pi],[None, None],[None,None],[0.01,None],[-5.,0],[1.,None],\
                [0,None],[0.01,None],[1.,None],[0.,pi]]
    p1[11:11+signal.nslice] = int_vec.tolist()
    b1[11:11+signal.nslice] = [[0,None] for i in range(signal.nslice)]
    p2 = sky_vec.tolist()
    b2 = [[0.,None] for i in range(signal.nslice)]
    p = [p1,p2]
    b = [b1,b2]
    f = ['SNIFS_psf_3D;0.42,%f'%(lbda_ref),'poly2D;0']

    # Instanciating the model class
    data_model = pySNIFS_fit.model(data=signal,var=var,param=p,func=f,bounds=b)
    
    print_msg("Fitting the data...",opts.verbosity_level,0)
    
    # The fit is launched twice. This is a dirty trick to avoid it to get quickly stuck on a bad solution... 
    if opts.verbosity_level == 2:
        data_model.fit(maxfun=200,save=True,msge=1) 
        data_model.fit(msge=1)
    else:
        data_model.fit(maxfun=200,save=True) 
        data_model.fit()
        
    # Storing result and guess parameters
    fitpar = data_model.fitpar
    guesspar = data_model.flatparam
    print_msg("Extracting the spectrum...",opts.verbosity_level,0)

    # Computing the final spectra for the object and the background
    spec = comp_spec(opts.incube,data_model.fitpar[0:11],intpar=[0.42,lbda_ref])
    # The 3D psf model is not normalized to 1 in integral. The result must be renormalized by (1+eps)
    spec[1] = spec[1] * (data_model.fitpar[7] + 1)
    
    # Create png images of spectra and slices fit
    isplot = opts.isplot
    
    if isplot:
        plot1 = os.path.splitext(opts.outspec)[0]+"_plt.png"
        plot2 = os.path.splitext(opts.outspec)[0]+"_fit1.png"
        plot3 = os.path.splitext(opts.outspec)[0]+"_fit2.png"
        plot4 = os.path.splitext(opts.outspec)[0]+"_fit3.png"
        plot5 = os.path.splitext(opts.outspec)[0]+"_fit4.png"
        plot6 = os.path.splitext(opts.outspec)[0]+"_fit5.png"
        
        # Plot of the star and sky spectra
        
        pylab.figure()
        pylab.subplot(2,1,1)
        pylab.plot(spec[0],spec[1])
        pylab.title("Star spectrum")
        pylab.subplot(2,1,2)
        pylab.plot(spec[0],spec[2])
        pylab.title("Background spectrum")
        pylab.savefig(plot1,dpi=150,facecolor='w',edgecolor='w',orientation='portrait')
        pylab.close()

        # Plot of the fit on each slice
        pylab.figure()
        ncol = numarray.floor(sqrt(signal.nslice))
        nrow = numarray.ceil(float(signal.nslice)/float(ncol))
        for i in range(signal.nslice):                 
            pylab.subplot(nrow,ncol,i+1)
            data = data_model.data.data[i,:]
            fit = data_model.evalfit()[i,:]
            imin = min((min(data),min(fit)))
            pylab.plot(data-imin+1e-2)
            pylab.plot(fit-imin+1e-2)
            pylab.semilogy()
            pylab.xticks(fontsize=4)
            pylab.yticks(fontsize=4)    
        pylab.savefig(plot2,dpi=150,facecolor='w',edgecolor='w',orientation='portrait')
        pylab.close()

        # Plot of the fit on rows and columns sum
        pylab.figure()
        # Creating a standard SNIFS cube with the fitted data
        cube_fit = pySNIFS.SNIFS_cube(lbda=signal.lbda)
        func1 = pySNIFS_fit.SNIFS_psf_3D(intpar=[data_model.func[0].pix,data_model.func[0].lbda_ref],cube=cube_fit)
        func2 = pySNIFS_fit.poly2D(0,cube_fit)
        cube_fit.data = func1.comp(fitpar[0:func1.npar]) + func2.comp(fitpar[func1.npar:func1.npar+func2.npar])
        print signal.nslice
        print numarray.shape(signal.data)
        for i in range(signal.nslice):                 
            pylab.subplot(nrow,ncol,i+1)
            pylab.plot(sum(signal.slice2d(i,coord='p'),0),'bo',markersize=3)
            pylab.plot(sum(cube_fit.slice2d(i,coord='p'),0),'b-')
            pylab.plot(sum(signal.slice2d(i,coord='p'),1),'r^',markersize=3)
            pylab.plot(sum(cube_fit.slice2d(i,coord='p'),1),'r-')
            pylab.xticks(fontsize=4)
            pylab.yticks(fontsize=4)    
        pylab.savefig(plot3,dpi=150,facecolor='w',edgecolor='w',orientation='portrait')
        pylab.close()
        
        # Plot of the star center of gravity and fitted center 
        sky = numarray.array(fitpar[11+signal.nslice:])
        pylab.figure()
        ax = pylab.subplot(2,1,2)
        pylab.plot(xc_vec,yc_vec,'bo')
        xfit = fitpar[0]*data_model.func[0].ADR_coef[:,0]*numarray.cos(fitpar[1]) + fitpar[2]
        yfit = fitpar[0]*data_model.func[0].ADR_coef[:,0]*numarray.sin(fitpar[1]) + fitpar[3]
        xguess = guesspar[0]*data_model.func[0].ADR_coef[:,0]*numarray.cos(guesspar[1]) + guesspar[2]
        yguess = guesspar[0]*data_model.func[0].ADR_coef[:,0]*numarray.sin(guesspar[1]) + guesspar[3]
        pylab.plot(xfit,yfit)
        pylab.plot(xguess,yguess,'k--')
        pylab.legend(("Fitted 2D PSF center","Guess 3D PSF center","Fitted 3D PSF center"),loc='lower right')
        pylab.text(0.05,0.9,r'$\rm{Guess:}\hspace{1} x_{0} = %4.2f\hspace{0.5} y_{0} = %4.2f \hspace{0.5} \alpha = %5.2f \hspace{0.5} \theta = %6.2f^o$'%(x0,y0,alpha,theta*180/scipy.pi),transform=ax.transAxes)
        pylab.text(0.05,0.8,r'$\rm{Fit:}\hspace{1} x_{0} = %4.2f\hspace{0.5} y_{0} = %4.2f\hspace{0.5} \alpha = %5.2f \hspace{0.5} \theta = %6.2f^o$'%(fitpar[2],fitpar[3],fitpar[0],fitpar[1]*180/scipy.pi),transform=ax.transAxes)
        pylab.xlabel("X center",fontsize=8)
        pylab.ylabel("Y center",fontsize=8)
        pylab.xticks(fontsize=8)
        pylab.yticks(fontsize=8)
        pylab.subplot(2,2,1)
        pylab.plot(signal.lbda,xc_vec,'bo')
        pylab.plot(signal.lbda,xfit)
        pylab.plot(signal.lbda,xguess,'k--')
        pylab.xlabel("Wavelength",fontsize=8)
        pylab.ylabel("X center",fontsize=8)
        pylab.xticks(fontsize=8)
        pylab.yticks(fontsize=8)
        pylab.subplot(2,2,2)
        pylab.plot(signal.lbda,yc_vec,'bo')
        pylab.plot(signal.lbda,yfit)
        pylab.plot(signal.lbda,yguess,'k--')
        pylab.xlabel("Wavelength",fontsize=8)
        pylab.ylabel("Y center",fontsize=8)
        pylab.xticks(fontsize=8)
        pylab.yticks(fontsize=8)
        pylab.savefig(plot4,dpi=150,facecolor='w',edgecolor='w',orientation='portrait')
        pylab.close()

        # Plot of the dispersion, fitted core dispersion and theoretical dispersion
        sky = numarray.array(fitpar[11+signal.nslice:])[:,NewAxis]
        pylab.figure()
        ax = pylab.subplot(2,1,2)       
        core_disp = fitpar[4]*(signal.lbda/lbda_ref)**fitpar[5]
        guess_disp = guesspar[4]*(signal.lbda/lbda_ref)**guesspar[5]
        th_disp = fitpar[4]*(signal.lbda/lbda_ref)**(-0.2)
        pylab.plot(signal.lbda,core_disp)
        pylab.plot(signal.lbda,th_disp,'b--')
        pylab.legend(("Sigma core (Model)","Sigma core (Theoretical)"))
        pylab.xlabel("Wavelength",fontsize=8)
        pylab.ylabel(r'$\sigma_c$',fontsize=15)
        pylab.xticks(fontsize=8)
        pylab.yticks(fontsize=8)
        pylab.subplot(2,1,1)
        pylab.plot(signal.lbda,sigc_vec,'bo')
        pylab.plot(signal.lbda,guess_disp,'k--')
        pylab.plot(signal.lbda,core_disp)
        pylab.legend(("2D model","3D Model guess","3D Model fit"))
        pylab.xlabel("Wavelength",fontsize=8)
        pylab.ylabel(r'$\sigma_c$',fontsize=15)
        pylab.xticks(fontsize=8)
        pylab.yticks(fontsize=8)
        pylab.text(0.05,0.9,r'$\rm{Guess:}\hspace{1} \sigma_{c} = %4.2f \hspace{0.5} \gamma = -0.2$'%(sigc),transform=ax.transAxes)
        pylab.text(0.05,0.8,r'$\rm{Fit:} \hspace{1} \sigma_{c} = %4.2f \hspace{0.5} \gamma = %4.2f$'%(fitpar[4],fitpar[5]),transform=ax.transAxes)
        pylab.savefig(plot5,dpi=150,facecolor='w',edgecolor='w',orientation='portrait')
        pylab.close()

        # Plot of the other model parameters
        pylab.figure()
        ax = pylab.subplot(5,1,1)
        plot_non_chromatic_param(ax,q_vec,signal.lbda,q,fitpar[6],'q')
        ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        ax = pylab.subplot(5,1,2)
        plot_non_chromatic_param(ax,eps_vec,signal.lbda,eps,fitpar[7],'\\epsilon')
        ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        ax = pylab.subplot(5,1,3)
        plot_non_chromatic_param(ax,sigk_vec,signal.lbda,sigk,fitpar[8],'\\sigma_k')
        ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        ax = pylab.subplot(5,1,4)
        plot_non_chromatic_param(ax,qk_vec,signal.lbda,qk,fitpar[9],'q_k')
        ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        ax = pylab.subplot(5,1,5)
        plot_non_chromatic_param(ax,theta_vec,signal.lbda,theta_k,fitpar[10],'\\theta_k')
        pylab.xlabel('Wavelength',fontsize=15)
        pylab.savefig(plot6,dpi=150,facecolor='w',edgecolor='w',orientation='portrait')
        pylab.close()
        
        

    # Save star spectrum

    hdu = pyfits.PrimaryHDU()
    hdu.data = numarray.array(spec[1])
    hdu.header.update('NAXIS', 1)
    hdu.header.update('NAXIS1', len(spec[1]),after='NAXIS')
    hdu.header.update('CRVAL1', spec[0][0])
    hdu.header.update('CDELT1', cube[1].header.get('CDELTS'))
    for desc in cube[1].header.items():
        if desc[0][0:5] != 'TUNIT' and desc[0][0:5] != 'TTYPE' and desc[0][0:5] != 'TFORM' and \
               desc[0][0:5] != 'TDISP' and desc[0] != 'EXTNAME' and desc[0] != 'XTENSION' and \
               desc[0] != 'GCOUNT' and desc[0] != 'PCOUNT' and desc[0][0:5] != 'NAXIS' and \
               desc[0] != 'BITPIX' and desc[0] != 'CTYPES' and desc[0] != 'CRVALS' and \
               desc[0] != 'CDELTS' and desc[0] != 'CRPIXS':
            hdu.header.update(desc[0],desc[1])
    hdulist = pyfits.HDUList()
    hdulist.append(hdu)
    if os.path.isfile(opts.outspec):
        os.system('rm %s'%opts.outspec)
    hdulist.writeto(opts.outspec)

    # Save sky spectrum

    hdu = pyfits.PrimaryHDU()
    hdu.data = numarray.array(spec[2])
    hdu.header.update('NAXIS', 1)
    hdu.header.update('NAXIS1', len(spec[1]),after='NAXIS')
    hdu.header.update('CRVAL1', spec[0][0])
    hdu.header.update('CDELT1', cube[1].header.get('CDELTS'))
    for desc in cube[1].header.items():
        if desc[0][0:5] != 'TUNIT' and desc[0][0:5] != 'TTYPE' and desc[0][0:5] != 'TFORM' and \
               desc[0][0:5] != 'TDISP' and desc[0] != 'EXTNAME' and desc[0] != 'XTENSION' and \
               desc[0] != 'GCOUNT' and desc[0] != 'PCOUNT' and desc[0][0:5] != 'NAXIS' and \
               desc[0] != 'BITPIX' and desc[0] != 'CTYPES' and desc[0] != 'CRVALS' and \
               desc[0] != 'CDELTS' and desc[0] != 'CRPIXS':
            hdu.header.update(desc[0],desc[1])
    hdulist = pyfits.HDUList()
    hdulist.append(hdu)
    if os.path.isfile(opts.outsky):
        os.system('rm %s'%opts.outsky)
    hdulist.writeto(opts.outsky)

