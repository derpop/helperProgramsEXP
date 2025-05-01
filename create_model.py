#!/usr/bin/env python3

"""
an example script to take the GSE model from Naidu+ (2021) and create a basis.
see the paper
https://iopscience.iop.org/article/10.3847/1538-4357/ac2d2d

the example requires the snapshot, obtainable from:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UFVSTH


"""

# pyEXP
import pyEXP

# some basic imports
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

# the Naidu model comes in fits format. sorry!
from astropy.io import fits



def return_density(logr,weights=1.,rangevals=[-2, 6],bins=500,d2=False):
    """return_density

    simple binned density using logarithmically spaced bins

    inputs
    ---------
    logr        : (array) log radii of particles to bin
    weights     : (float or array) if float, single-mass of particles, otherwise array of particle masses
    rangevals   : (two value list) minimum log r, maximum log r
    bins        : (int) number of bins
    d2          : (bool) if True, compute surface density

    returns
    ---------
    rcentre     : (array) array of sample radii (NOT LOG)
    density     : (array) array of densities sampled at rcentre (NOT LOG)

    """

    # assume evenly spaced logarithmic bins
    dr      = (rangevals[1]-rangevals[0])/bins
    rcentre = np.zeros(bins)
    density = np.zeros(bins)

    # check if single mass, or an array of masses being passed
    # construct array of weights
    if isinstance(weights,float):
        w = weights*np.ones(logr.size)
    else:
        w = weights

    for indx in range(0,bins):

        # compute the centre of the bin (log r)
        rcentre[indx] = rangevals[0] + (indx+0.5)*dr

        # compute dr (not log)
        rmin,rmax = 10.**(rangevals[0] + (indx)*dr),10.**(rangevals[0] + (indx+1)*dr)
        if d2:
            shell = np.pi*(rmax**2-rmin**2)
        else:
            shell = (4./3.)*np.pi*(rmax**3.-rmin**3.)

        # find all particles in bin
        inbin = np.where((logr>=(rangevals[0] + (indx)*dr)) & (logr<(rangevals[0] + (indx+1)*dr)))

        # compute M/V for the bin
        density[indx] = np.nansum(w[inbin])/shell

    # return
    return 10.**rcentre,density




def makemodel_empirical(rvals,dvals,pfile='',plabel = '',verbose=True):
    """make an EXP-compatible spherical basis function table

    inputs
    -------------
    rvals       : (array of floats) radius values to evaluate the density function
    pfile       : (string) the name of the output file. If '', will not print file
    plabel      : (string) comment string, printed to the top of the file
    verbose     : (boolean)

    outputs
    -------------
    R           : (array of floats) the radius values
    D           : (array of floats) the density
    M           : (array of floats) the mass enclosed
    P           : (array of floats) the potential

    """
    M = 1.
    R = np.nanmax(rvals)

    # query out the density values
    #dvals = D#func(rvals,*funcargs)
    #print(R.size,)

    # make the mass and potential arrays
    mvals = np.zeros(dvals.size)
    pvals = np.zeros(dvals.size)
    pwvals = np.zeros(dvals.size)

    # initialise the mass enclosed an potential energy
    mvals[0] = 1.e-15
    pwvals[0] = 0.

    # evaluate mass enclosed and potential energy by recursion
    for indx in range(1,dvals.size):
        mvals[indx] = mvals[indx-1] +\
          2.0*np.pi*(rvals[indx-1]*rvals[indx-1]*dvals[indx-1] +\
                 rvals[indx]*rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);
        pwvals[indx] = pwvals[indx-1] + \
          2.0*np.pi*(rvals[indx-1]*dvals[indx-1] + rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);

    # evaluate potential (see theory document)
    pvals = -mvals/(rvals+1.e-10) - (pwvals[dvals.size-1] - pwvals)

    # get the maximum mass and maximum radius
    M0 = mvals[dvals.size-1]
    R0 = rvals[dvals.size-1]

    # compute scaling factors
    Beta = (M/M0) * (R0/R);
    Gamma = np.sqrt((M0*R0)/(M*R)) * (R0/R);
    if verbose:
        print("! Scaling:  R=",R,"  M=",M)

    rfac = np.power(Beta,-0.25) * np.power(Gamma,-0.5);
    dfac = np.power(Beta,1.5) * Gamma;
    mfac = np.power(Beta,0.75) * np.power(Gamma,-0.5);
    pfac = Beta;

    if verbose:
        print(rfac,dfac,mfac,pfac)

    # save file if desired
    if pfile != '':
        f = open(pfile,'w')
        print('! ',plabel,file=f)
        print('! R    D    M    P',file=f)

        print(rvals.size,file=f)

        for indx in range(0,rvals.size):
            print('{0} {1} {2} {3}'.format( rfac*rvals[indx],\
              dfac*dvals[indx],\
              mfac*mvals[indx],\
              pfac*pvals[indx]),file=f)

        f.close()

    return rvals*rfac,dfac*dvals,mfac*mvals,pfac*pvals



def makemodel(func,M,funcargs,rvals = 10.**np.linspace(-2.,4.,2000),pfile='',plabel = '',verbose=True):
    """make an EXP-compatible spherical basis function table from a functional input

    inputs
    -------------
    func        : (function) the callable functional form of the density
    M           : (float) the total mass of the model, sets normalisations
    funcargs    : (list) a list of arguments for the density function.
    rvals       : (array of floats) radius values to evaluate the density function
    pfile       : (string) the name of the output file. If '', will not print file
    plabel      : (string) comment string, printed to the top of the file
    verbose     : (boolean)

    outputs
    -------------
    R           : (array of floats) the radius values
    D           : (array of floats) the density
    M           : (array of floats) the mass enclosed
    P           : (array of floats) the potential

    """

    R = np.nanmax(rvals)

    # query out the density values
    dvals = func(rvals,*funcargs)

    # make the mass and potential arrays
    mvals = np.zeros(dvals.size)
    pvals = np.zeros(dvals.size)
    pwvals = np.zeros(dvals.size)

    # initialise the mass enclosed an potential energy
    mvals[0] = 1.e-15
    pwvals[0] = 0.

    # evaluate mass enclosed and potential energy by recursion
    for indx in range(1,dvals.size):
        mvals[indx] = mvals[indx-1] +\
          2.0*np.pi*(rvals[indx-1]*rvals[indx-1]*dvals[indx-1] +\
                 rvals[indx]*rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);
        pwvals[indx] = pwvals[indx-1] + \
          2.0*np.pi*(rvals[indx-1]*dvals[indx-1] + rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);

    # evaluate potential (see theory document)
    pvals = -mvals/(rvals+1.e-10) - (pwvals[dvals.size-1] - pwvals)

    # get the maximum mass and maximum radius
    M0 = mvals[dvals.size-1]
    R0 = rvals[dvals.size-1]

    # compute scaling factors
    Beta = (M/M0) * (R0/R);
    Gamma = np.sqrt((M0*R0)/(M*R)) * (R0/R);
    if verbose:
        print("! Scaling:  R=",R,"  M=",M)

    rfac = np.power(Beta,-0.25) * np.power(Gamma,-0.5);
    dfac = np.power(Beta,1.5) * Gamma;
    mfac = np.power(Beta,0.75) * np.power(Gamma,-0.5);
    pfac = Beta;

    if verbose:
        print(rfac,dfac,mfac,pfac)

    # save file if desired
    if pfile != '':
        f = open(pfile,'w')
        print('! ',plabel,file=f)
        print('! R    D    M    P',file=f)

        print(rvals.size,file=f)

        for indx in range(0,rvals.size):
            print('{0} {1} {2} {3}'.format( rfac*rvals[indx],\
              dfac*dvals[indx],\
              mfac*mvals[indx],\
              pfac*pvals[indx]),file=f)

        f.close()

    return rvals*rfac,dfac*dvals,mfac*mvals,pfac*pvals


def twopower_density_withrolloff(r,a,alpha,beta,rcen,wcen):
    """a twopower density profile"""
    ra = r/a
    prefac = 0.5*(1.-scipy.special.erf((ra-rcen)/wcen))
    return prefac*(ra**-alpha)*(1+ra)**(-beta+alpha)

# Using Einasto density distribution
def twopower_einasto_density(r, R, r_core, r_scale, w_scale, alpha):
    power = (R/(r+r_core))**alpha
    ex = np.exp(-(r/r_scale)**w_scale)
    return power*ex

def hernquist_profile_density(r, r_scale, alpha):
    return 1/(r/r_scale * (1 + r/r_scale)**alpha)

def plummer_profile_density(r, plummer_radius):
    return (1+r**2/plummer_radius**2)**-2.5

#Reads data from a .fits file
'''
# read in the data
fits_image_filename = 'GSEz0snapshot.fits'
hdul = fits.open(fits_image_filename)

# all we really care about from this is hdul[1].data['X'],hdul[1].data['Y'],hdul[1].data['Z']
# compute 3d radius
xpos = hdul[1].data['X']
ypos = hdul[1].data['Y']
zpos = hdul[1].data['Z']
mass = np.ones(xpos.size)
Rgse = np.sqrt((xpos)**2.+(ypos)**2.+((zpos)**2.))
print(np.nanmin(Rgse),np.nanmax(Rgse))

# compute the density. note that the masses are all equal (1.) and the radius bins are given in log space, so we will get 1-100 kpc
rbins,dreturn = return_density(np.log10(Rgse),1.,rangevals=[0., 2.],bins=100)

# for both models below, if you want to save the file, simply uncomment the pfile instance.
# call the empirical model maker
R,D,M,P = makemodel_empirical(rbins,dreturn,pfile='GSEbasis_empirical.txt')
'''

#Milky Way Halo
r_scale = 16
alpha = 2
#note: linspace arg 2 is log(max radius)
R,D,M,P = makemodel(hernquist_profile_density,1965914.,[r_scale, alpha],rvals = 10.**np.linspace(0.,2.477,4000),pfile='MW_halo_NFW.model')

#Milky Way Bulge
r_scale = 0.442
alpha = 3
R,D,M,P = makemodel(hernquist_profile_density,20244., [r_scale, alpha],rvals = 10.**np.linspace(-2.,0.662, 2000),pfile='MW_bulge_Hernquist.model')

'''#Dwarf Galaxy for Orphen Stream
R = 0.8
r_core = 0.2 #bulge, not core
r_scale = 0.75
w_scale = 1/2.2
alpha = 2
#note: extra /10 added to account for <10 radius
r,m,d,p = makemodel(twopower_einasto_density,60000.,[R,r_core,r_scale,w_scale,alpha],rvals = 10.**np.linspace(0.,0.9,2000)/10,pfile='DW_halo.model')'''

#LMC
plummer_radius = 9.05
r,m,d,p = makemodel(plummer_profile_density,224933.,[plummer_radius],rvals = 10.**np.linspace(0.,1.898,2000),pfile='LMC_Plummer.model')
