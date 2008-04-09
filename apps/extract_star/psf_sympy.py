#!/usr/bin/env python

from sympy import *

r = Symbol('r')
a = Symbol('alpha')

def gaussian(x, s):
    return exp(-(x/s)**2/2)

def moffat(x, a, b):
    return (1 + (x/a)**2)**(-b)

def PSF_explicit(r, a):
    """PSF description including explicit linear correlation between alpha and
    other parameters beta,eta,sigma."""

    def sigma(a):
        s0 = Symbol('sigma_0')
        s1 = Symbol('sigma_1')
        return s0 + s1*a

    def beta(a):
        b0 = Symbol('beta_0')
        b1 = Symbol('beta_1')
        return b0 + b1*a

    def eta(a):
        n0 = Symbol('eta_0')
        n1 = Symbol('eta_1')
        return n0 + n1*a

    s = sigma(a)                        # Explicit function of alpha
    b = beta(a)
    n = eta(a)

    return moffat(r,a,b) + n*gaussian(r,s)

def PSF(r, a):
    """PSF description with implicit correlation between alpha and
    other parameters beta,eta,sigma."""

    s = Function('sigma')               # Implicit function of alpha
    b = Function('beta')
    n = Function('eta')

    return moffat(r,a,b(a)) + n(a)*gaussian(r,s(a))

print "PSF model:"
pprint(PSF(r,a))
print "PSF derivative wrt. alpha (implicit correlations):"
pprint(PSF(r,a).diff(a))
print "PSF derivative wrt. alpha (explicit linear correlations):"
pprint(PSF_explicit(r,a).diff(a))

    
    
