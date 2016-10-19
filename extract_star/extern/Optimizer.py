#!/usr/bin/env python
# -*- coding: utf-8 -*-

# select functions and classes copied from ToolBox.Optimizer on 2016 Oct 19.
#
# Version:       $Revision: 1.42 $
# Description:   Test library
# Author:        Yannick Copin <yannick@ipnl.in2p3.fr>
# Author:        $Author: ycopin $
# Modified at:   Mon Mar 29 18:01:06 2010
# Modified by:   Yannick Copin <ycopin@ipnl.in2p3.fr>
# $Id: Optimizer.py,v 1.42 2015/06/18 15:21:16 ycopin Exp $
######################################################################

"""
.. _Optimizer:

ToolBox.Optimizer - Optimization-related routines
=================================================

Utility classes for optimization procedures from Scipy and pyMinuit.

Should be updated/completed/corrected/tested from other sources, including:

* http://www.phy.uct.ac.za/courses/python/examples/moreexamples.html
* http://scipy.org/scipy/scikits/wiki/OpenOpt
* http://code.google.com/p/pyminuit/

.. Todo::

   * implement robust regression using `Huber loss
     <https://en.wikipedia.org/wiki/Huber_loss>`_
"""

from __future__ import print_function

import numpy as N
import scipy.optimize as SO
import scipy.stats as SS
import scipy.linalg as SL                # Includes pinvh
import warnings

from .Misc import deprecated, cached_property
from .IO import str_magn

# Classes and functions ##############################

# 1st derivative central finite difference coefficients
# Ref: http://en.wikipedia.org/wiki/Finite_difference_coefficients
_CFDcoeffs = ((1 / 2.,),                                # 2nd-order
              (2 / 3., -1 / 12.,),                      # 4th-order
              (3 / 4., -3 / 20., 1 / 60.,),             # 6th-order
              (4 / 5., -1 / 5., 4 / 105., -1 / 280.,),  # 8th-order
              )


def approx_deriv(func, pars, dpars=None, eps=1e-6, order=3, args=()):
    """
    Function 1st derivative approximation using central finite differences.

    Hereafter, m=len(pars) and func returns a array of shape
    S=[m×[m×]]n.

    .. Note::

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
        coeffs = _CFDcoeffs[horder - 1]
    else:
        raise NotImplementedError("approx_deriv supports order up to 8/9")

    if dpars is None:
        dpars = N.zeros(len(pars)) + eps  # m
    mat = N.diag(dpars)                   # m×m diagonal matrix

    der = 0                               # Finite differences
    for i, c in enumerate(coeffs):        # Faster than N.sum(axis=0)
        der += c * N.array([(func(pars + (i + 1) * dpi, *args) -
                             func(pars - (i + 1) * dpi, *args))
                            for dpi in mat])  # m×S
    # der = N.sum([ [ c*(func(pars+(i+1)*dpi,*args)-func(pars-(i+1)*dpi,*args))
    # for dpi in mat ]
    # for i,c in enumerate(coeffs) ], axis=0)

    # func actually returns a scalar (n=0)
    if der.ndim == 1:
        der /= dpars                  # m×0 / m = m
    else:                             # func returns an array of shape S
        der /= dpars[..., N.newaxis]  # m×S / m×1 = S

    return der                        # S


def vec2corr(vec):
    """
    Define n×n correlation matrix from *vec* of length n, such that corr[0,:] =
    corr[:,0] = vec.
    """

    assert N.ndim(vec) == 1

    n = len(vec)
    tmp = N.concatenate((vec[:0:-1], vec))
    corr = N.array([tmp[n - (i + 1):2 * n - (i + 1)] for i in range(n)])

    return corr


def cov2corr(cov):
    """
    Compute 2D correlation matrix from input (2D) covariance matrix *cov*
    ::

      corr = cov / sqrt(outer(diag,diag))
    """

    sig = N.sqrt(N.diagonal(cov))       # Square root of main diagonal
    return cov / N.outer(sig, sig)


def corr2cov(corr, sig):
    """
    Compute 2D covariance matrix from input correlation matrix *corr* and
    (diagonal) standard error *sig*
    ::

      cov = corr * outer(sig,sig)
    """

    return corr * N.outer(sig, sig)


class DataSet(object):

    """Description of a dataset (length n)."""

    def __init__(self, y, dy=None, cov=None, x=None):
        """Dataset.

        :param y: input 1D dataset (length n)
        :param dy: (optional) standard error on *y*
        :param cov: (optional) full covariance matrix on *y*, or None
        :param x: (optional) associated independant variable
        """

        self.y = N.asarray(y)   # n
        self.n = len(self.y)

        self.hasError = (dy is not None)
        self.hasCovariance = (cov is not None)

        # Complain if both covariance and errors are specified
        if self.hasError and self.hasCovariance:
            raise ValueError("Cannot specify both dy and cov.")

        if self.hasError:               # Uncorrelated (diagonal) errors
            assert len(dy) == self.n, "y and dy don't share same length."
            self.dy = N.asarray(dy)

        elif self.hasCovariance:        # Covariance matrix
            if N.ndim(cov) == 1:        # Actually a covariance diagonal
                self.cov = N.diag(cov)
            else:
                self.cov = N.asarray(cov)  # n×n
            assert self.cov.shape == (self.n, self.n), \
                "y and cov don't share same length."
            self.dy = N.sqrt(N.diagonal(self.cov))

        if x is None:
            self.x = N.arange(self.n)   # Default independant variables
        else:
            assert len(x) == self.n, "x and y don't share same length."
            self.x = N.asarray(x)

    def __str__(self):

        s = "Dataset: %d points" % self.n
        if self.hasError:
            s += " with errors"
        elif self.hasCovariance:
            s += " with covariance"
        else:
            s += " without errors"

        return s

    def __getitem__(self, key):
        """Returns a sub-dataset selected on key."""

        return DataSet(self.y[key],
                       dy=self.dy[key] if self.hasError else None,
                       cov=self.cov[key, key] if self.hasCovariance else None,
                       x=self.x[key])

    @cached_property
    def var(self):
        """
        Inverse of the covariance matrix (so-called 'precision' matrix).
        """

        try:
            return SL.pinvh(self.cov)  # n×n
        except SL.LinAlgError as err:
            self.hasCovariance = False
            raise ValueError("Invalid covariance matrix (%s)" % str(err))
            # warnings.warn("DataSet: " \
            #               "Invalid covariance matrix (%s), " \
            #               "fall back to plain variance" % err)

    @cached_property
    def svar(self):
        """
        Cholesky decomposition of variance (inverse of covariance), such that
        var = dot(svar,svar.T).
        """

        try:
            return SL.cholesky(self.var, lower=True)  # n×n
        except SL.LinAlgError as err:
            self.hasCovariance = False
            raise ValueError("Invalid covariance matrix (%s)" % str(err))

    def residuals(self, model, explicitCov=True):
        """
        Error-weighted (model-data) residuals.

        :param model: model estimate (length n)
        :return residuals: (model - y) (if no error), (model - y)/dy
            (if `hasError`) or (model - y).T · svar (if
            `hasCovariance`)

        .. Note:: Residuals with covariance matrix

           One can compute (correlated) residuals in presence of a
           covariance matrix by using the Cholesky decomposition of
           the variance matrix (inverse of the covariance), such that
           `var = dot(svar,svar.T)`. However, this decomposition may
           raise a `LinAlgError: xx-th leading minor not positive
           definite` even though the covariance matrix is
           invertible. An explicit handling of the (co)variance matrix
           is therefore preferable, but does not allow the computation
           of correlated residuals.
        """

        res = model - self.y            # n
        if self.hasError:
            res /= self.dy              # n
        elif self.hasCovariance:
            if not explicitCov:
                res = N.dot(res, self.svar)  # n·(n×n) = n
            else:
                raise NotImplementedError(
                    "Correlated residuals cannot be computed in explicit case")

        return res

    def chi2(self, model, explicitCov=True):
        """chi2 = sum_i residuals**2."""

        if explicitCov:           # Explicit treatment of (co)variance
            res = model - self.y
            # Faster than res.dot(self.var).dot(res)
            return N.dot(N.dot(res, self.var), res)
        # Implicit treatment of (co)variance through correlated residuals
        else:
            res = self.residuals(model, explicitCov=False)  # n
            return N.dot(res, res)      # ~×5 faster than (res**2).sum()

    def correlation(self):
        """Return correlation matrix of input dataset."""

        if self.hasCovariance:
            corr = cov2corr(self.cov)      # n×n
        else:
            corr = N.eye(self.n)           # n×n

        return corr


class Model(object):

    """
    Description of a model ℝ**m → ℝ**n (m parameters modeling n observations)
    """

    def __init__(self, fn, jac=None, hess=None):
        """
        Model description.

        :param fn: model ℝ**m → ℝ**n
        :param jac: model jacobian ℝ**m → ℝ**m×ℝ**n (optional)
        :param hess: model hessian ℝ**m → ℝ**m×ℝ**m×ℝ**n (optional)

        Jacobian and hessian are approximated numerically using
        central finite differences if absent.
        """

        self.fn = fn                    # Model m → n

        # Jacobian m → m×n
        self.hasJac = (jac is not None)
        if self.hasJac:                 # Input jacobian
            self.jac = jac
        else:                           # Approximated jacobian
            self.jac = lambda pars, *args: \
                       approx_deriv(self.fn, pars, args=args)

        # Hessian m → m×m×n
        self.hasHess = (hess is not None)
        if self.hasHess:                # Input hessian
            self.hess = hess
        else:                           # Approximated hessian
            self.hess = lambda pars, *args: \
                        approx_deriv(self.jac, pars, args=args)

    def __str__(self):

        s = "Model: objfun='%s'" % self.fn.__name__
        if self.hasJac:
            s += ", jac='%s'" % self.jac.__name__
        else:
            s += ", no jacobian"
        if self.hasHess:
            s += ", hess='%s'" % self.hess.__name__
        else:
            s += ", no hessian"

        return s

    def __call__(self, pars, *args):

        return self.fn(pars, *args)

    # Gradient checks ==============================

    def check_jac(self, pars, args=(), verbose=False, **kwargs):
        """
        Check jacobian against the one approximated from function finite
        differences.
        """

        if self.hasJac:
            jac = self.jac(pars, *args)
            app = approx_deriv(self.fn, pars, args=args, **kwargs)
            ret = N.allclose(jac, app)
            if verbose:
                print("Model jacobian:", jac.shape)
                if verbose > 1:
                    print(jac)
                print("Approx. jacob.:", app.shape)
                if verbose > 1:
                    print(app)
        else:
            ret = True

        return ret

    def check_hess(self, pars, args=(), verbose=0, **kwargs):
        """Check hessian against the one approximated from jacobian
        finite differences."""

        if self.hasHess:
            hes = self.hess(pars, *args)
            app = approx_deriv(self.jac, pars, args=args, **kwargs)
            ret = N.allclose(hes, app)
            if verbose:
                print("Model hessian:  ", hes.shape)
                if verbose > 1:
                    print(hes)
                print("Approx. hessian:", app.shape)
                if verbose > 1:
                    print(app)
        else:
            ret = True

        return ret

    # Errors ==============================

    def covariance(self, pars, cov, *args):
        """
        Covariance matrix on model estimates (n×n) from parameters *pars* (m)
        and parameter covariance matrix *cov* (m×m)
        ::

          mcov = jac · cov · jac.T
        """

        jac = self.jac(pars, *args)           # Model jacobian m×n
        if N.ndim(cov) == 1:                  # Make it a m×m diagonal matrix
            cov = N.diag(cov)

        return N.dot(N.dot(jac.T, cov), jac)  # (n×m)·(m×m)·(m×n) = n×n

    def errors(self, pars, errors, *args):
        """
        Uncorrelated (diagonal) standard errors on model (n) from parameters
        *pars* (m) and associated errors *errors*.

        *errors* is either a 1D-vector of uncorrelated (diagonal)
        standard errors *dpars* (m)::

          df = sqrt( dpars**2 · jac**2 )

        or a full 2D-covariance matrix *cov* (m×m)::

          df = sqrt( (jac · cov · jac.T).diag )
        """

        if N.ndim(errors) == 1:           # errors = dpar
            ret = N.dot(errors ** 2, self.jac(pars, *args) ** 2)  # m·(m×n) = n
        elif N.ndim(errors) == 2:         # errors = cov(par)
            ret = self.covariance(pars, errors, *args).diagonal()

        return N.sqrt(ret)                # n


class Fitter(object):

    """
    A `Fitter` is a :class:`Model` tested against a :class:`DataSet`.
    """

    def __init__(self, model, dataset, hyper=None, explicitCov=True):
        """
        :param model: input :class:`Model`
        :param dataset: input :class:`DataSet`
        :param hyper: model hyper-term :class:`Model` (optional)
        :param explicitCov: (optional) explicit treatment of the
          (co)variance matrix. The implicit treatment relies on the
          additional Cholesky decomposition of the inverse of the
          covariance matrix (so-called the precision matrix).
        """

        assert isinstance(model, Model)
        assert isinstance(dataset, DataSet)

        self.model = model
        self.dataset = dataset
        self.explicitCov = self.dataset.hasCovariance and explicitCov

        # Hyper-term to be added to the chi2
        self.hyper = hyper
        if hyper:
            assert isinstance(hyper, Model)

        # Parameter selection (e.g. to discard parameters hitting the bounds)
        self.sel = None

    def __str__(self):

        s = 'Fitter ' + str(self.dataset)
        if self.dataset.hasCovariance:
            if self.explicitCov:
                s += " (explicitly managed)"
            else:
                s += " (implicitely managed through correlated residuals)"
        s += '\nFitter ' + str(self.model)
        if self.hyper:
            s += '\nFitter Hyper ' + str(self.hyper)

        return s

    # Residuals ==============================

    def residuals(self, pars, *args):
        """Residuals: res = (model - y)/dy"""

        return self.dataset.residuals(self.model.fn(pars, *args),
                                      explicitCov=self.explicitCov)  # n
