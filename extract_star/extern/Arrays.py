# copied from ToolBox on 19 Oct 2016 with no modifications

"""
.. _Arrays:

ToolBox.Arrays - Numpy array utilities
======================================
"""

import numpy as N

__author__ = 'Rui Pereira <rui.pereira@in2p3.fr>'
__version__ = '$Revision: 1.11 $'


def unsqueeze(data, axis, oldshape=None):
    """
    Unsqueeze a collapsed array *data* previously squeezed along
    *axis*.

    :param ndarray data: input array
    :param integer axis: axis to be restored
    :param tuple oldshape: shape to be recovered (**deprecated**)
    :return: unsqueezed array

    >>> from numpy import arange, mean
    >>> x = arange(3*4*5).reshape((3,4,5))
    >>> m = mean(x, axis=1)
    >>> m.shape
    (3, 5)
    >>> m = unsqueeze(m, 1)
    >>> m.shape
    (3, 1, 5)

    .. Note:: this is now just a fairly empty wrapper to
              `numpy.expand_dims`, except for the axis=None case
              handling.
    """

    if axis is None:
        return data

    if oldshape:
        import warnings
        warnings.warn("unsqueeze: oldshape keyword is deprecated",
                      DeprecationWarning)

    return N.expand_dims(data, axis)


def count(arr, axis=None, keepdims=False):
    """
    Count the number of elements of array *arr* along *axis*. This is similar to
    ::

      N.ones_like(arr).sum(axis)

    with additional support for masked arrays and 'keepdims' option.
    """

    if N.ma.isMaskedArray(arr):
        counts = arr.count(axis=axis)
        if keepdims:            # N.ma.sum has no keepdims option (v1.11.1)
            counts = N.ma.expand_dims(counts, axis)
    else:
        counts = N.ones_like(arr, dtype=int).sum(axis, keepdims=keepdims)

    return counts


def rebin(arr, binfactor, fn=N.sum):
    """
    Rebin array *arr* by *binfactor* using array-function *fn*.

    :param array arr: input array
    :param int binfactor: binning factor, can be a tuple for (dim0, dim1)
    :param fn: numpy function `fn(arr, axis)` to be applied while rebinning
    :return: rebinned array
    :raises: `AssertionError` (input array must be 2D and binfactors
             must be multiples of input array dimensions)
    """

    assert arr.ndim == 2, 'rebin only works with 2D arrays'
    try:
        i, j = binfactor
    except TypeError:
        i = j = binfactor
    s = arr.shape
    assert (~N.mod(s, [i, j]).astype('bool')).all(), \
        'binfactors must be multiples of the array dimensions'
    arj = fn(arr.reshape(-1, j), 1).reshape(-1, s[1]/j).T
    ari = fn(arj.reshape(-1, i), 1).reshape(-1, s[0]/i).T

    return ari


def metaslice(alen, nmeta, trim=0, thickness=False):
    """
    Return *[imin,imax,istp]* so that `range(imin,imax+1,istp)` are the
    boundary indices for *nmeta* (centered) metaslices.

    Note: *imax* is the end boundary index of last metaslice,
          therefore element *imax* is *not* included in last
          metaslice.

    If *thickness* is True, *nmeta* is actually the thickness of
    metaslices. Non-null *trim* is the number of trimmed elements at
    each edge of the array.

    >>> a = range(0,141,10); a                        # 15 elements
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    >>> imin,imax,istp = metaslice(len(a), 3, trim=2) # Make 3 metaslices
    >>> imin,imax,istp
    (3, 12, 3)
    >>> ibounds = range(imin,imax+1,istp); ibounds
    [3, 6, 9, 12]
    >>> for i in xrange(len(ibounds)-1): print a[ibounds[i]:ibounds[i+1]]
    [30, 40, 50]
    [60, 70, 80]
    [90, 100, 110]
    >>> N.reshape(a[imin:imax],(-1,istp))
    array([[ 30,  40,  50],
           [ 60,  70,  80],
           [ 90, 100, 110]])
    """

    if alen <= 0 or nmeta <= 0 or trim < 0:
        raise ValueError("Invalid input (alen=%d>0, nmeta=%d>0, trim=%d>=0)" %
                         (alen, nmeta, trim))
    elif alen <= 2*trim:
        raise ValueError("Trimmed array would be empty")

    if thickness:
        istep = nmeta                    # Metaslice thickness
        nmeta = (alen - 2*trim) // istep  # Nb of metaslices
        if nmeta == 0:
            raise ValueError("Metaslice thickness is too big")
    else:
        istep = (alen - 2*trim) // nmeta  # Metaslice thickness

    if istep <= 0:
        raise ValueError("Null-thickness metaslices")

    # Center metaslices on (trimmed) array
    imin = trim + ((alen - 2*trim) % nmeta) // 2  # Index of 1st px of 1st slice
    imax = imin + nmeta*istep - 1                # Index of last px of last sl.

    return [imin, imax+1, istep]         # Return a list to please pySNIFS


def isTriangular(arr, lower=True):
    """
    Check if arr is a lower- (resp. upper-) triangular matrix, i.e. if all
    elements above (resp. below) the diagonal are null.
    """

    if lower:
        return (arr[N.triu_indices_from(arr, 1)] == 0).all()
    else:
        return (arr[N.tril_indices_from(arr, -1)] == 0).all()


def isDiagonal(arr):
    """
    Check if arr is a diagonal matrix, i.e. if all non-diagonal elements are
    null.
    """

    return isTriangular(arr, lower=True) and isTriangular(arr, lower=False)
