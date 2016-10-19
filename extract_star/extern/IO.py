# copied one function (str_magn) from ToolBox.IO on 19 Oct 2016

def str_magn(x, dx=None, digits=2, signed=False):
    """Return properly formatted value and associated error as strings.

    :param float x: value
    :param float dx: associated error
    :param int digits: number of *significant* digits
    :param bool signed: value will always be signed if True
    :return: 2-tuple of formatted strings
             (or single formatted string if dx is null)

    >>> print ' +/- '.join(str_magn(255.323, 23.12856))
    255 +/- 24
    >>> print ' +/- '.join(str_magn(255.323, 23.12856, digits=3))
    255.3 +/- 23.2
    >>> print str_magn(255.323, digits=2) # Keep 2 significant digits
    260
    """

    from math import ceil

    if dx is None:                      # Do not return error component
        if not x:                       # '0'
            return '0'
        _dx = 0
    else:                               # Return error component
        _dx = dx
        if not x and not _dx:           # '0' +/- '0'
            return '0','0'

    if _dx: # Non-null error: format error with requested significant digits
        ndx = digits - order_magn(_dx) - 1  # Nb of digits after dec. point
        rdx = ceil(_dx * 10**ndx)/10**ndx   # Round-up error
        #print "dx",_dx,"ndx",ndx,"rdx",rdx,"order_magn(rdx)",order_magn(rdx)
        sdx = '%#.*g' % (max(digits, order_magn(rdx) + 1), rdx) # Error string
    else:   # Null error: format value with requested significant digits
        ndx = digits - order_magn(x) - 1
        sdx = '%#.*g' % (ndx + 1, 0)      # Error string

    # Format value accordingly
    nx = order_magn(x) + ndx               # Total number of digits
    rx = round(x, ndx)                     # Rounded value
    fmt = "%#+.*g" if signed else "%#.*g"  # Keep trailing zeros
    #print "x",x,"nx",nx,"rx",rx,"order_magn(rx)",order_magn(rx)
    sx = fmt % (max(nx + 1, order_magn(rx) + 1), rx) # Value string

    return (sx,sdx) if dx is not None else sx
