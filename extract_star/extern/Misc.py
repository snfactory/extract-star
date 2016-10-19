# copied from Toolbox on 19 Oct 2016 with no modifications

"""
.. _Misc:

ToolBox.Misc - Miscellaneous utilities
======================================
"""

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"
__version__ = "$Revision: 1.6 $"


def make_method(obj):
    """
    Decorator to make the function a method of *obj*, e.g.
    ::

      @make_method(MyClass)
      def func(myClassInstance, ...):
          ...

    makes *func* a method of `MyClass`, so that one can directly use::

      myClassInstance.func()
    """

    def decorate(f):
        setattr(obj, f.__name__, f)
        return f

    return decorate


def add_attrs(**kwargs):
    """
    Decorator adding attributes to a function, e.g.
    ::

      @attrs(source='NIST/IAPWS')
      def func(...):
          ...
    """

    def decorate(f):
        for key, val in kwargs.iteritems():
            setattr(f, key, val)
        return f

    return decorate


def deprecated(func):
    """
    This is a decorator which can be used to mark functions as deprecated.  It
    will result in a warning being emitted when the function is used.

    Source: https://wiki.python.org/moin/PythonDecoratorLibrary
    """

    import warnings, functools

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % (func.__name__),
                      category=DeprecationWarning,
                      filename=func.__code__.co_filename,
                      lineno=func.__code__.co_firstlineno + 1)
        return func(*args, **kwargs)

    return new_func


def cached_property(func):
    """
    A memoize decorator for class properties: the property will be computed
    only once, and cached for latter use.

    Source: http://code.activestate.com/recipes/576563-cached-property/
    """

    import functools

    @functools.wraps(func)
    def get(self):
        try:
            return self._cache[func]
        except AttributeError:
            self._cache = {}
        except KeyError:
            pass
        ret = self._cache[func] = func(self)
        return ret

    return property(get)


def warning2stdout(message, category, filename, lineno, file=None, line=None):
    """
    Alternative to default `warnings.showwarning`, to print out warning
    messages to stdout rather than stderr.::

      import warnings
      warnings.showwarning = warning2stdout
    """

    import sys, warnings

    sys.stdout.write(
        "WARNING: " +
        warnings.formatwarning(message, category, filename, lineno))


def catch(func, handle=lambda e: e, *args, **kwargs):
    """
    Exception-catching function.

    >>> eggs = (1, 3, 0)
    >>> [ catch(lambda: 1/i) for i in range(3) ]
    [('integer division or modulo by zero'), 1, 0]

    Source: http://stackoverflow.com/questions/1528237/
    """
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return handle(e)
