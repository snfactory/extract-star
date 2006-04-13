from distutils.core import setup
import sys

if not hasattr(sys, 'version_info') or sys.version_info < (2,3,0,'alpha',0):
    raise SystemExit, "Python 2.3 or later required to build pyfits."

def dolocal():
    """Adds a command line option --local=<install-dir> which is an abbreviation for
    'put all of pyfits in <install-dir>/pyfits'."""
    if "--help" in sys.argv:
        print >>sys.stderr
        print >>sys.stderr, " options:"
        print >>sys.stderr, "--local=<install-dir>    same as --install-lib=<install-dir>"
    for a in sys.argv:
        if a.startswith("--local="):
            dir = a.split("=")[1]
            sys.argv.extend([
                "--install-lib="+dir,
                ])
            sys.argv.remove(a)

def main():
    dolocal()
    setup(name = "pySNIFS",
          version = "1.0",
          description = "SNIFS data handling and processing package",
          author = "E. Pecontal",
          maintainer_email = "pecontal@obs.univ-lyon1.fr",
          platforms = ["Linux"],
          py_modules = ['pySNIFS', 'pySNIFS_fit'],
          package_dir={'':'lib'},
          scripts=['apps/extract_star/extract_star.py'])
          #packages=['pySNIFS','extract_star','cube_explorer'],
          #package_dir={'pySNIFS':'lib'})
 

if __name__ == "__main__":
    main()

