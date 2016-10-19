extract-star
============

Fit a 3-d PSF to a point source in a SNFactory data cube.

About
-----

This is an *experimental* git-hosted fork of the extract star code in CVS, mostly written by Yannick Copin. **The official extract star development remains in CVS** and any improvements for production should be submitted there. The purpose of this experiment is mainly for Kyle Barbary to experiment with integrating extract star and cubefit. There will be significant divergence from the original branch in terms of modernization and reorganization.

The code here was imported using `git cvsimport` from the `SNFactory/Offline/PySNIFS` CVS module (note the name change) on 18 Oct 2016. The most recent CVS commit at that time was:

```
commit f773b9e3b771d866582782f7f47d15abeee835d2
Author: ycopin <ycopin>
Date:   Thu Jun 23 13:17:20 2016 +0000

    * lib/libExtractStar.py (ExposurePSF.comp): error in normalization
    [#1558, thanks to K. Boone].
```
