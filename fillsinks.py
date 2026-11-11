"""
fillsinks.py
------------
Python translation of TopoToolbox (v1) fillsinks.m
(W. Schwanghart, 8 October 2009): fill pits / sinks / depressions in a DEM.

Only the one-argument form fillsinks(dem) is used by the model:

    NaN -> -inf;  demfs = imfill(dem, 8, 'holes');  NaN restored

MATLAB imfill(I, 8, 'holes') on a GRAYSCALE image raises every regional
minimum not connected to the image border to its spill level. It is exactly
grayscale morphological reconstruction by erosion with 8-connectivity, using
a marker equal to I on the border and max(I) inside. skimage's
reconstruction() implements the same exact algorithm. Both only copy existing
elevation values (no arithmetic), so the filled DEM is bit-identical.

MATLAB-specific detail: NaN cells are set to -inf before filling.
  * NaN areas CONNECTED TO THE GRID BORDER act as outlets (water drains
    into the off-ice area).
  * NaN areas ENCLOSED by ice (not touching the border) are NOT outlets:
    imfill treats them as holes and fills them to their spill level, so the
    surrounding ice drains past them. (Checked: this behaviour is reproduced.)
skimage cannot take -inf, so these cells are replaced by a finite value
strictly below every finite elevation. Because a filled value only depends on
the ORDER of elevations along spill paths, the result for every non-NaN cell
is unchanged. NaN cells are restored afterwards, as in MATLAB.

The two-argument form (maxdepth) is not used by the model and not translated.
"""
import numpy as np
from skimage.morphology import reconstruction


def _imfill_holes_grayscale(I):
    I = np.asarray(I, dtype=float)
    neg_inf = np.isneginf(I)
    finite = np.isfinite(I)
    if not finite.any():
        return I.copy()
    work = I.copy()
    lo = np.min(I[finite]) - 1.0
    if not lo < np.min(I[finite]):          # guard for huge magnitudes
        lo = np.nextafter(np.min(I[finite]), -np.inf)
    work[neg_inf] = lo

    if work.shape[0] <= 2 or work.shape[1] <= 2:
        out = work.copy()                   # every pixel is a border pixel
    else:
        seed = work.copy()
        seed[1:-1, 1:-1] = work.max()
        out = reconstruction(seed, work, method='erosion',
                             footprint=np.ones((3, 3), dtype=bool))
    out[neg_inf] = -np.inf
    return out


def fillsinks(dem, maxdepth=None):
    if maxdepth is not None:
        raise NotImplementedError('fillsinks(dem, maxdepth) is not used by the model '
                                  'and has not been translated')
    dem = np.array(dem, dtype=float, copy=True)
    Inan = np.isnan(dem)                    # identify nans
    dem[Inan] = -np.inf                     # set nans to -inf
    demfs = _imfill_holes_grayscale(dem)    # imfill(dem,8,'holes')
    demfs[Inan] = np.nan                    # set NaN again
    return demfs
