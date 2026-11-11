"""
ixneighbors.py
--------------
Python translation of TopoToolbox (v1) ixneighbors.m
(W. Schwanghart, 15 March 2009): neighbour indexing in a matrix.

    ic, icd = ixneighbors(A)            all neighbours of all cells
    ic, icd = ixneighbors(A, ix)        neighbours of cells in index vector ix
    ic, icd = ixneighbors(A, I)         neighbours of cells where logical I is True
    ic, icd = ixneighbors(..., conn)    conn = 4 or 8 (default 8)

ic, icd are 0-based MATLAB LINEAR (column-major) indices: ic = cell,
icd = its neighbour. NaN cells of A are discarded both in ic and icd.

OUTPUT ORDER IS REPRODUCED EXACTLY (it matters: routeflats picks the first
occurrence with unique()). The order is: neighbour direction outermost
  1 right (east, +nrows)   2 down (south, +1)     3 left (west, -nrows)
  4 up (north, -1)         5 up-right             6 up-left
  7 down-right             8 down-left
and, within each direction, source cells in column-major order.

MATLAB-specific notes
  * isnan() of a LOGICAL matrix is all False in MATLAB, so when A is
    logical (as in routeflats: ixneighbors(flats, boundaryf)) no cells are
    discarded. Reproduced by testing NaN on a float copy.
  * MATLAB's size check `if size(X) ~= size(ix)` only errors if BOTH
    dimensions differ (an `if` on a vector needs all elements true); a
    one-dimension mismatch fails one line later anyway. Here any mismatch
    raises immediately (only affects invalid input).
"""
import numpy as np


def ixneighbors(*args):
    if len(args) < 1 or len(args) > 3:
        raise TypeError('wrong number of input arguments')

    X = np.asarray(args[0])
    siz = X.shape
    nrc = siz[0] * siz[1]
    In = np.isnan(X.astype(float))           # all False for logical input

    if len(args) == 1:
        method = 'getall'
        nhood = 8
    else:
        ix = args[1]
        if ix is None or np.size(ix) == 0:
            method = 'getall'
        else:
            method = 'getsome'
            ix = np.asarray(ix)
            if ix.dtype == bool:
                if ix.shape != siz:
                    raise ValueError('if I is logical I and X must have same size')
            else:
                ixvec = ix.ravel(order='F').astype(np.int64)   # 0-based linear indices
                tmp = np.zeros(nrc, dtype=bool)                 # ix = false(siz); ix(ixvec) = true
                tmp[ixvec] = True
                ix = tmp.reshape(siz, order='F')
            ix = ~In & ix

        if len(args) == 3:
            nhood = np.atleast_1d(args[2])[0]
            if nhood not in (4, 8):
                raise ValueError('conn must be either 4 or 8')
        else:
            nhood = 8

    # replace values in X by index vector (0-based here; MATLAB 1-based)
    Xi = np.arange(nrc, dtype=float).reshape(siz, order='F')
    Xi[In] = np.nan

    # Pad array
    ic = np.full((siz[0] + 2, siz[1] + 2), np.nan)
    ic[1:-1, 1:-1] = Xi

    if method == 'getall':
        I = ~np.isnan(ic)
    else:
        I = np.zeros((siz[0] + 2, siz[1] + 2), dtype=bool)
        I[1:-1, 1:-1] = ix

    icF = ic.ravel(order='F')

    def pick(mask):
        # MATLAB ic(mask): values at true positions of mask, column-major order
        return icF[mask.ravel(order='F')]

    cols = []
    # Shift logical matrix I across the neighbors
    cols.append(pick(np.roll(I, 1, axis=1)))                       # I(:,[end 1:end-1])  shift to the right
    cols.append(pick(np.roll(I, 1, axis=0)))                       # I([end 1:end-1],:)  shift down
    cols.append(pick(np.roll(I, -1, axis=1)))                      # I(:,[2:end 1])      shift left
    cols.append(pick(np.roll(I, -1, axis=0)))                      # I([2:end 1],:)      shift up
    if nhood == 8:
        cols.append(pick(np.roll(np.roll(I, -1, axis=0), 1, axis=1)))   # shift up and right
        cols.append(pick(np.roll(np.roll(I, -1, axis=0), -1, axis=1)))  # shift up and left
        cols.append(pick(np.roll(np.roll(I, 1, axis=0), 1, axis=1)))    # shift down and right
        cols.append(pick(np.roll(np.roll(I, 1, axis=0), -1, axis=1)))   # shift down and left

    # Create output:  ic = repmat(ic(I(:)),nhood,1);  icd = icd(:)
    ic_out = np.tile(pick(I), nhood)
    icd_out = np.concatenate(cols)

    # Remove NaNs in neighbors
    keep = ~np.isnan(icd_out)
    return ic_out[keep].astype(np.int64), icd_out[keep].astype(np.int64)
