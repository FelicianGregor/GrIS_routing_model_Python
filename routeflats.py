"""
routeflats.py
-------------
Python translation of TopoToolbox (v1) routeflats.m
(W. Schwanghart, 15 March 2009): route flow through flats of a DEM.
Called by flowdir(...,'type','single') with the default 'routeflats','route'.

    IXf, IXn = routeflats(dem, type)

IXf : 0-based linear (column-major) indices of cells in flats
IXn : their downstream neighbours (flowdir adds a link of weight 1 IXf->IXn)

Only type 'single' is used by the model and translated. The MATLAB file's
nested subfunctions routeflatssingle() and neighs() are translated as nested
Python functions (closures). Like MATLAB nested functions they read the
parent's `dem` and `siz` AS THEY ARE AT CALL TIME, i.e. after NaN cells and
the outer border ring have been set to `fillval`.

Algorithm (as in routeflats.m)
  1. flats = cells with no strictly lower 8-neighbour (imerode == dem),
     with NaN -> +Inf (so a missing neighbour never counts as lower).
  2. Flat cells in the second ring that touch a border cell of equal or
     lower elevation are linked to that border cell (one link per cell,
     first occurrence in ixneighbors order).
  3. NaN cells and the outer border ring are lowered to fillval = min(dem)-1;
     flats in the outer TWO rings and regional minima ("one-pixel pits",
     whole plateaus without an outlet) are removed.
  4. routeflatssingle: every flat cell on the flat's perimeter (bwperim,
     8-conn) that has a non-flat neighbour of equal or lower elevation gets
     ONE such neighbour; then, ring by ring inward, every remaining flat
     cell is linked to an already-routed neighbour.

!!! MATLAB-SPECIFIC BEHAVIOUR THAT AFFECTS RESULTS !!!
  a) RANDOMNESS: neighs() uses randperm(8) to shuffle the neighbour order.
     In step 4, when a perimeter flat cell has several valid downstream
     neighbours, the one chosen is RANDOM. MATLAB results therefore depend
     on the random-number state; because the main script recomputes
     flowdir every day, the routing across flats can change from day to day
     and from run to run. Python cannot reproduce MATLAB's random stream
     (MATLAB's randperm algorithm and session RNG state are not
     available), so here the shuffle uses a NumPy generator that is
     seeded for reproducibility (seed_routeflats(seed)). Statistically the
     behaviour is the same; individual realisations are not.
     The inward propagation (the while loop) is deterministic: a receiving
     cell is assigned to the giving cell with the smallest linear index.
  b) unique(): modern MATLAB (R2013a+) returns the FIRST occurrence, which
     np.unique(return_index=True) also does. MATLAB R2012b and earlier
     returned the LAST occurrence. If the original results were produced
     with an older MATLAB, ties are broken differently (flagged).
  c) NaN cells are lowered to fillval in step 3, so flat ice cells next to
     the NaN (off-ice) area are routed INTO the NaN cells. Water leaves the
     ice there (see the main-script notes for how that water is counted).
  d) Links are only returned if interior flats exist; if all flats were
     removed in step 3, the border links from step 2 are dropped as well
     (exactly as in MATLAB).
  e) If a flat cell could never be reached, MATLAB's while loop would run
     forever. This cannot happen for the flats left after step 3, but a
     guard raises an error instead of hanging.

UNCERTAINTY (flagged): bwperim(flats,8) is implemented as
flats & ~erode(flats, 3x3) with zero padding, and imregionalmin with
8-connectivity and the image outside treated as not lower. Both are the
documented MATLAB semantics.
"""
import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from scipy.sparse.csgraph import connected_components

from ixneighbors import ixneighbors


# Random generator replacing MATLAB's global stream for randperm(8)
_RNG = np.random.default_rng(0)


def seed_routeflats(seed):
    """Reset the generator used for neighs()/randperm(8)."""
    global _RNG
    _RNG = np.random.default_rng(seed)


def _randperm8():
    return _RNG.permutation(8)


def _imerode3(A):
    """imerode(A, ones(3)); MATLAB pads with +Inf for erosion."""
    return ndimage.grey_erosion(A, footprint=np.ones((3, 3), dtype=bool),
                                mode='constant', cval=np.inf)


def _bwperim8(BW):
    """bwperim(BW, 8): True pixels with at least one False 8-neighbour
    (outside the image counts as False)."""
    er = ndimage.binary_erosion(BW, structure=np.ones((3, 3), dtype=bool),
                                border_value=0)
    return BW & ~er


def _imregionalmin8(A):
    """imregionalmin(A) with 8-connectivity: connected plateaus of equal
    value none of whose cells has a strictly lower neighbour."""
    nr, nc = A.shape
    idx = np.arange(nr * nc).reshape(nr, nc)
    has_lower = np.zeros(A.shape, dtype=bool)
    rows, cols = [], []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r0, r1 = max(0, -dr), nr - max(0, dr)
            c0, c1 = max(0, -dc), nc - max(0, dc)
            a = A[r0:r1, c0:c1]
            b = A[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            has_lower[r0:r1, c0:c1] |= b < a
            eq = a == b
            rows.append(idx[r0:r1, c0:c1][eq])
            cols.append(idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc][eq])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    G = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(nr * nc, nr * nc))
    _, lab = connected_components(G, directed=False)
    comp_has_lower = np.zeros(lab.max() + 1, dtype=bool)
    np.logical_or.at(comp_has_lower, lab, has_lower.ravel())
    return (~comp_has_lower[lab]).reshape(nr, nc)


def routeflats(dem, type):
    # type: 'single' or 'multi'
    type = type.lower()
    dem = np.array(dem, dtype=float, copy=True)      # MATLAB passes by value
    with np.errstate(all='ignore'):
        mindem = np.nanmin(dem) if not np.all(np.isnan(dem)) else np.nan   # min ignores NaN
    fillval = mindem - 1
    siz = dem.shape

    # border cells
    # nhood = ones(3,3)   (used via _imerode3)

    # handle NaNs
    log_nans = np.isnan(dem)
    flag_nans = bool(np.any(log_nans))

    # if there are nans.
    if flag_nans:
        dem[log_nans] = np.inf

    # identify flats
    # flats: logical matrix with true where cells don't have lower neighbors
    if flag_nans:
        flats = (_imerode3(dem) == dem) & ~log_nans
    else:
        flats = _imerode3(dem) == dem

    # create connectivity between boundary flat pixels and proximate, interior
    # flat pixels
    boundaryf = np.ones(siz, dtype=bool)
    boundaryf[1:-1, 1:-1] = False

    icb, icdb = ixneighbors(flats, boundaryf)
    flatsF = flats.ravel(order='F')
    boundF = boundaryf.ravel(order='F')
    I = flatsF[icdb] & ~boundF[icdb]
    icdb = icdb[I]
    icb = icb[I]

    demF = dem.ravel(order='F')
    I = demF[icdb] >= demF[icb]
    icdb = icdb[I]
    icb = icb[I]

    if type == 'single':
        # [icdb,m] = unique(icdb); icb = icb(m)   (m = FIRST occurrence, MATLAB >= R2013a)
        icdb, m = np.unique(icdb, return_index=True)
        icb = icb[m]

    # if there are nans.
    if flag_nans:
        dem[log_nans] = fillval

    # remove flats at the border
    flats[:, [0, 1, -2, -1]] = False
    flats[[0, 1, -2, -1], :] = False

    # lower dem a bit along the edges
    dem[:, [0, -1]] = fillval
    dem[[0, -1], :] = fillval

    # remove one-pixel pits
    flats[_imregionalmin8(dem)] = False

    # ------------------------------------------------------------------
    # nested subfunctions (see module doc: they see dem/siz at call time)
    def neighs(IXf):
        # get neighbors of a cell in a matrix  (linear indices, no bounds check,
        # as in MATLAB; only called for cells >= 2 cells away from the border)
        ix = _randperm8()
        IXn = np.zeros((IXf.shape[0], 8), dtype=np.int64)
        IXn[:, ix[0]] = IXf + 1
        IXn[:, ix[1]] = IXf - 1
        IXn[:, ix[2]] = IXf + siz[0]
        IXn[:, ix[3]] = IXf - siz[0]

        IXn[:, ix[4]] = IXf + 1 + siz[0]
        IXn[:, ix[5]] = IXf - 1 + siz[0]
        IXn[:, ix[6]] = IXf - 1 - siz[0]
        IXn[:, ix[7]] = IXf + 1 - siz[0]
        return IXn

    def routeflatssingle(flats):
        # recursive routing through flats using single flow direction
        demF = dem.ravel(order='F')
        flatsF = flats.ravel(order='F').copy()

        # linear indices of flat cells
        ic = np.flatnonzero(flatsF)
        # number of flat cells
        nrcf = ic.size
        # create nan array that will be filled with downstream
        # neigbors of indices in ic in the subsequent lines of codes
        icd = np.full(nrcf, np.nan)

        # inner border in flats
        finline = _bwperim8(flats).ravel(order='F')

        # first create connectivity to downstream cells
        Igive = finline[ic].copy()
        IXgive = np.flatnonzero(Igive)

        # find neighbors of
        IXn = neighs(ic[Igive])
        I2 = (~np.isin(IXn, ic[Igive])) & (~flatsF[IXn]) & \
             (demF[IXn] <= demF[ic[Igive]][:, None])
        # cells to which an flow direction can be assigned
        # do have at least one downstream neighbor (I2)
        Igive[IXgive[~np.any(I2, axis=1)]] = False

        # I2 = I2 & cumsum(I2) == 1  -> first valid neighbour in the
        # (randomly permuted) column order, per cell
        I2 = I2 & (np.cumsum(I2, axis=1) == 1)

        # icd(Igive) = IXn'(I2'(:)): row-major boolean indexing gives the
        # same cell-by-cell order
        icd[Igive] = IXn[I2]

        # indicate the cells as nonflat
        flatsF[ic[Igive]] = False

        # another logical array that indicates if a flat cell receives from
        # upstream neighbor
        Ireceive = np.zeros(nrcf, dtype=bool)

        # iteratively remove non-giving cells
        while np.any(~Igive):
            n_before = int(np.sum(Igive))

            # find cells that give but not receive
            Itemp = Igive & ~Ireceive
            IXn = ic[Itemp]

            # find neighbors of these cells
            IXf = neighs(IXn)
            # which of these neighbors are flat cells
            I2 = flatsF[IXf]
            # which IXn do have flat neighbors at all
            II = np.any(I2, axis=1)

            IXn = IXn[II]
            IXf = IXf[II, :]
            I2 = I2[II, :]

            # IXn = repmat(IXn,1,8)'; IXf = IXf'; I2 = I2';
            # IXn = IXn(I2(:)); IXf = IXf(I2(:));
            # -> giver by giver, columns in (random) column order
            IXn = np.repeat(IXn[:, None], 8, axis=1)[I2]
            IXf = IXf[I2]

            # [IXf,m] = unique(IXf); IXn = IXn(m)   (first occurrence)
            IXf, m = np.unique(IXf, return_index=True)
            IXn = IXn[m]

            I = np.isin(ic, IXf)            # ismembc(ic,IXf)
            Igive[I] = True
            icd[I] = IXn
            Ireceive[Itemp] = True

            flatsF[IXf] = False

            if int(np.sum(Igive)) == n_before and not np.any(Itemp):
                # MATLAB would loop forever here
                raise RuntimeError('routeflats: unreachable flat cells; the MATLAB '
                                   'code would not terminate on this DEM')

        return ic, icd.astype(np.int64)
    # ------------------------------------------------------------------

    # any flats there? If yes, start recursive flat routing
    if np.any(flats):
        if type == 'single':
            IXf, IXn = routeflatssingle(flats)
            IXf = np.concatenate([IXf, icdb])
            IXn = np.concatenate([IXn, icb])
        elif type == 'multi':
            # routeflatsmulti is not used by the model (flowdir 'single')
            raise NotImplementedError("routeflats(...,'multi') is not translated")
        else:
            IXf = np.zeros(0, dtype=np.int64)
            IXn = np.zeros(0, dtype=np.int64)
    else:
        IXf = np.zeros(0, dtype=np.int64)
        IXn = np.zeros(0, dtype=np.int64)

    return IXf.astype(np.int64), IXn.astype(np.int64)
