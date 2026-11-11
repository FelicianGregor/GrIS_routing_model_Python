"""
flowdir.py
----------
Python translation of TopoToolbox (v1) flowdir.m (W. Schwanghart,
15 March 2009) for the options used by the model:

    Ms = flowdir(X, Y, dem, 'type', 'single')

which implies the defaults  mode='default', routeflats='route',
edges='closed', fillsinks=false, exponent=1.
Other option values are not used by the model and raise NotImplementedError.

Returned M is an (nrc x nrc) scipy.sparse CSR matrix in MATLAB LINEAR
(column-major, 0-based) cell numbering: M[k, l] = 1 if cell k drains to l.

Algorithm (as in flowdir.m):
  1. [ic1,icd1] = ixneighbors(dem): all 8-neighbour pairs, NaN cells excluded.
     e = (z_k - z_l) / hypot(dx, dy) with dx, dy from the X, Y grids.
  2. e = max(e, 0).  MATLAB max() ignores NaN; np.fmax is used (no NaN can
     occur anyway because ixneighbors drops NaN cells).
  3. M = sparse(ic1,icd1,e) + sparse(icf,icn,1) with [icf,icn] =
     routeflats(dem,'single')  (links across flats get weight 1).
  4. 'single': [m,IX2] = max(M,[],2); rows with m == 0 get no receiver.
     Tie-break: MATLAB max returns the FIRST maximum = smallest column
     (linear) index. Reproduced here.
     UNCERTAINTY (flagged): first-index tie-breaking for max() on a SPARSE
     row is the documented behaviour for full arrays and assumed for sparse.
     Ties only arise between neighbours with exactly equal slope.

routeflats contains a random element in MATLAB (randperm in neighs); see
routeflats.py. Consequently flowdir is not deterministic in MATLAB either.
"""
import numpy as np
import scipy.sparse as sp

from ixneighbors import ixneighbors
from routeflats import routeflats


def flowdir(X, Y, dem, type='single', mode='default', routeflats_method='route',
            edges='closed', fillsinks=False, exponent=1):
    if type != 'single' or mode != 'default' or routeflats_method != 'route' \
            or edges != 'closed' or fillsinks or exponent != 1:
        raise NotImplementedError('Only flowdir(X,Y,dem,"type","single") with '
                                  'default options is translated')
    dem = np.asarray(dem, dtype=float)
    siz = dem.shape
    if X.shape != Y.shape or X.shape != siz:
        raise ValueError('X, Y and dem must have same size')

    # general values
    nrc = dem.size

    Zf = dem.ravel(order='F')
    Xf = np.asarray(X, dtype=float).ravel(order='F')
    Yf = np.asarray(Y, dtype=float).ravel(order='F')

    # *********************************************************************
    # normal multiple FLOW DIRECTION calculation
    # calculate maximum slope and slope direction
    # find neighbors of cells
    ic1, icd1 = ixneighbors(dem)
    e = (Zf[ic1] - Zf[icd1]) / np.hypot(Xf[ic1] - Xf[icd1], Yf[ic1] - Yf[icd1])

    e = np.fmax(e, 0)                                   # MATLAB max ignores NaN

    # *********************************************************************
    # flow direction matrix
    rows, cols, vals = [ic1], [icd1], [e]

    # *********************************************************************
    # routing through flats ('route')
    icf, icn = routeflats(dem, type)
    rows.append(icf)
    cols.append(icn)
    vals.append(np.ones(icf.size))

    M = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nrc, nrc)).tocsr()
    M.sum_duplicates()                                  # sparse() sums duplicates

    # ******************************************************************
    # single flow direction, flow concentration
    # [m,IX2] = max(M,[],2); i = m==0; remove those rows
    coo = M.tocoo()
    r, c, v = coo.row, coo.col, coo.data
    order = np.lexsort((c, -v, r))      # by row, then value desc, then column asc
    r, c, v = r[order], c[order], v[order]
    first = np.ones(r.size, dtype=bool)
    first[1:] = r[1:] != r[:-1]
    r, c, v = r[first], c[first], v[first]
    keep = v > 0                        # m == 0 -> no outflow
    IX1, IX2 = r[keep], c[keep]
    M = sp.csr_matrix((np.ones(IX1.size), (IX1, IX2)), shape=(nrc, nrc))
    return M
