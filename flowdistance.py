"""
flowdistance.py
---------------
Python translation of TopoToolbox (v1) flowdistance.m
(W. Schwanghart, 5 February 2010): flow distance to the catchment outlet.

    D = flowdistance(M, X, Y)          2-D distance along the flow path
    D = flowdistance(M, X, Y, dem)     3-D distance (dem not empty)

Used by flowaccwithlakes as flowdistance(M, X, Y) with X, Y = unit-cell
meshgrid; only max(D) is used there, to set the number of routing iterations.

Algorithm (as in flowdistance.m): for every link k -> l in the single flow
direction matrix M, f_k = hypot(X_k - X_l, Y_k - Y_l); B(k) = f_k;
D = (speye - M) \\ B, i.e. D_k = f_k + D_receiver(k), D = 0 at outlets.

MATLAB-specific notes
  * The sparse solve `\\` is reproduced with scipy.sparse.linalg.spsolve.
    Path lengths are sums of 1 and sqrt(2); the order of additions may
    differ from MATLAB's solver at round-off level only.
  * If M contained a flow cycle, (I - M) would be singular: MATLAB warns
    and returns Inf/NaN. spsolve behaves similarly (warning, NaN/Inf).
  * The 5-argument form (distance to cells ix / logical I) requires
    dependencemap.m, which was not provided; it is not used by the model.
  * The second output L is not used by the model and is not translated.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def flowdistance(M, X, Y, dem=None, ix=None):
    if ix is not None:
        raise NotImplementedError('flowdistance(...,ix) needs dependencemap.m '
                                  '(not provided, not used by the model)')

    siz = X.shape
    nrc = X.size

    flag3d = dem is not None and np.size(dem) > 0

    # do the matrices have same size
    if not flag3d:
        if siz != Y.shape:
            raise ValueError('X and Y must have same size')
    else:
        if not (siz == Y.shape == np.shape(dem)):
            raise ValueError('X, Y and dem must have same size')

    # does the flow direction matrix correspond to the DEM
    if nrc != M.shape[0] or nrc != M.shape[1]:
        raise ValueError('M must be a square, sparse matrix with size [numel(X) numel(X)]')

    # check if single flow direction Matrix is used
    M = sp.csr_matrix(M, dtype=float, copy=True)
    M.eliminate_zeros()
    M.data[:] = 1.0                                     # spones(M)
    if np.any(np.asarray(M.sum(axis=1)).ravel() > 1):
        raise ValueError('single flow direction matrix must be used')

    # get linear indices of flow connectivity
    coo = M.tocoo()
    ic, icd = coo.row, coo.col
    Xf = np.asarray(X, dtype=float).ravel(order='F')
    Yf = np.asarray(Y, dtype=float).ravel(order='F')
    # calculate distance between nodes
    f = np.hypot(Xf[ic] - Xf[icd], Yf[ic] - Yf[icd])
    if flag3d:
        demF = np.asarray(dem, dtype=float).ravel(order='F')
        f = np.hypot(f, demF[ic] - demF[icd])

    B = np.zeros(nrc)
    B[ic] = f

    D = spsolve((sp.identity(nrc, format='csc') - M.tocsc()).tocsc(), B)
    D = np.asarray(D).reshape(siz, order='F')
    return D
