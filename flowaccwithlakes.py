"""
flowaccwithlakes.py
-------------------
Python translation of flowaccwithlakes.m (weighted flow accumulation with
lake filling and overspill; the active "lakes can overspill" version. The
commented-out, non-working "channel downcutting" version is not translated).

    A, LVactual = flowaccwithlakes(M, W0, RR, LV, LVactual)

M        : single-flow-direction sparse matrix from flowdir (linear indexing)
W0       : today's melt per cell (grid, mm w.e.)
RR       : runoff ratio grid/vector (1 = pass water on, 0 = cell absorbs it)
LV       : lake capacity per cell (grid, mm)
LVactual : current lake fill (grid, mm), carried over between days

Algorithm per iteration r = 1 : nrruns+5
    LVactual  = LVactual + W0           water arriving at each cell enters its lake
    W0        = LVactual - min(LV, LVactual)   overspill above capacity
    LVactual  = min(LV, LVactual)
    W0        = M' * W0                 overspill moves one cell downstream,
                                        scaled by RR of the SENDING cell
    A         = A + W0                  A = local melt + everything that arrives

MATLAB-specific behaviour reproduced:
  * min(LV, LVactual) IGNORES NaN in MATLAB: where LV is NaN, LVactual is
    kept (unlimited storage, nothing spills). NumPy np.minimum would give
    NaN -> np.fmin is used.
  * `for r = 1:nrruns+5` with non-integer nrruns runs floor(nrruns+5) times.
  * M = spdiags(RR(:),0,nrc,nrc)*M zeroes the outflow rows of RR == 0 cells,
    so their water is removed from the routing (absorbed by the crevasse /
    drained lake). It still appears in A at that cell.

  * nrruns = max(D(:)) with D from flowdistance (translated in
    flowdistance.py). Extra iterations beyond the longest flow path change
    nothing, so round-off differences in D cannot change the result. A
    warning is raised if water is still moving after the last iteration.
    If D were all NaN (flow cycle), MATLAB's loop 1:NaN+5 is empty; this is
    reproduced (zero iterations).

UNCERTAINTY (flagged):
  * ismulti.m and multi2single.m were not provided. ismulti() here returns
    True if any cell has more than one receiver; the model only uses single
    flow direction, so multi2single is never needed.
  * M' * W0 sums inflows in a different order than MATLAB's sparse product;
    differences are at round-off level (~1e-16 relative).
"""
import math
import warnings

import numpy as np
import scipy.sparse as sp

from flowdistance import flowdistance


def ismulti(M):
    """True if any cell has more than one receiver (multiple flow direction)."""
    M = sp.csr_matrix(M)
    return bool(np.any(np.diff(M.indptr) > 1))


def flowaccwithlakes(M, W0, RR, LV, LVactual):
    W0 = np.asarray(W0, dtype=float)
    siz = W0.shape
    nrc = W0.size

    # evaluate the number of runs
    if ismulti(M):
        # multi2single.m was not provided; the model only uses 'single'
        raise NotImplementedError('multiple-flow-direction M is not supported')
    multi = False

    X, Y = np.meshgrid(np.arange(1, siz[1] + 1), np.arange(1, siz[0] + 1))
    D = flowdistance(M, X, Y)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        nrruns = np.nanmax(D) if not np.all(np.isnan(D)) else np.nan   # MATLAB max ignores NaN

    # adjust M to include RR
    RRv = np.asarray(RR).ravel(order='F').astype(float)
    if np.any(RRv < 0) or np.any(RRv > 1):
        raise ValueError('values in RR must range between 0 and 1')
    M = sp.diags(RRv, 0, shape=(nrc, nrc)) @ sp.csr_matrix(M)
    MT = M.T.tocsr()

    A = W0.copy()
    A = A.ravel(order='F')
    LVactual = np.asarray(LVactual, dtype=float).ravel(order='F').copy()
    LV = np.asarray(LV, dtype=float).ravel(order='F')
    W0 = W0.ravel(order='F').copy()

    # This version allows lakes to overspill and contribute to downstream
    # accumulation (without downcutting channels) when they reach their
    # maximum fill volume.
    n_iter = 0 if np.isnan(nrruns) else int(math.floor(nrruns + 5))   # MATLAB 1:nrruns+5
    for r in range(1, n_iter + 1):
        # the amount trapped in lakes
        LVactual = LVactual + W0
        LVactualnew = np.fmin(LV, LVactual)      # MATLAB min ignores NaN
        # amount that is further transported is
        W0 = LVactual - LVactualnew
        # the new lake volume is
        LVactual = LVactualnew
        # then water is routed further downstream
        W0 = MT @ W0
        # flow accumulation
        A = A + W0

    if np.any(W0 != 0):
        warnings.warn('flowaccwithlakes: water still in transit after the last '
                      'iteration (nrruns too small); routing was truncated')

    A = A.reshape(siz, order='F')
    LVactual = LVactual.reshape(siz, order='F')
    return A, LVactual
