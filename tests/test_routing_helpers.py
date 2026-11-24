"""
Level 1: TopoToolbox-derived routing helpers on small synthetic DEMs.

These tests check properties that any correct implementation must satisfy
(they do not need MATLAB). Exact comparison with MATLAB on the same small
DEMs is in test_helpers_vs_matlab_fixtures.py (needs MATLAB fixtures).
"""
import numpy as np
import pytest
import scipy.sparse as sp

import routeflats
from fillsinks import fillsinks
from flowaccwithlakes import flowaccwithlakes
from flowdir import flowdir
from flowdistance import flowdistance
from ixneighbors import ixneighbors


def grid(nr, nc, cs=1.0):
    X, Y = np.meshgrid(np.arange(nc) * cs, np.arange(nr)[::-1] * cs)
    return X, Y


def receivers(M):
    M = sp.csr_matrix(M)
    rec = np.full(M.shape[0], -1)
    rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    rec[rows] = M.indices
    return rec


def is_8_neighbour(k, l, nr):
    rk, ck = k % nr, k // nr
    rl, cl = l % nr, l // nr
    return max(abs(rk - rl), abs(ck - cl)) == 1


# ---------------------------------------------------------------- DEMs
def dem_pit():
    """Tilted plane (draining to the west/left edge) with a 2-cell-deep pit."""
    X, Y = grid(7, 7)
    dem = 10 + X * 1.0 + 0.1 * Y
    dem[3, 3] -= 5
    return dem


def dem_flat():
    """Plane draining west with a flat 3x3 plateau in the middle."""
    X, Y = grid(9, 9)
    dem = 10 + X * 1.0 + 0.01 * Y
    dem[3:6, 3:6] = dem[4, 4]
    return dem


def dem_with_nan_margin():
    """Plane draining west; western columns are off-ice (NaN)."""
    X, Y = grid(8, 8)
    dem = 10 + X * 1.0 + 0.05 * Y
    dem[:, :2] = np.nan
    return dem


ALL_DEMS = [dem_pit, dem_flat, dem_with_nan_margin]


# ---------------------------------------------------------------- ixneighbors
def test_ixneighbors_counts_and_adjacency():
    A = np.zeros((3, 3))
    ic, icd = ixneighbors(A)
    # 3x3 grid: 12 orthogonal + 8 diagonal undirected pairs -> 40 directed
    assert ic.size == icd.size == 40
    assert all(is_8_neighbour(k, l, 3) for k, l in zip(ic, icd))
    assert len(set(zip(ic.tolist(), icd.tolist()))) == 40      # no duplicates


def test_ixneighbors_first_direction_is_east():
    # documented order: direction 1 = right (east, +nrows in linear index)
    A = np.zeros((3, 3))
    ic, icd = ixneighbors(A)
    assert np.all(icd[:6] - ic[:6] == 3)


def test_ixneighbors_drops_nan_cells():
    A = np.zeros((3, 3))
    A[1, 1] = np.nan
    ic, icd = ixneighbors(A)
    centre = 1 + 1 * 3
    assert centre not in ic and centre not in icd


# ---------------------------------------------------------------- fillsinks
@pytest.mark.parametrize('make', ALL_DEMS)
def test_fillsinks_basic_properties(make):
    dem = make()
    f = fillsinks(dem)
    nan = np.isnan(dem)
    np.testing.assert_array_equal(np.isnan(f), nan)          # NaN restored
    assert np.all(f[~nan] >= dem[~nan])                       # only raises
    np.testing.assert_array_equal(fillsinks(f), f)            # idempotent


def test_fillsinks_fills_pit_to_spill_level():
    dem = dem_pit()
    f = fillsinks(dem)
    # pit at (3,3) is filled to the lowest of its 8 neighbours
    nb = dem[2:5, 2:5].copy()
    nb[1, 1] = np.inf
    assert f[3, 3] == nb.min()
    # all other cells untouched
    mask = np.ones(dem.shape, bool)
    mask[3, 3] = False
    np.testing.assert_array_equal(f[mask], dem[mask])


def test_fillsinks_border_connected_nan_is_outlet():
    dem = dem_with_nan_margin()
    np.testing.assert_array_equal(fillsinks(dem), dem)        # nothing to fill


def test_fillsinks_enclosed_nan_is_not_outlet():
    X, Y = grid(7, 7)
    dem = 10 + X * 1.0
    dem[3, 3] = np.nan                                        # hole inside ice
    f = fillsinks(dem)
    assert np.isnan(f[3, 3])
    np.testing.assert_array_equal(f[~np.isnan(dem)], dem[~np.isnan(dem)])


# ---------------------------------------------------------------- flowdir
def _check_flowdir_valid(dem_filled, M):
    nr = dem_filled.shape[0]
    rec = receivers(M)
    M = sp.csr_matrix(M)
    assert np.all(np.diff(M.indptr) <= 1)                     # single flow direction
    assert np.all(M.data == 1)
    z = dem_filled.ravel(order='F')
    zfill = np.where(np.isnan(z), np.nanmin(z) - 1, z)        # routeflats lowers NaN
    for k, l in enumerate(rec):
        if l < 0:
            continue
        assert is_8_neighbour(k, l, nr), 'cell %d drains to non-neighbour %d' % (k, l)
        assert zfill[l] <= zfill[k], 'cell %d drains uphill' % k
    # no cycles: every flow path ends -> flow distance finite
    X, Y = np.meshgrid(np.arange(1, dem_filled.shape[1] + 1), np.arange(1, nr + 1))
    D = flowdistance(M, X, Y)
    assert np.all(np.isfinite(D))
    return rec


@pytest.mark.parametrize('make', ALL_DEMS)
@pytest.mark.parametrize('fixed', [False, True])
def test_flowdir_produces_valid_single_flow_directions(make, fixed):
    routeflats.set_fixed_neighbour_order(fixed)
    dem = fillsinks(make())
    X, Y = grid(*dem.shape)
    _check_flowdir_valid(dem, flowdir(X, Y, dem, type='single'))


def test_flowdir_steepest_descent_on_plane():
    X, Y = grid(6, 6)
    dem = 10 + 2.0 * X + 0.5 * Y        # steepest descent = west (-x)
    M = flowdir(X, Y, dem, type='single')
    rec = receivers(M)
    nr = dem.shape[0]
    for k in range(dem.size):
        r, c = k % nr, k // nr
        if 1 <= r < nr - 1 and 1 <= c < dem.shape[1] - 1:   # interior cells
            # expected receiver = neighbour with the maximum slope
            best, bestl = -np.inf, -1
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    s = (dem[r, c] - dem[rr, cc]) / np.hypot(dr, dc)
                    if s > best:
                        best, bestl = s, rr + cc * nr
            assert rec[k] == bestl


def test_flowdir_routes_every_interior_flat_cell():
    dem = fillsinks(dem_flat())
    X, Y = grid(*dem.shape)
    for fixed in (True, False):
        routeflats.set_fixed_neighbour_order(fixed)
        rec = receivers(flowdir(X, Y, dem, type='single'))
        flat_cells = [r + c * dem.shape[0] for r in range(3, 6) for c in range(3, 6)]
        assert np.all(rec[flat_cells] >= 0)


def test_routeflats_fixed_order_is_deterministic():
    dem = fillsinks(dem_flat())
    X, Y = grid(*dem.shape)
    routeflats.set_fixed_neighbour_order(True)
    r1 = receivers(flowdir(X, Y, dem, type='single'))
    routeflats.seed_routeflats(123)                  # seed must not matter now
    r2 = receivers(flowdir(X, Y, dem, type='single'))
    np.testing.assert_array_equal(r1, r2)


def test_routeflats_seed_reproducible():
    dem = fillsinks(dem_flat())
    X, Y = grid(*dem.shape)
    routeflats.seed_routeflats(7)
    r1 = receivers(flowdir(X, Y, dem, type='single'))
    routeflats.seed_routeflats(7)
    r2 = receivers(flowdir(X, Y, dem, type='single'))
    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------- flowdistance
def test_flowdistance_chain():
    # 1 x 4 chain draining east: 0 -> 1 -> 2 -> 3 (linear indices, 1 row)
    M = sp.csr_matrix((np.ones(3), ([0, 1, 2], [1, 2, 3])), shape=(4, 4))
    X, Y = np.meshgrid(np.arange(1, 5), [1])
    np.testing.assert_allclose(flowdistance(M, X, Y), [[3, 2, 1, 0]])


def test_flowdistance_diagonal_step():
    # 2 x 2 grid, cell (0,0) [lin 0] drains diagonally to (1,1) [lin 3]
    M = sp.csr_matrix((np.ones(1), ([0], [3])), shape=(4, 4))
    X, Y = np.meshgrid([1, 2], [1, 2])
    D = flowdistance(M, X, Y)
    assert D.ravel(order='F')[0] == pytest.approx(np.sqrt(2), rel=1e-15)


# ---------------------------------------------------------------- flowaccwithlakes
def _setup_acc(make, fixed=True):
    routeflats.set_fixed_neighbour_order(fixed)
    dem = fillsinks(make())
    X, Y = grid(*dem.shape)
    M = flowdir(X, Y, dem, type='single')
    return dem, M


@pytest.mark.parametrize('make', ALL_DEMS)
def test_flowacc_mass_conservation_no_lakes(make):
    dem, M = _setup_acc(make)
    rng = np.random.default_rng(0)
    W0 = rng.uniform(0, 10, dem.shape)
    RR = np.ones(dem.shape, bool)
    LV = np.zeros(dem.shape)
    A, LVa = flowaccwithlakes(M, W0, RR, LV, np.zeros(dem.shape))
    outlets = receivers(M) < 0
    # everything generated ends up in outlet cells
    assert A.ravel(order='F')[outlets].sum() == pytest.approx(W0.sum(), rel=1e-12)
    assert np.all(A >= W0 - 1e-12)                       # accumulation never loses water
    np.testing.assert_array_equal(LVa, 0)


def test_flowacc_lake_stores_until_full():
    dem, M = _setup_acc(dem_pit)
    W0 = np.ones(dem.shape)
    RR = np.ones(dem.shape, bool)
    LV = np.zeros(dem.shape)
    LV[3, 3] = 5.0                                       # lake capacity at pit
    A, LVa = flowaccwithlakes(M, W0, RR, LV, np.zeros(dem.shape))
    assert LVa[3, 3] <= 5.0
    outlets = receivers(M) < 0
    stored = LVa.sum()
    out = A.ravel(order='F')[outlets].sum()
    assert out + stored == pytest.approx(W0.sum(), rel=1e-12)   # mass balance


def test_flowacc_runoff_ratio_zero_absorbs_water():
    dem, M = _setup_acc(dem_flat)
    W0 = np.ones(dem.shape)
    RR = np.ones(dem.shape, bool)
    RR[:, 5] = False                                    # a "crevasse field" column
    A, _ = flowaccwithlakes(M, W0, RR, np.zeros(dem.shape), np.zeros(dem.shape))
    outlets = receivers(M) < 0
    absorbed = A[:, 5].sum()
    # outlet water + water captured by RR=0 cells == generated (minus double
    # counting: RR=0 cells' own melt is in both A and W0 only once)
    out = A.ravel(order='F')[outlets & RR.ravel(order='F')].sum()
    assert out + absorbed == pytest.approx(W0.sum(), rel=1e-12)
