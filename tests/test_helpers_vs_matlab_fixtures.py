"""
Level 1 (exact): helper functions against MATLAB on small synthetic DEMs.

Inputs AND outputs come from tests/fixtures/helpers_matlab.mat, written by
tests/matlab/make_matlab_references.m (MATLAB, fixed neighbour order in
routeflats). Skipped until that file exists.

Case k (k = 1..N) provides  case<k>_dem, _X, _Y, _W0, _RR, _LV,
and MATLAB results  _filled, _Ms, _A, _LVactual, _D, _ic, _icd
(ic/icd are 1-based MATLAB linear indices).
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
from conftest import assert_same, MATLAB_HELPER_FIXTURES, load_mat


def _cases():
    if not MATLAB_HELPER_FIXTURES.exists():
        return [1]          # placeholder; fixture below skips
    d = load_mat(MATLAB_HELPER_FIXTURES)
    return sorted({int(k[4:k.index('_')]) for k in d if k.startswith('case')})


CASES = _cases()


def get(fx, k, name):
    return np.asarray(fx['case%d_%s' % (k, name)])


def receivers(M):
    M = sp.csr_matrix(M)
    rec = np.full(M.shape[0], -1)
    rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    rec[rows] = M.indices
    return rec


@pytest.mark.parametrize('k', CASES)
def test_ixneighbors(k, matlab_helper_fixtures):
    fx = matlab_helper_fixtures
    ic, icd = ixneighbors(get(fx, k, 'dem'))
    assert_same(ic + 1, get(fx, k, 'ic').ravel(), name='ic (order matters)')
    assert_same(icd + 1, get(fx, k, 'icd').ravel(), name='icd (order matters)')


@pytest.mark.parametrize('k', CASES)
def test_fillsinks(k, matlab_helper_fixtures):
    fx = matlab_helper_fixtures
    assert_same(fillsinks(get(fx, k, 'dem')), get(fx, k, 'filled'), name='filled dem')


@pytest.mark.parametrize('k', CASES)
def test_flowdir_fixed_order(k, matlab_helper_fixtures):
    fx = matlab_helper_fixtures
    routeflats.set_fixed_neighbour_order(True)
    M = flowdir(get(fx, k, 'X'), get(fx, k, 'Y'), get(fx, k, 'filled'), type='single')
    rp, rm = receivers(M), receivers(fx['case%d_Ms' % k])
    diff = np.flatnonzero(rp != rm)
    assert diff.size == 0, 'case %d: %d cells differ, e.g. %s' % (k, diff.size, diff[:10])


@pytest.mark.parametrize('k', CASES)
def test_flowdistance(k, matlab_helper_fixtures):
    fx = matlab_helper_fixtures
    M = fx['case%d_Ms' % k]
    dem = get(fx, k, 'dem')
    X, Y = np.meshgrid(np.arange(1, dem.shape[1] + 1), np.arange(1, dem.shape[0] + 1))
    assert_same(flowdistance(M, X, Y), get(fx, k, 'D'), rtol=1e-12, name='D')


@pytest.mark.parametrize('k', CASES)
def test_flowaccwithlakes(k, matlab_helper_fixtures):
    fx = matlab_helper_fixtures
    A, LVa = flowaccwithlakes(fx['case%d_Ms' % k], get(fx, k, 'W0'),
                              get(fx, k, 'RR').astype(bool), get(fx, k, 'LV'),
                              np.zeros(get(fx, k, 'dem').shape))
    assert_same(A, get(fx, k, 'A'), rtol=1e-12, name='A')
    assert_same(LVa, get(fx, k, 'LVactual'), rtol=1e-12, name='LVactual')
