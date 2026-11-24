"""
Python-only consistency checks on the model output (no MATLAB needed).

These are properties of the MODEL as written (checked to hold in the MATLAB
reference run too). They protect against regressions during further
development. They are NOT a complete mass balance (see proposed change 4).
"""
import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.model


@pytest.mark.parametrize('name', ['moulin', 'full_crevasses', 'drained_lakes'])
def test_connections_persist(name, python_ws):
    x = python_ws[name]
    assert set(np.unique(x)) <= {0.0, 1.0}
    assert np.all(np.diff(x, axis=1) >= 0), '%s switches off again' % name


def test_no_moulins_off_ice_after_day_1(python_ws):
    off = python_ws['ice_limits']
    assert np.all(python_ws['moulin'][off, 1:] == 0)


def test_melt_to_bed_non_negative_and_finite(python_ws):
    x = python_ws['melt_to_bed']
    assert np.all(np.isfinite(x)) and np.all(x >= 0)


def test_cumulative_quantities_monotone(python_ws):
    assert np.all(np.diff(python_ws['snow_total'], axis=1) >= 0)
    assert np.all(np.diff(python_ws['cumu_melt'], axis=1) >= 0)
    assert np.all(np.diff(python_ws['b'], axis=1) >= 0)


def test_lake_fill_never_exceeds_capacity(python_ws):
    LV, LVa = python_ws['LV'], python_ws['LVactual']
    fin = np.isfinite(LV)
    assert np.all(LVa[fin] <= LV[fin])
    assert np.all(LVa[fin] >= 0)


def test_totals_consistent(python_ws):
    ws = python_ws
    np.testing.assert_allclose(ws['total_daily_melt_delivery'],
                               ws['melt_to_bed'].sum(axis=0), rtol=1e-12)
    np.testing.assert_allclose(ws['melt_delivery_discharge'],
                               ws['total_daily_melt_delivery'] / 1000 * 500 * 500 / 86400,
                               rtol=1e-12)
    np.testing.assert_allclose(ws['total_melt_to_bed'],
                               ws['total_daily_melt_delivery'].sum(), rtol=1e-12)
    assert ws['total_generated_melt'] <= ws['total_generated_melt_inc']


def test_off_ice_share_of_bed_delivery_reported(python_ws, record_property):
    """Not a pass/fail check: records how much 'bed delivery' comes from
    off-ice cells (proposed change 1). About 2.1 % in the standard run."""
    off = python_ws['ice_limits']
    share = python_ws['melt_to_bed'][off].sum() / python_ws['total_melt_to_bed']
    record_property('off_ice_share_of_melt_to_bed', float(share))
    assert 0 <= share < 1


def test_flow_matrix_single_direction(python_ws):
    M = sp.csr_matrix(python_ws['Ms'])
    assert np.all(np.diff(M.indptr) <= 1)
