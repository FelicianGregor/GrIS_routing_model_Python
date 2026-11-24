"""Level 1: MATLAB built-in behaviour reproduced in matlab_compat.py."""
import math

import numpy as np
import pytest

from matlab_compat import matlab_round, lin, sortrows, xlsread
from conftest import INPUT_DIR, assert_same


# ---------------------------------------------------------------- round()
@pytest.mark.parametrize('x, expected', [
    (0.5, 1.0), (1.5, 2.0), (2.5, 3.0), (3.5, 4.0),
    (-0.5, -1.0), (-1.5, -2.0), (-2.5, -3.0),
    (0.49999999999999994, 0.0),            # largest double below 0.5
    (2.4999999999999996, 2.0),
    (1e15 + 0.5, 1e15 + 1.0),
    (0.0, 0.0), (7.0, 7.0), (-7.2, -7.0),
])
def test_matlab_round_scalar(x, expected):
    assert matlab_round(x) == expected


def test_matlab_round_differs_from_numpy_on_halves():
    # documents WHY the helper exists: NumPy rounds halves to even
    assert np.round(2.5) == 2.0 and matlab_round(2.5) == 3.0


@pytest.mark.parametrize('x', [math.nan, math.inf, -math.inf])
def test_matlab_round_nonfinite(x):
    r = matlab_round(x)
    assert (math.isnan(r) and math.isnan(x)) or r == x


@pytest.mark.filterwarnings('ignore:invalid value encountered in subtract:RuntimeWarning')
def test_matlab_round_array_matches_scalar():
    # (the inf - inf inside matlab_round warns harmlessly; result is correct)
    x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 0.49999999999999994, np.nan, np.inf])
    r = matlab_round(x)
    for xi, ri in zip(x, r):
        s = matlab_round(float(xi))
        assert (np.isnan(ri) and np.isnan(s)) or ri == s


def test_matlab_round_keeps_negative_zero_sign():
    # MATLAB round(-0.4) is -0; equality with 0 is what matters downstream
    assert matlab_round(-0.4) == 0.0


# ---------------------------------------------------------- linear index
def test_lin_is_column_major():
    A = np.arange(12.0).reshape(3, 4)
    flat_f = A.ravel(order='F')
    for k in range(A.size):
        assert lin(A, k) == flat_f[k]


def test_lin_on_wide_matrix_reads_first_column_for_small_k():
    # model quirk: snow_total(j) with j <= SIZ always reads column 1
    A = np.arange(20.0).reshape(5, 4)
    for k in range(5):
        assert lin(A, k) == A[k, 0]


# ---------------------------------------------------------------- sortrows
def test_sortrows_easting_asc_northing_desc():
    A = np.array([[2, 10, 0], [1, 5, 1], [1, 9, 2], [2, 3, 3]], float)
    out = sortrows(A, [1, -2])
    np.testing.assert_array_equal(out[:, 2], [2, 1, 0, 3])


def test_sortrows_is_stable():
    A = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 2], [1, 1, 3]], float)
    out = sortrows(A, [1, -2])
    np.testing.assert_array_equal(out[:, 2], [2, 0, 1, 3])


def test_sortrows_rejects_nonfinite_keys():
    with pytest.raises(ValueError):
        sortrows(np.array([[np.nan, 1.0], [1.0, 2.0]]), [1, -2])


def test_sortrows_gridding_equals_fortran_reshape():
    """The xyz -> sortrows -> reshape(order='F') gridding used in the model
    must give back the original north-up grid (basis for proposed change 10)."""
    from rasterread import rasterread
    _, X, Y, _ = rasterread(str(INPUT_DIR / 'SP_DEM_LEV.asc'))
    Z = np.arange(X.size, dtype=float).reshape(X.shape)
    xyz = np.column_stack([X.ravel('F'), Y.ravel('F'), Z.ravel('F')])
    xyz = sortrows(xyz, [1, -2])
    h = int(np.sum(xyz[:, 0] == xyz[0, 0]))
    w = int(np.sum(xyz[:, 1] == xyz[0, 1]))
    np.testing.assert_array_equal(xyz[:, 2].reshape((h, w), order='F'), Z)


# ---------------------------------------------------------------- xlsread
def test_xlsread_trims_nan_rows_and_columns(tmp_path):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws['A1'] = 'header'
    ws['B3'] = 1.0
    ws['C3'] = 2.0
    ws['B4'] = 'text'     # text -> NaN, but row kept (not all NaN)
    ws['C4'] = 4
    f = tmp_path / 't.xlsx'
    wb.save(f)
    out = xlsread(str(f), 'A1:D6')
    expected = np.array([[1.0, 2.0], [np.nan, 4.0]])
    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    np.testing.assert_array_equal(np.nan_to_num(out), np.nan_to_num(expected))


def test_xlsread_met_data_matches_matlab(matlab_ws):
    metdata = xlsread(str(INPUT_DIR / 'Lev_met_input09.xlsx'), 'A2:C100')
    assert_same(metdata, matlab_ws['metdata'], name='metdata')
