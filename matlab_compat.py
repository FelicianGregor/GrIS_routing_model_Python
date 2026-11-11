"""
matlab_compat.py
----------------
Small helpers that reproduce MATLAB built-in behaviour where Python/NumPy
behave differently. Each function documents the MATLAB behaviour it mirrors.

Used by the translation of Devel_SNOW_DELAY_FULL_MOULINS_AND_LAKES_MODEL_updated.m
"""
import math
import datetime as _dt

import numpy as np


# ---------------------------------------------------------------------------
# round()
# ---------------------------------------------------------------------------
def matlab_round(x):
    """MATLAB round(): rounds halves AWAY from zero (2.5 -> 3, -2.5 -> -3).

    MATLAB-SPECIFIC: Python's round() and np.round() round halves to EVEN
    (2.5 -> 2), so they must NOT be used where MATLAB used round().

    Implementation note: x - floor(x) is exact in IEEE double for |x| < 2**52,
    so the >= 0.5 test is exact (floor(|x| + 0.5) is NOT exact for
    0.49999999999999994).
    """
    if np.ndim(x) == 0:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return x
        ax = abs(x)
        r = math.floor(ax)
        if ax - r >= 0.5:
            r += 1.0
        return math.copysign(float(r), x)
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    r = np.floor(ax)
    r = np.where(ax - r >= 0.5, r + 1.0, r)
    return np.where(np.isfinite(x), np.copysign(r, x), x)


# ---------------------------------------------------------------------------
# linear indexing
# ---------------------------------------------------------------------------
def lin(A, k):
    """MATLAB linear indexing A(k+1) on a 2-D array (0-based k here).

    MATLAB-SPECIFIC: MATLAB arrays are column-major, so A(k) walks DOWN the
    first column first. For an (nrows x ncols) matrix, A(k) == A(row, col)
    with row = mod(k-1, nrows)+1 and col = floor((k-1)/nrows)+1.
    """
    nrows = A.shape[0]
    return A[k % nrows, k // nrows]


# ---------------------------------------------------------------------------
# sortrows(A, [1 -2])
# ---------------------------------------------------------------------------
def sortrows(A, columns):
    """MATLAB sortrows(A, columns) with 1-based signed column spec.

    Positive column -> ascending, negative -> descending. MATLAB's sortrows
    is STABLE (rows with equal keys keep their original order); np.lexsort
    is also stable, so the result is identical for finite keys.
    Only finite keys are supported (coordinates here are never NaN).
    """
    A = np.asarray(A)
    keys = []
    # np.lexsort: LAST key is primary -> pass keys in reverse priority
    for c in reversed(columns):
        col = A[:, abs(c) - 1]
        if not np.all(np.isfinite(col)):
            raise ValueError('sortrows helper: non-finite sort keys not supported')
        keys.append(col if c > 0 else -col)
    idx = np.lexsort(tuple(keys))
    return A[idx]


# ---------------------------------------------------------------------------
# xlsread(file, range)
# ---------------------------------------------------------------------------
def _excel_serial(d):
    """Excel serial date number (1900 system) for a datetime/date."""
    base = _dt.datetime(1899, 12, 30)
    if isinstance(d, _dt.date) and not isinstance(d, _dt.datetime):
        d = _dt.datetime(d.year, d.month, d.day)
    delta = d - base
    return delta.days + delta.seconds / 86400.0 + delta.microseconds / 86400e6


def xlsread(filename, cell_range, sheet=None):
    """Numeric output of MATLAB xlsread(filename, cell_range).

    Reproduced MATLAB behaviour:
      * reads the first worksheet unless `sheet` is given;
      * formula cells return their cached (last calculated) values;
      * non-numeric cells (text, empty) become NaN;
      * leading and trailing rows and columns that are entirely NaN are
        removed (xlsread trims them from the numeric output).

    UNCERTAINTY (flagged): exact xlsread treatment of Excel booleans and of
    date-formatted cells depends on MATLAB version/platform. Here booleans
    become 1.0/0.0 and dates become Excel serial numbers. Neither is expected
    in Lev_met_input09.xlsx (JD, temperature, precipitation are plain numbers).
    """
    from openpyxl import load_workbook
    from openpyxl.utils import range_boundaries

    min_col, min_row, max_col, max_row = range_boundaries(cell_range)
    wb = load_workbook(filename, read_only=True, data_only=True)
    try:
        ws = wb[sheet] if sheet is not None else wb.worksheets[0]
        nr = max_row - min_row + 1
        nc = max_col - min_col + 1
        data = np.full((nr, nc), np.nan)
        for r, row in enumerate(ws.iter_rows(min_row=min_row, max_row=max_row,
                                             min_col=min_col, max_col=max_col,
                                             values_only=True)):
            for c, v in enumerate(row):
                if isinstance(v, bool):
                    data[r, c] = float(v)
                elif isinstance(v, (int, float)):
                    data[r, c] = float(v)
                elif isinstance(v, (_dt.datetime, _dt.date)):
                    data[r, c] = _excel_serial(v)
                # str / None -> stays NaN
    finally:
        wb.close()

    # trim leading/trailing all-NaN rows and columns (xlsread behaviour)
    good_rows = np.where(~np.all(np.isnan(data), axis=1))[0]
    good_cols = np.where(~np.all(np.isnan(data), axis=0))[0]
    if good_rows.size == 0 or good_cols.size == 0:
        return np.zeros((0, 0))
    return data[good_rows[0]:good_rows[-1] + 1, good_cols[0]:good_cols[-1] + 1]
