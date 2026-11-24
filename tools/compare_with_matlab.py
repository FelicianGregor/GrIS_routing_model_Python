"""
compare_with_matlab.py
----------------------
Variable-by-variable comparison of a MATLAB workspace (.mat) with the Python
workspace written by the model when SAVE_FULL_WORKSPACE = True.

Usage
    python tools/compare_with_matlab.py MATLAB.mat PYTHON.mat [--csv out.csv]

For every variable present in both files it reports
    shape check, NaN-pattern check, max |abs diff|, max |rel diff|,
    number of differing elements (beyond rtol/atol), and - for daily
    matrices (cells x days) - the FIRST DAY on which a difference appears.
The first divergent day is the most useful clue for locating a bug: every
later difference may just be a consequence of the first one.

The sparse flow matrix Ms is compared by receiver per cell.

Notes on what is skipped / treated specially (see SKIP below):
  * loop counters (i, j, I, a, loop, d, Ki ...) hold "last value" state that is
    1-based in MATLAB and partly 0-based in Python -> not meaningful;
  * `ans` is MATLAB-only.
"""
import argparse
import csv
import sys

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

# loop counters / scratch values whose last value is not comparable
SKIP = {'i', 'j', 'I', 'a', 'loop', 'ans', 'd', 'Ki', 'r',
        'moulin_index', 'lakes_index',
        'SAVE_FULL_WORKSPACE', 'ROUTEFLATS_FIXED_ORDER', 'ROUTEFLATS_SEED'}

RTOL = 1e-12
ATOL = 1e-9


def load(path):
    d = sio.loadmat(path, squeeze_me=False)
    return {k: v for k, v in d.items() if not k.startswith('__')}


def _as_array(v):
    if sp.issparse(v):
        return v
    a = np.asarray(v)
    if a.dtype == bool:
        a = a.astype(float)
    return a


def receivers(M):
    """Receiver (0-based linear index) per cell of a single-flow-direction
    matrix; -1 where a cell has no outflow."""
    M = sp.csr_matrix(M)
    rec = np.full(M.shape[0], -1, dtype=np.int64)
    rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    rec[rows] = M.indices
    return rec


def compare_arrays(name, a, b, rtol=RTOL, atol=ATOL):
    res = {'variable': name, 'status': '', 'shape_matlab': str(a.shape),
           'shape_python': str(b.shape), 'nan_pattern_equal': '',
           'max_abs_diff': '', 'max_rel_diff': '', 'n_diff': '',
           'n_elements': '', 'first_divergent_day': ''}
    # MATLAB saves vectors as (n,1); Python 1-D arrays are saved as columns too
    if a.shape != b.shape:
        if a.size == b.size and np.squeeze(a).shape == np.squeeze(b).shape:
            a, b = np.squeeze(a), np.squeeze(b)
        else:
            res['status'] = 'SHAPE MISMATCH'
            return res
    if not (np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)):
        res['status'] = 'NON-NUMERIC (skipped)'
        return res
    a = a.astype(float)
    b = b.astype(float)
    na, nb = np.isnan(a), np.isnan(b)
    res['nan_pattern_equal'] = bool(np.array_equal(na, nb))
    both = ~na & ~nb
    with np.errstate(divide='ignore', invalid='ignore'):
        absd = np.where(both, np.abs(a - b), 0.0)
        # treat equal infinities as equal
        absd = np.where(both & np.isinf(a) & (a == b), 0.0, absd)
        rel = np.where(both & (np.abs(a) > 0), absd / np.abs(a), 0.0)
    differs = (absd > atol + rtol * np.where(both, np.abs(a), 0.0)) | (na != nb)
    res['n_elements'] = a.size
    res['n_diff'] = int(differs.sum())
    res['max_abs_diff'] = float(np.nanmax(absd)) if absd.size else 0.0
    res['max_rel_diff'] = float(np.nanmax(rel)) if rel.size else 0.0
    if a.ndim == 2 and a.shape[1] == 99 and a.shape[0] > 99 and differs.any():
        res['first_divergent_day'] = int(np.argmax(differs.any(axis=0))) + 1   # 1-based day
    elif a.ndim == 2 and 1 in a.shape and max(a.shape) == 99 and differs.any():
        res['first_divergent_day'] = int(np.argmax(differs.ravel())) + 1
    res['status'] = 'OK' if res['n_diff'] == 0 else 'DIFFERS'
    return res


def compare(mat_path, py_path, rtol=RTOL, atol=ATOL):
    m = load(mat_path)
    p = load(py_path)
    rows = []
    for name in sorted(set(m) & set(p)):
        if name in SKIP:
            continue
        a, b = _as_array(m[name]), _as_array(p[name])
        if sp.issparse(a) or sp.issparse(b):
            ra, rb = receivers(a), receivers(b)
            nd = int(np.sum(ra != rb))
            rows.append({'variable': name, 'status': 'OK' if nd == 0 else 'DIFFERS',
                         'shape_matlab': str(a.shape), 'shape_python': str(b.shape),
                         'nan_pattern_equal': '', 'max_abs_diff': '',
                         'max_rel_diff': '', 'n_diff': nd,
                         'n_elements': ra.size, 'first_divergent_day': '',
                         'note': 'cells with a different receiver'})
            continue
        rows.append(compare_arrays(name, a, b, rtol, atol))
    only_m = sorted(set(m) - set(p) - SKIP)
    only_p = sorted(set(p) - set(m) - SKIP)
    return rows, only_m, only_p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('matlab_mat')
    ap.add_argument('python_mat')
    ap.add_argument('--csv')
    ap.add_argument('--rtol', type=float, default=RTOL)
    ap.add_argument('--atol', type=float, default=ATOL)
    args = ap.parse_args(argv)
    rows, only_m, only_p = compare(args.matlab_mat, args.python_mat, args.rtol, args.atol)

    fmt = '{:<28} {:<14} {:>10} {:>12} {:>12} {:>9} {:>5} {}'
    print(fmt.format('variable', 'status', 'NaN equal', 'max abs', 'max rel',
                     'n diff', 'day1', 'shapes'))
    for r in sorted(rows, key=lambda r: (r['status'] == 'OK', r['variable'])):
        def f(x):
            return '{:.3g}'.format(x) if isinstance(x, float) else str(x)
        print(fmt.format(r['variable'], r['status'], str(r['nan_pattern_equal']),
                         f(r['max_abs_diff']), f(r['max_rel_diff']), str(r['n_diff']),
                         str(r['first_divergent_day']),
                         r['shape_matlab'] + (' / ' + r['shape_python']
                                              if r['shape_python'] != r['shape_matlab'] else '')))
    print('\nOnly in MATLAB:', ', '.join(only_m) or '-')
    print('Only in Python:', ', '.join(only_p) or '-')
    n_bad = sum(r['status'] != 'OK' for r in rows)
    print('\n{} of {} compared variables differ (rtol={}, atol={})'.format(
        n_bad, len(rows), args.rtol, args.atol))
    if args.csv:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return 0 if n_bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
