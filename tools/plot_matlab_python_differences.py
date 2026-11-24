"""
plot_matlab_python_differences.py
---------------------------------
Map WHERE the MATLAB and Python model results differ, together with the
ice margin.

Panels
  1-4  As, q, Q (final day) and melt_to_bed (net over 99 days): every
       differing cell is a marker filled with the signed difference
       MATLAB - Python (red = MATLAB larger, blue = Python larger);
       circle = off-ice cell, square = on-ice cell. Off-ice cells are shaded grey, the ice margin is a black line.
  5    Flow directions (Ms, final day): every cell routed differently (black
       circle) with its MATLAB receiver (orange) and Python receiver (green).
  6    Time series: number of cells with a different melt_to_bed per day,
       split into on-ice and off-ice cells.

Usage (from the repository folder)
  1) produce a Python workspace with the SAME routing mode as the MATLAB file:
       python -c "import runpy; runpy.run_path('Devel_SNOW_DELAY_FULL_MOULINS_AND_LAKES_MODEL_updated.py', init_globals={'SAVE_FULL_WORKSPACE': True, 'ROUTEFLATS_FIXED_ORDER': True})"
     (use ROUTEFLATS_FIXED_ORDER False to compare with the standard MATLAB run)
  2) python tools/plot_matlab_python_differences.py
     or with explicit files:
       python tools/plot_matlab_python_differences.py MATLAB.mat PYTHON.mat --out figure.png

Default files:
  MATLAB : MATLAB_reference_output/workspace_routing_model_matlab_fixedorder.mat
  Python : results/workspace_routing_model_python.mat
  Figure : results/differences_matlab_python.png

Only reads the two files; nothing in the model is changed.
"""
import argparse
import os

import matplotlib
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

ATOL = 1e-9        # absolute tolerance: smaller differences count as equal
RTOL = 1e-12       # relative tolerance (round-off of sums)


def load(path):
    d = sio.loadmat(path)
    return {k: v for k, v in d.items() if not k.startswith('__')}


def to_grid(v, shape):
    """MATLAB column vector (cells x 1) -> grid, column-major as in MATLAB."""
    v = np.asarray(v, dtype=float)
    if v.shape == shape:
        return v
    return v.ravel(order='F').reshape(shape, order='F')


def receivers(M):
    M = sp.csr_matrix(M)
    rec = np.full(M.shape[0], -1)
    rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    rec[rows] = M.indices
    return rec


def differs(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    nan_diff = np.isnan(a) != np.isnan(b)
    with np.errstate(invalid='ignore'):
        d = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
        return nan_diff | (d > ATOL + RTOL * np.abs(np.nan_to_num(a)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('matlab_mat', nargs='?',
                    default=os.path.join('MATLAB_reference_output',
                                         'workspace_routing_model_matlab_fixedorder.mat'))
    ap.add_argument('python_mat', nargs='?',
                    default=os.path.join('results', 'workspace_routing_model_python.mat'))
    ap.add_argument('--out', default=os.path.join('results', 'differences_matlab_python.png'))
    ap.add_argument('--show', action='store_true', help='also open the figure window')
    args = ap.parse_args(argv)

    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D

    m = load(args.matlab_mat)
    p = load(args.python_mat)

    X = np.asarray(m['X'], float)
    Y = np.asarray(m['Y'], float)
    shape = X.shape
    nr = shape[0]
    cs = abs(X[0, 1] - X[0, 0])
    extent = [X.min() - cs / 2, X.max() + cs / 2, Y.min() - cs / 2, Y.max() + cs / 2]
    off_ice = to_grid(m['ice_limits'], shape).astype(bool)   # True = off the ice
    on_ice = ~off_ice
    xkm = lambda v: v / 1000.0   # noqa: E731  (axes in km)

    # ---------------- per-cell signed differences (MATLAB - Python)
    panels = []
    for name in ['As', 'q', 'Q']:
        a = to_grid(m[name], shape)
        b = to_grid(p[name], shape)
        panels.append((name + ' (final day)', a - b, differs(a, b)))
    a = np.asarray(m['melt_to_bed'], float)
    b = np.asarray(p['melt_to_bed'], float).reshape(a.shape)
    dmask_daily = differs(a, b)                                    # cells x days
    dsum = (a - b).sum(axis=1)                                     # net over 99 days
    panels.append(('melt_to_bed (sum over 99 days)', to_grid(dsum, shape),
                   to_grid(dmask_daily.any(axis=1), shape).astype(bool)))

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    title = 'MATLAB - Python differences\nMATLAB: %s\nPython: %s' % (
        os.path.basename(args.matlab_mat), os.path.basename(args.python_mat))
    fig.suptitle(title, fontsize=11)

    def background(ax):
        ax.imshow(np.where(off_ice, 1.0, np.nan), extent=[xkm(e) for e in extent],
                  cmap='Greys', vmin=0, vmax=3, interpolation='nearest')
        ax.contour(xkm(X), xkm(Y), on_ice.astype(float), levels=[0.5],
                   colors='k', linewidths=0.8)
        ax.set_xlabel('Easting (km)')
        ax.set_ylabel('Northing (km)')
        ax.set_aspect('equal')

    for ax, (label, diff, mask) in zip(axes.ravel()[:4], panels):
        background(ax)
        n_on = int((mask & on_ice).sum())
        n_off = int((mask & off_ice).sum())
        if mask.any():
            lim = np.nanmax(np.abs(diff[mask]))
            lim = lim if lim > 0 else 1.0
            norm = TwoSlopeNorm(0, -lim, lim)
            # single 500 m cells are invisible at full-map scale, so every
            # differing cell is drawn as a marker filled with its difference
            # (circle = off-ice, square = on-ice)
            for sel, marker, edge in ((mask & off_ice, 'o', 'magenta'),
                                      (mask & on_ice, 's', 'red')):
                if sel.any():
                    sc = ax.scatter(xkm(X[sel]), xkm(Y[sel]), c=diff[sel], cmap='RdBu_r',
                                    norm=norm, s=70, marker=marker, edgecolors=edge,
                                    linewidths=1.2, zorder=3)
            fig.colorbar(sc, ax=ax, shrink=0.75, label='MATLAB - Python')
        ax.set_title('%s\ndiffering cells: %d on-ice, %d off-ice' % (label, n_on, n_off),
                     fontsize=10)

    # ---------------- flow directions
    ax = axes[1, 1]
    background(ax)
    rm, rp = receivers(m['Ms']), receivers(p['Ms'])
    dcells = np.flatnonzero(rm != rp)
    Xf, Yf = X.ravel(order='F'), Y.ravel(order='F')

    # receivers are only 500 m away (invisible at map scale): mark the giver
    # cell and the two receivers with symbols instead of arrows
    g = dcells
    ax.scatter(xkm(Xf[g]), xkm(Yf[g]), s=70, marker='o', facecolors='none',
               edgecolors='k', linewidths=1.2, zorder=3)
    okm, okp = rm[g] >= 0, rp[g] >= 0
    ax.scatter(xkm(Xf[rm[g][okm]]), xkm(Yf[rm[g][okm]]), s=40, marker='^',
               color='darkorange', zorder=4)
    ax.scatter(xkm(Xf[rp[g][okp]]), xkm(Yf[rp[g][okp]]), s=40, marker='v',
               color='green', zorder=4)
    off_flat = off_ice.ravel(order='F')
    ax.set_title('Flow direction Ms (final day): %d cells routed differently\n'
                 '(%d of them drain off-ice in both versions)'
                 % (dcells.size, int(np.sum(off_flat[rm[dcells]] & off_flat[rp[dcells]]))),
                 fontsize=10)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc='none', mec='k', label='cell routed differently'),
                       Line2D([], [], marker='^', ls='', color='darkorange', label='MATLAB receiver'),
                       Line2D([], [], marker='v', ls='', color='green', label='Python receiver')],
              loc='upper right', fontsize=8)

    # ---------------- time series
    ax = axes[1, 2]
    days = np.arange(1, a.shape[1] + 1)
    on_flat = ~off_flat
    ax.plot(days, dmask_daily[on_flat].sum(axis=0), color='red', label='on-ice cells')
    ax.plot(days, dmask_daily[off_flat].sum(axis=0), color='magenta', label='off-ice cells')
    daily_sum_diff = np.abs(a[off_flat].sum(0) - b[off_flat].sum(0)) + \
        np.abs(a[on_flat].sum(0) - b[on_flat].sum(0))
    ax.set_xlabel('Day of model run')
    ax.set_ylabel('cells with different melt_to_bed')
    ax.set_title('melt_to_bed: differing cells per day\n'
                 'max |difference of daily totals| = %.3g' % daily_sum_diff.max(), fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # common legend for map symbols
    fig.legend(handles=[
        Line2D([], [], color='k', lw=1, label='ice margin'),
        Line2D([], [], marker='s', ls='', mfc='lightgrey', mec='lightgrey', label='off-ice'),
        Line2D([], [], marker='s', ls='', mfc='white', mec='red', mew=1.2, label='differs, on-ice (fill = difference)'),
        Line2D([], [], marker='o', ls='', mfc='white', mec='magenta', mew=1.2, label='differs, off-ice (fill = difference)')],
        loc='lower left', ncol=4, fontsize=9)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print('saved', args.out)
    print('differing cells (on-ice / off-ice):')
    for label, _, mask in panels:
        print('  %-32s %4d / %4d' % (label, (mask & on_ice).sum(), (mask & off_ice).sum()))
    print('  %-32s %4d' % ('Ms receivers', dcells.size))
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
