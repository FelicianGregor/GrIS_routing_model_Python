"""
TEST 3: run the model on small artificial ice surfaces
======================================================
Instead of the Leverett data, the model gets a small made-up glacier
(20 x 30 cells of 500 m). Three different surfaces are tried:

  'ramp'           a simple slope from the margin up to the interior
  'flat_and_lake'  slope with a completely flat area and a lake basin
  'rough'          slope with random bumps (many pits and flat spots)

Each run must finish and give sensible results (no NaN, no negative water,
moulins only on the ice, not more water at the bed than was produced).

For every surface a figure is saved to results/synthetic_dem_tests/<name>.png
showing the DEM, the lakes, the crevasses that drained to the bed (last day)
and the drainage network (flow directions of the last day).
The met data is the real Lev_met_input09.xlsx. On these surfaces crevasses,
moulins and (for 'flat_and_lake') lake drainage all occur, so every part of
the model is used.

The grids are small, so these runs are much faster than a full model run.
Run only this test:   pytest tests/test_3_synthetic_dem.py
"""
import os
import shutil
import tempfile

import matplotlib
matplotlib.use('Agg')                 # save figures to files, do not open windows
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.sparse as sp

from helpers import INPUT_DATA, REPO, run_model

FIGURE_FOLDER = os.path.join(REPO, 'results', 'synthetic_dem_tests')

N_ROWS, N_COLS = 20, 30          # grid size
CELL = 500                       # cell size (m); the model assumes 500 m cells
NODATA = -9999


def write_asc(path, grid):
    """Write a grid as ESRI ASCII raster (first row = north), NaN -> NODATA."""
    with open(path, 'w') as f:
        f.write(f'ncols {N_COLS}\nnrows {N_ROWS}\n')
        f.write('xllcorner 500000\nyllcorner 7400000\n')
        f.write(f'cellsize {CELL}\nNODATA_value {NODATA}\n')
        for row in np.where(np.isnan(grid), NODATA, grid):
            f.write(' '.join(f'{value:.4f}' for value in row) + '\n')


def make_surface(kind):
    """Elevation (m) of the artificial glacier. The western 3 columns are
    off the ice (NaN); the ice rises towards the east."""
    rng = np.random.default_rng(1)
    column = np.arange(N_COLS)
    row = np.arange(N_ROWS)[:, None]
    dem = 300 + 30.0 * column + 2.0 * row     # 300 m to about 1200 m

    if kind == 'flat_and_lake':
        dem[5:12, 10:18] = dem[8, 14]                   # completely flat area
        dem[14:17, 20:23] -= 25                         # basin for a lake
    elif kind == 'rough':
        dem = dem + rng.uniform(-40, 40, dem.shape).round()

    dem[:, :3] = np.nan                                 # off-ice margin
    return dem


def make_inputs(folder, kind):
    """Write all input files the model needs into folder/."""
    dem = make_surface(kind)
    ice = ~np.isnan(dem)
    rng = np.random.default_rng(2)

    thickness = np.where(ice, 50 + 0.5 * (dem - 300), np.nan)          # m
    tensile_kpa = np.where(ice, rng.uniform(0, 200, dem.shape), np.nan)  # kPa
    lakes = np.where(ice, 0.0, np.nan)                                   # 10^6 m^3
    if kind == 'flat_and_lake':
        lakes[14:17, 20:23] = 0.2

    os.makedirs(folder)
    write_asc(os.path.join(folder, 'SP_DEM_LEV.asc'), dem)
    write_asc(os.path.join(folder, 'ice_thickness_LEV.asc'), thickness)
    write_asc(os.path.join(folder, 'tensile_stresses_LEV.asc'), tensile_kpa)
    write_asc(os.path.join(folder, 'lake_volume_LEV.asc'), lakes)
    shutil.copy(os.path.join(INPUT_DATA, 'Lev_met_input09.xlsx'), folder)


def plot_run(results, kind):
    """Save a map of one synthetic run: DEM, lakes, crevasses that drained
    to the bed, and the drainage network."""
    X = results['X'] / 1000.0                       # cell centres in km
    Y = results['Y'] / 1000.0
    shape = X.shape
    dem = results['DEM'].reshape(shape, order='F')  # elevation as read (NaN = off-ice)

    # last day of the run (column -1), converted from vectors to grids
    lakes = results['LV'] > 0                                          # lake cells
    drained_lake = results['drained_lakes'][:, -1].reshape(shape, order='F') == 1
    crevasse_to_bed = results['full_crevasses'][:, -1].reshape(shape, order='F') == 1
    crevasse_to_bed = crevasse_to_bed & ~np.isnan(dem)                 # on the ice only

    fig, ax = plt.subplots(figsize=(10, 7))

    # 1) DEM
    image = ax.pcolormesh(X, Y, dem, cmap='terrain', shading='nearest')
    fig.colorbar(image, ax=ax, label='surface elevation (m)')

    # 2) drainage network: a line from each cell to the cell it drains into
    #    (Ms = flow directions of the last day); thicker = more water (As)
    Ms = sp.csr_matrix(results['Ms'])
    giver, receiver = Ms.nonzero()
    x = X.ravel(order='F')
    y = Y.ravel(order='F')
    water = results['As'].ravel(order='F')[giver]
    width = 0.3 + 2.5 * np.log10(1 + water) / np.log10(1 + water.max())
    for g, r, w in zip(giver, receiver, width):
        ax.plot([x[g], x[r]], [y[g], y[r]], color='navy', linewidth=w)

    # 3) crevasses that drained to the bed, and lakes
    ax.scatter(X[crevasse_to_bed], Y[crevasse_to_bed], marker='x', color='red', s=25,
               label='crevasse drained to the bed')
    ax.scatter(X[lakes & ~drained_lake], Y[lakes & ~drained_lake], marker='s', s=70,
               facecolors='none', edgecolors='blue', linewidths=1.5, label='lake (not drained)')
    ax.scatter(X[drained_lake], Y[drained_lake], marker='s', s=70, color='cyan',
               edgecolors='blue', label='lake (drained)')
    ax.plot([], [], color='navy', label='drainage network (thicker = more water)')

    ax.set_title(f"Synthetic DEM '{kind}' - last day of the run\n"
                 f"{int(crevasse_to_bed.sum())} crevasses drained to the bed, "
                 f"{int(drained_lake.sum())} of {int(lakes.sum())} lake cells drained")
    ax.set_xlabel('Easting (km)')
    ax.set_ylabel('Northing (km)')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', bbox_to_anchor=(1.25, 1), fontsize=9)

    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    fig.savefig(os.path.join(FIGURE_FOLDER, kind + '.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)


@pytest.mark.parametrize('kind', ['ramp', 'flat_and_lake', 'rough'])
def test_model_runs_on_artificial_glacier(kind):
    folder = os.path.join(tempfile.mkdtemp(prefix='synthetic_'), 'input_data')
    make_inputs(folder, kind)
    try:
        results = run_model(input_folder=folder)
    finally:
        shutil.rmtree(os.path.dirname(folder), ignore_errors=True)

    plot_run(results, kind)                       # figure in results/synthetic_dem_tests/

    melt_to_bed = results['melt_to_bed']          # cells x days (mm w.e.)
    off_ice = results['ice_limits']               # True = cell is off the ice

    # 1) no NaN in the main outputs
    assert np.all(np.isfinite(melt_to_bed))
    assert np.all(np.isfinite(results['melt_delivery_discharge']))

    # 2) no negative water
    assert np.all(melt_to_bed >= 0)
    assert np.all(results['b'] >= 0)

    # 3) moulins only on the ice (from day 2 on; on day 1 the model does
    #    not yet apply the ice mask)
    assert np.all(results['moulin'][off_ice, 1:] == 0)

    # 4) melt was produced, and not more water reached the bed than was produced
    assert results['total_generated_melt_inc'] > 0
    assert 0 <= results['percent_melt_to_bed'] <= 100
