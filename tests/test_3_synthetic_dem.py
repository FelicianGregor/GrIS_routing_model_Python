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
The met data is the real Lev_met_input09.xlsx. On these surfaces crevasses,
moulins and (for 'flat_and_lake') lake drainage all occur, so every part of
the model is used.

The grids are small, so these runs are much faster than a full model run.
Run only this test:   pytest tests/test_3_synthetic_dem.py
"""
import os
import shutil
import tempfile

import numpy as np
import pytest

from helpers import INPUT_DATA, run_model

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


@pytest.mark.parametrize('kind', ['ramp', 'flat_and_lake', 'rough'])
def test_model_runs_on_artificial_glacier(kind):
    folder = os.path.join(tempfile.mkdtemp(prefix='synthetic_'), 'input_data')
    make_inputs(folder, kind)
    try:
        results = run_model(input_folder=folder)
    finally:
        shutil.rmtree(os.path.dirname(folder), ignore_errors=True)

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
