"""Level 1/2A: reading the ESRI ASCII rasters (rasterread)."""
import numpy as np
import pytest

from rasterread import rasterread
from conftest import INPUT_DIR, assert_same


def _write_asc(path, data, nodata=-9999, xll=100.0, yll=200.0, cs=10.0):
    nr, nc = data.shape
    with open(path, 'w') as f:
        f.write('ncols %d\nnrows %d\nxllcorner %g\nyllcorner %g\ncellsize %g\nNODATA_value %g\n'
                % (nc, nr, xll, yll, cs, nodata))
        for row in data:
            f.write(' '.join('%g' % v for v in row) + '\n')


def test_rasterread_orientation_and_coordinates(tmp_path):
    data = np.array([[1, 2, 3], [4, -9999, 6]], float)    # first row = NORTH
    p = tmp_path / 'g.asc'
    _write_asc(p, data)
    dem, X, Y, hdr = rasterread(str(p))
    assert dem.shape == (2, 3)
    assert dem[0, 0] == 1 and dem[1, 2] == 6               # file order kept
    assert np.isnan(dem[1, 1])                             # NODATA -> NaN
    # cell centres; Y decreases down the rows (north up)
    np.testing.assert_array_equal(X[0], [105.0, 115.0, 125.0])
    np.testing.assert_array_equal(Y[:, 0], [215.0, 205.0])


@pytest.mark.parametrize('fname', ['SP_DEM_LEV.asc', 'ice_thickness_LEV.asc',
                                   'lake_volume_LEV.asc', 'tensile_stresses_LEV.asc'])
def test_input_rasters_share_one_grid(fname):
    _, X0, Y0, _ = rasterread(str(INPUT_DIR / 'SP_DEM_LEV.asc'))
    z, X, Y, _ = rasterread(str(INPUT_DIR / fname))
    assert z.shape == X0.shape
    np.testing.assert_array_equal(X, X0)
    np.testing.assert_array_equal(Y, Y0)


def test_dem_and_coordinates_match_matlab(matlab_ws):
    dem, X, Y, _ = rasterread(str(INPUT_DIR / 'SP_DEM_LEV.asc'))
    assert_same(X, matlab_ws['X'], name='X')
    assert_same(Y, matlab_ws['Y'], name='Y')
    assert_same(dem.ravel(order='F'), matlab_ws['DEM'], name='DEM (raw, column-major)')


def test_other_rasters_match_matlab(matlab_ws):
    tensile, _, _, _ = rasterread(str(INPUT_DIR / 'tensile_stresses_LEV.asc'))
    h, _, _, _ = rasterread(str(INPUT_DIR / 'ice_thickness_LEV.asc'))
    lakes, _, _, _ = rasterread(str(INPUT_DIR / 'lake_volume_LEV.asc'))
    assert_same(tensile, matlab_ws['tensile'], name='tensile')
    assert_same(h.ravel(order='F'), matlab_ws['H'], name='H')
    assert_same(lakes, matlab_ws['lakes'], name='lakes')


def test_column_major_flatten_is_required(matlab_ws):
    """Guard against the most likely translation bug: C-order flattening."""
    dem, _, _, _ = rasterread(str(INPUT_DIR / 'SP_DEM_LEV.asc'))
    ref = np.asarray(matlab_ws['DEM']).ravel()
    wrong = dem.ravel(order='C')
    assert not np.array_equal(np.nan_to_num(wrong), np.nan_to_num(ref))
