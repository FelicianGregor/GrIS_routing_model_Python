"""
Level 2: the full Python model against the standard MATLAB run
(MATLAB_reference_output/workspace_routing_model_matlab.mat).

The MATLAB run used RANDOM flat routing (routeflats/randperm). The
randomness only decides which of several equally low neighbours a flat
perimeter cell drains to. On the Leverett DEM this only happens at the ice
margin, where on-ice cells drain into off-ice (NaN) cells, all lowered to
the same fill value. Therefore:

  * all on-ice variables must be identical,
  * off-ice cells may receive the margin runoff in a different order, so
    off-ice values are compared through their daily sums.

If any of these tests fails, the translation (or an input) differs from
MATLAB at a place NOT explained by the random routing.

Groups (as in the testing plan):
  A  inputs and set-up                   -> bit-identical
  B  snow / melt component (no routing)  -> bit-identical
  C  flow directions                     -> identical except margin cells
  D  routing-dependent results           -> on-ice identical, totals to round-off
"""
import numpy as np
import pytest
import scipy.sparse as sp

from conftest import assert_same

pytestmark = pytest.mark.model

# summation order (np.sum vs MATLAB sum) may differ at round-off level
SUM_RTOL = 1e-12


def receivers(M):
    M = sp.csr_matrix(M)
    rec = np.full(M.shape[0], -1)
    rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    rec[rows] = M.indices
    return rec


def ice_mask(matlab_ws):
    """True for ON-ice cells (MATLAB ice_limits is True OFF the ice)."""
    return ~np.asarray(matlab_ws['ice_limits']).ravel().astype(bool)


# ------------------------------------------------------------------ Group A
GROUP_A = ['DEM', 'surface_nans', 'SIZ', 'size_days', 'surf_elev', 'metdata', 'JD',
           'temp', 'precip', 'X', 'Y', 'eastings', 'northings', 'tensile', 'Rxx_pa',
           'Rxx', 'h', 'H', 'lakes', 'LV', 'LV_vector', 'ice_limits',
           'snow_depth', 'pmax_snow_depth',
           # parameters
           'met_elev', 'temp_lapse', 'DDF_i', 'DDF_s', 'Pi', 'Pw', 'P_s', 'g', 't',
           'Kic', 'TS', 'crev_width', 'cell_width', 'crev_length',
           'percolation_time', 'perc_grad', 'pmax', 's_depth', 'snow_elev',
           'accum_grad', 'melt_start']


@pytest.mark.parametrize('name', GROUP_A)
def test_group_A_inputs_identical(name, python_ws, matlab_ws):
    assert_same(python_ws[name], matlab_ws[name], name=name)


def test_group_A_filled_dem_identical(python_ws, matlab_ws):
    # dem is overwritten by fillsinks(dem) every day
    assert_same(python_ws['dem'], matlab_ws['dem'], name='dem (filled)')


# ------------------------------------------------------------------ Group B
GROUP_B_EXACT = ['snow_total', 'cumu_melt', 'pmax_filling', 'delay_timer',
                 'daily_melt_inc_days', 'perc_lapse', 'melt_delay', 'lapse_elev',
                 'temp_elev', 'pos_temp', 'snow', 'precip_snow',
                 'daily_melt', 'daily_melt_inc', 'melt_grid', 'W0', 'melt_xyz']
GROUP_B_SUMS = ['tot_melt', 'tot_tot_melt', 'total_generated_melt',
                'total_generated_melt_inc', 'percent_melt_stored_supra']


@pytest.mark.parametrize('name', GROUP_B_EXACT)
def test_group_B_melt_identical(name, python_ws, matlab_ws):
    assert_same(python_ws[name], matlab_ws[name], name=name)


@pytest.mark.parametrize('name', GROUP_B_SUMS)
def test_group_B_melt_totals(name, python_ws, matlab_ws):
    assert_same(python_ws[name], matlab_ws[name], rtol=SUM_RTOL, name=name)


# ------------------------------------------------------------------ Group C
def test_group_C_flow_directions(python_ws, matlab_ws):
    """Receivers identical everywhere except where an on-ice margin cell
    drains into an off-ice cell (random choice among equal NaN cells)."""
    rp = receivers(python_ws['Ms'])
    rm = receivers(matlab_ws['Ms'])
    off = ~ice_mask(matlab_ws)
    diff = np.flatnonzero(rp != rm)
    assert np.all(rp[diff] >= 0) and np.all(rm[diff] >= 0), \
        'a cell has an outflow in one version only'
    bad = diff[~(off[rp[diff]] & off[rm[diff]])]
    assert bad.size == 0, ('%d cells route differently NOT explained by the random '
                           'margin routing, e.g. linear indices %s' % (bad.size, bad[:10]))


def test_group_C_single_flow_direction(python_ws):
    M = sp.csr_matrix(python_ws['Ms'])
    assert np.all(np.diff(M.indptr) <= 1)


# ------------------------------------------------------------------ Group D
GROUP_D_DAILY = ['b', 'moulin', 'full_crevasses', 'drained_lakes',
                 'lake_drainage_events']
GROUP_D_FINAL = ['LVactual', 'LV_percent_fill', 'q_lake', 'Q_lake', 'crevasse_depth',
                 'Ki_lake', 'b_lake', 'B_lake', 'B', 'dmax', 'RR',
                 'moulin_grid', 'drainage_grid', 'moulin_xyz', 'drainage_xyz']


@pytest.mark.parametrize('name', GROUP_D_DAILY + GROUP_D_FINAL)
def test_group_D_routing_dependent_identical(name, python_ws, matlab_ws):
    """These must agree EVERYWHERE: on-ice they do not depend on the random
    choice, and off-ice they are not affected by where margin water goes."""
    assert_same(python_ws[name], matlab_ws[name], name=name)


@pytest.mark.parametrize('name', ['As', 'q', 'Q'])
def test_group_D_accumulation_on_ice_identical(name, python_ws, matlab_ws):
    on = ice_mask(matlab_ws)
    py = np.asarray(python_ws[name]).ravel(order='F')
    ml = np.asarray(matlab_ws[name]).ravel(order='F')
    assert_same(py[on], ml[on], name=name + ' (on-ice)')


def test_group_D_melt_to_bed_on_ice_identical(python_ws, matlab_ws):
    on = ice_mask(matlab_ws)
    assert_same(python_ws['melt_to_bed'][on], matlab_ws['melt_to_bed'][on],
                name='melt_to_bed (on-ice)')


def test_group_D_melt_to_bed_off_ice_daily_sum(python_ws, matlab_ws):
    off = ~ice_mask(matlab_ws)
    assert_same(python_ws['melt_to_bed'][off].sum(axis=0),
                matlab_ws['melt_to_bed'][off].sum(axis=0),
                rtol=SUM_RTOL, atol=1e-9, name='daily off-ice melt_to_bed')


@pytest.mark.parametrize('name', ['total_daily_melt_delivery', 'melt_delivery_volume',
                                  'melt_delivery_discharge', 'total_melt_to_bed',
                                  'percent_melt_to_bed'])
def test_group_D_delivery_totals(name, python_ws, matlab_ws):
    assert_same(python_ws[name], matlab_ws[name], rtol=SUM_RTOL, name=name)
