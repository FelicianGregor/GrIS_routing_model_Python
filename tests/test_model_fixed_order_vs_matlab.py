"""
Level 3: deterministic comparison. Both MATLAB and Python run with the
FIXED neighbour order in routeflats (randperm(8) -> 1:8), so every variable,
including the flow matrix and the off-ice cells, must agree.

Needs MATLAB_reference_output/workspace_routing_model_matlab_fixedorder.mat,
produced by tests/matlab/make_matlab_references.m. Skipped until it exists.

A failure here (with Level 2 passing) points to the flat routing
(routeflats tie-breaking, unique(), bwperim, imregionalmin).
"""
import numpy as np
import pytest
import scipy.sparse as sp

from conftest import assert_same
from test_model_vs_matlab import (GROUP_A, GROUP_B_EXACT, GROUP_B_SUMS,
                                  GROUP_D_DAILY, GROUP_D_FINAL, SUM_RTOL, receivers)

pytestmark = [pytest.mark.model, pytest.mark.matlab_fixed]

EXACT = GROUP_A + ['dem'] + GROUP_B_EXACT + GROUP_D_DAILY + GROUP_D_FINAL + \
    ['As', 'q', 'Q', 'melt_to_bed']
SUMS = GROUP_B_SUMS + ['total_daily_melt_delivery', 'melt_delivery_volume',
                       'melt_delivery_discharge', 'total_melt_to_bed',
                       'percent_melt_to_bed']


def test_reference_really_used_fixed_order(matlab_ws_fixed):
    assert 'ROUTEFLATS_FIXED_ORDER_MATLAB' in matlab_ws_fixed, \
        'reference file was not produced by make_matlab_references.m'


def test_flow_matrix_identical(matlab_ws_fixed, python_ws_fixed):
    rp = receivers(python_ws_fixed['Ms'])
    rm = receivers(matlab_ws_fixed['Ms'])
    diff = np.flatnonzero(rp != rm)
    assert diff.size == 0, '%d cells with a different receiver, e.g. %s' % (diff.size, diff[:10])


@pytest.mark.parametrize('name', EXACT)
def test_variable_identical(name, matlab_ws_fixed, python_ws_fixed):
    assert_same(python_ws_fixed[name], matlab_ws_fixed[name], name=name)


@pytest.mark.parametrize('name', SUMS)
def test_totals(name, matlab_ws_fixed, python_ws_fixed):
    assert_same(python_ws_fixed[name], matlab_ws_fixed[name], rtol=SUM_RTOL, name=name)
