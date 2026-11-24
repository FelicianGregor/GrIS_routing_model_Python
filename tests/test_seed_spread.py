"""
Level 4: does the random flat routing matter? Runs the model with other
seeds (each run ~1-2 min) and checks that the results agree with MATLAB in
the same way as the seed-0 run (Level 2): identical on the ice, totals equal
to round-off. Marked slow; run with  pytest -m slow
"""
import numpy as np
import pytest

from conftest import assert_same, run_model
from test_model_vs_matlab import SUM_RTOL, ice_mask

pytestmark = [pytest.mark.model, pytest.mark.slow]


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_other_seeds_agree_with_matlab(seed, matlab_ws, tmp_path):
    ws = run_model(tmp_path, ROUTEFLATS_SEED=seed)
    on = ice_mask(matlab_ws)
    for name in ['b', 'moulin', 'full_crevasses', 'drained_lakes', 'LVactual']:
        assert_same(ws[name], matlab_ws[name], name=name)
    assert_same(ws['melt_to_bed'][on], matlab_ws['melt_to_bed'][on], name='melt_to_bed on-ice')
    assert_same(ws['melt_delivery_discharge'], matlab_ws['melt_delivery_discharge'],
                rtol=SUM_RTOL, name='melt_delivery_discharge')
