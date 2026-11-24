"""
Smoke tests for the plotting scripts: they must run without error on the
output of a Python model run (non-interactive backend, no windows).
Visual comparison with the MATLAB figures is still done by eye.
"""
import os
import pickle
import runpy

import matplotlib.pyplot as plt
import pytest

from conftest import ROOT

pytestmark = pytest.mark.model

PLOT_SCRIPTS = ['spatial_lakes_plot.py', 'spatial_moulins_plot.py',
                'tstress_lakes_spatial_temporal.py']
PKL_KEYS = ['drainage_xyz', 'moulin_xyz', 'dem', 'X', 'Y', 'JD', 'DEM',
            'drained_lakes', 'Rxx', 'LV_vector']


@pytest.mark.parametrize('script', PLOT_SCRIPTS)
def test_plot_script_runs(script, python_ws, tmp_path, monkeypatch):
    (tmp_path / 'results').mkdir()
    with open(tmp_path / 'results' / 'results_routing_model.pkl', 'wb') as f:
        pickle.dump({k: python_ws[k] for k in PKL_KEYS}, f)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)
    runpy.run_path(str(ROOT / script))
    assert plt.get_fignums(), '%s produced no figure' % script
    plt.close('all')
