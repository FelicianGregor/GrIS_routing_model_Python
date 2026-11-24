"""
Shared pytest fixtures for validating the Python translation against MATLAB.

Fixtures
  matlab_ws        : MATLAB reference workspace (standard run, random flat routing)
  matlab_ws_fixed  : MATLAB reference with fixed neighbour order in routeflats
                     (produced by tests/matlab/make_matlab_references.m); tests
                     that need it are skipped until the file exists
  python_ws        : Python model run, default settings (seed 0), run ONCE
                     per pytest session in a temporary folder
  python_ws_fixed  : Python model run with ROUTEFLATS_FIXED_ORDER = True

The model is run in a temporary copy of input_data/ so that the real
results/ folder is never overwritten by the tests.
"""
import os
import runpy
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          # no windows during tests

import numpy as np
import pytest
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_SCRIPT = ROOT / 'Devel_SNOW_DELAY_FULL_MOULINS_AND_LAKES_MODEL_updated.py'
INPUT_DIR = ROOT / 'input_data'
REF_DIR = ROOT / 'MATLAB_reference_output'
MATLAB_REF = REF_DIR / 'workspace_routing_model_matlab.mat'
MATLAB_REF_FIXED = REF_DIR / 'workspace_routing_model_matlab_fixedorder.mat'
MATLAB_HELPER_FIXTURES = ROOT / 'tests' / 'fixtures' / 'helpers_matlab.mat'


def load_mat(path):
    d = sio.loadmat(str(path))
    return {k: v for k, v in d.items() if not k.startswith('__')}


def run_model(workdir, **init_globals):
    """Run the model script in `workdir` (with a copy of input_data/) and
    return its global namespace (= the MATLAB 'workspace')."""
    workdir = Path(workdir)
    shutil.copytree(INPUT_DIR, workdir / 'input_data', dirs_exist_ok=True)
    (workdir / 'results').mkdir(exist_ok=True)
    old = os.getcwd()
    os.chdir(workdir)
    try:
        return runpy.run_path(str(MODEL_SCRIPT), init_globals=dict(init_globals))
    finally:
        os.chdir(old)


@pytest.fixture(scope='session')
def matlab_ws():
    if not MATLAB_REF.exists():
        pytest.skip('MATLAB reference workspace not found: %s' % MATLAB_REF)
    return load_mat(MATLAB_REF)


@pytest.fixture(scope='session')
def matlab_ws_fixed():
    if not MATLAB_REF_FIXED.exists():
        pytest.skip('Fixed-order MATLAB reference not found (run '
                    'tests/matlab/make_matlab_references.m in MATLAB): %s' % MATLAB_REF_FIXED)
    return load_mat(MATLAB_REF_FIXED)


@pytest.fixture(scope='session')
def matlab_helper_fixtures():
    if not MATLAB_HELPER_FIXTURES.exists():
        pytest.skip('MATLAB helper fixtures not found (run '
                    'tests/matlab/make_matlab_references.m in MATLAB)')
    return load_mat(MATLAB_HELPER_FIXTURES)


@pytest.fixture(scope='session')
def python_ws(tmp_path_factory):
    return run_model(tmp_path_factory.mktemp('run_default'))


@pytest.fixture(scope='session')
def python_ws_fixed(tmp_path_factory):
    return run_model(tmp_path_factory.mktemp('run_fixed'), ROUTEFLATS_FIXED_ORDER=True)


@pytest.fixture(autouse=True)
def _reset_routeflats_mode():
    """Make sure no test leaves routeflats in fixed-order mode."""
    yield
    import routeflats
    routeflats.set_fixed_neighbour_order(False)


# --------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------
def as_matlab_shape(py, ref):
    """Bring a Python array to the shape of the MATLAB array (MATLAB stores
    vectors as n x 1 or 1 x n; Python as 1-D)."""
    py = np.asarray(py)
    ref = np.asarray(ref)
    if py.shape != ref.shape and py.size == ref.size:
        py = py.reshape(ref.shape)
    return py


def assert_same(py, ref, rtol=0.0, atol=0.0, name=''):
    """NaN-aware comparison. rtol=atol=0 means bit-identical."""
    ref = np.asarray(ref)
    if ref.dtype == bool:
        ref = ref.astype(float)
    py = as_matlab_shape(np.asarray(py, dtype=float), ref).astype(float)
    ref = ref.astype(float)
    assert py.shape == ref.shape, '%s: shape %s vs MATLAB %s' % (name, py.shape, ref.shape)
    np.testing.assert_array_equal(np.isnan(py), np.isnan(ref),
                                  err_msg='%s: NaN pattern differs' % name)
    if rtol == 0 and atol == 0:
        np.testing.assert_array_equal(py, ref, err_msg='%s differs' % name)
    else:
        np.testing.assert_allclose(py, ref, rtol=rtol, atol=atol, equal_nan=True,
                                   err_msg='%s differs' % name)
