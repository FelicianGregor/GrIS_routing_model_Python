"""
helpers.py
----------
Small helper functions used by the test scripts and by make_gold_standard.py.

run_model(...)        runs the model once in a separate (temporary) folder
                      and returns all its variables as a dictionary
load_gold_standard()  reads tests/gold/gold_standard.npz
"""
import os
import runpy
import shutil
import sys
import tempfile

import numpy as np

# folder of the repository (one level above this tests/ folder)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SCRIPT = os.path.join(REPO, 'Devel_SNOW_DELAY_FULL_MOULINS_AND_LAKES_MODEL_updated.py')
INPUT_DATA = os.path.join(REPO, 'input_data')
GOLD_FOLDER = os.path.join(REPO, 'tests', 'gold')

# the model imports its helper functions (fillsinks, flowdir, ...) from the
# repository folder, so that folder must be on Python's search path
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def run_model(Kic=None, TS=None, crev_width=None, input_folder=INPUT_DATA, seed=0):
    """Run the model once and return its variables (a dictionary).

    Kic, TS, crev_width : model parameters; None = the model's default value
    input_folder        : folder with the input files (default: input_data/)
    seed                : seed of the random routing over flat areas

    The model reads 'input_data/...' and writes 'results/...' relative to the
    folder it runs in. To keep your own results/ folder untouched, the model
    runs in a new temporary folder with a copy of the input files.
    """
    settings = {'ROUTEFLATS_SEED': seed}
    if Kic is not None:
        settings['Kic'] = Kic
    if TS is not None:
        settings['TS'] = TS
    if crev_width is not None:
        settings['crev_width'] = crev_width

    run_folder = tempfile.mkdtemp(prefix='model_run_')
    shutil.copytree(input_folder, os.path.join(run_folder, 'input_data'))
    os.makedirs(os.path.join(run_folder, 'results'))

    old_folder = os.getcwd()
    os.chdir(run_folder)
    try:
        results = runpy.run_path(MODEL_SCRIPT, init_globals=settings)
    finally:
        os.chdir(old_folder)
        shutil.rmtree(run_folder, ignore_errors=True)
    return results


def load_gold_standard():
    """Read the gold standard (saved by tests/make_gold_standard.py)."""
    path = os.path.join(GOLD_FOLDER, 'gold_standard.npz')
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def sensitivity_summary(results):
    """The numbers collected per run in the sensitivity analysis
    (same as the MATLAB sensitivity script)."""
    return {
        'PercentMeltToBed': float(results['percent_melt_to_bed']),
        'NumDrainedLakes': float(np.sum(results['drainage_xyz'][:, 2])),
        'NumMoulins': float(np.sum(results['moulin_xyz'][:, 2])),
    }
