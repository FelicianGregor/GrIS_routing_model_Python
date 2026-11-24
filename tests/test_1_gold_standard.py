"""
TEST 1: compare one model run with the gold standard
=====================================================
Runs the model ONCE with default settings and compares the important
variables with the gold standard (tests/gold/gold_standard.npz), which was
made from the version validated against MATLAB (tag validation-v1).

If this test fails, the model results have changed. That is fine when you
changed the model on purpose - then make a new gold standard with
    python tests/make_gold_standard.py
Otherwise, something went wrong.

Run only this test:   pytest tests/test_1_gold_standard.py
"""
import numpy as np

from helpers import run_model, load_gold_standard


# How much each variable may differ from the gold standard (RELATIVE
# difference, e.g. 1e-12 = 0.0000000001 %).
#   0      -> must be exactly the same (counts, 0/1 flags, simple values)
#   1e-12  -> tiny round-off allowed (sums of many numbers can differ in the
#             last digits on another computer or NumPy version)
ALLOWED_DIFFERENCE = {
    # snow and melt, every cell and day
    'snow_total':                1e-12,
    'cumu_melt':                 1e-12,
    'pmax_filling':              1e-12,
    'delay_timer':               0,
    'daily_melt_inc_days':       1e-12,
    # crevasses, moulins, lakes, every cell and day
    'b':                         1e-12,   # water in crevasses
    'melt_to_bed':               1e-12,
    'moulin':                    0,
    'full_crevasses':            0,
    'drained_lakes':             0,
    'lake_drainage_events':      0,
    # one value per day
    'tot_melt':                  1e-12,
    'tot_tot_melt':              1e-12,
    'total_daily_melt_delivery': 1e-12,
    'melt_delivery_volume':      1e-12,
    'melt_delivery_discharge':   1e-12,
    # one value for the whole run
    'total_generated_melt':      1e-12,
    'total_generated_melt_inc':  1e-12,
    'total_melt_to_bed':         1e-12,
    'percent_melt_to_bed':       1e-12,
    'percent_melt_stored_supra': 1e-12,
    # last day
    'dem':                       0,       # filled DEM
    'As':                        1e-12,   # accumulated melt
    'LVactual':                  1e-12,   # water in lakes
    'crevasse_depth':            1e-12,
    'Ki_lake':                   1e-12,
    'RR':                        0,
    'moulin_grid':               0,
    'drainage_grid':             0,
}


def test_model_matches_gold_standard():
    results = run_model()            # one run, default settings
    gold = load_gold_standard()

    problems = []
    for name, allowed in ALLOWED_DIFFERENCE.items():
        new = np.asarray(results[name], dtype=float)
        old = np.asarray(gold[name], dtype=float)

        if new.shape != old.shape:
            problems.append(f'{name}: shape {new.shape}, gold standard {old.shape}')
            continue

        # NaN (no data) must be in the same places
        if not np.array_equal(np.isnan(new), np.isnan(old)):
            problems.append(f'{name}: NaN values are in different places')
            continue

        # compare the numbers (NaN counts as equal to NaN)
        same = np.isclose(new, old, rtol=allowed, atol=0, equal_nan=True)
        if not same.all():
            n_cells = int((~same).sum())
            biggest = np.nanmax(np.abs(new - old))
            problems.append(f'{name}: {n_cells} values differ, biggest difference {biggest:.3g}')

    # show ALL problems at once, not only the first one
    assert not problems, 'Differences to the gold standard:\n  ' + '\n  '.join(problems)
