"""
TEST 2: small sensitivity analysis (slow: 4 model runs)
=======================================================
Same idea as the MATLAB sensitivity script, but with only 2 x 2 parameter
combinations. For each run it collects
    PercentMeltToBed, NumDrainedLakes, NumMoulins
and compares them with tests/gold/sensitivity_gold.csv.
It also checks two things that must hold physically.

This test is marked "slow". It is skipped by   pytest -m "not slow"
Run only this test:                            pytest tests/test_2_sensitivity.py -s
(-s shows the progress messages)

The full sensitivity analysis (all parameter values) is in run_sensitivity.py.
"""
import csv
import os
import time

import pytest

from helpers import GOLD_FOLDER, run_model, sensitivity_summary

TS_VALUES = [50000, 400000]      # tensile strength (Pa)
CREV_WIDTH_VALUES = [1, 2]       # crevasse width (m)

ALLOWED_DIFFERENCE_PERCENT = 1e-12   # relative difference allowed for PercentMeltToBed
# NumDrainedLakes and NumMoulins are counts: they must be exactly the same


def read_gold_table():
    rows = []
    with open(os.path.join(GOLD_FOLDER, 'sensitivity_gold.csv')) as f:
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items()})
    return rows


@pytest.mark.slow
def test_sensitivity_analysis():
    gold_rows = read_gold_table()
    assert len(gold_rows) == len(TS_VALUES) * len(CREV_WIDTH_VALUES), \
        'parameter values changed: make a new gold standard (tests/make_gold_standard.py)'

    start = time.time()
    table = []
    for gold in gold_rows:
        TS, crev_width = gold['TS'], gold['CrevWidth']
        print(f'\nRun {len(table) + 1} of {len(gold_rows)}: TS = {TS:.0f} Pa, crev_width = {crev_width} m')

        result = sensitivity_summary(run_model(TS=TS, crev_width=crev_width))
        result['TS'] = TS
        result['CrevWidth'] = crev_width
        table.append(result)
        print(f'  -> {result}   (elapsed {(time.time() - start) / 60:.1f} min)')

        # 1) same as the gold standard
        assert result['PercentMeltToBed'] == pytest.approx(gold['PercentMeltToBed'],
                                                           rel=ALLOWED_DIFFERENCE_PERCENT)
        assert result['NumDrainedLakes'] == gold['NumDrainedLakes']
        assert result['NumMoulins'] == gold['NumMoulins']

        # 2) physically possible
        assert 0 <= result['PercentMeltToBed'] <= 100

    # 3) higher tensile strength -> crevasses form less easily -> not MORE moulins
    for crev_width in CREV_WIDTH_VALUES:
        moulins = [row['NumMoulins'] for row in table if row['CrevWidth'] == crev_width]
        # rows are ordered by increasing TS
        assert moulins == sorted(moulins, reverse=True), \
            f'more moulins with higher tensile strength (crev_width {crev_width}): {moulins}'
