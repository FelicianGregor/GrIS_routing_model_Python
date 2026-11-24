"""
make_gold_standard.py
---------------------
Creates the "gold standard" that the tests compare against:

  tests/gold/gold_standard.npz     key outputs of one run with default settings
  tests/gold/sensitivity_gold.csv  summary of the small sensitivity analysis

ONLY run this when you have deliberately accepted a change in the model
results (and say so in your git commit message). The first gold standard was
made from tag validation-v1, the version validated against MATLAB.

Run from the repository folder:
    python tests/make_gold_standard.py
"""
import csv
import datetime
import os
import subprocess

import numpy as np

from helpers import GOLD_FOLDER, REPO, run_model, sensitivity_summary
from test_2_sensitivity import TS_VALUES, CREV_WIDTH_VALUES

# Variables saved in the gold standard (and compared in test_1_gold_standard.py)
GOLD_VARIABLES = [
    # every cell, every day (cells x days)
    'snow_total', 'cumu_melt', 'pmax_filling', 'delay_timer', 'daily_melt_inc_days',
    'b', 'melt_to_bed', 'moulin', 'full_crevasses', 'drained_lakes', 'lake_drainage_events',
    # one value per day
    'tot_melt', 'tot_tot_melt', 'total_daily_melt_delivery',
    'melt_delivery_volume', 'melt_delivery_discharge',
    # one value for the whole run
    'total_generated_melt', 'total_generated_melt_inc', 'total_melt_to_bed',
    'percent_melt_to_bed', 'percent_melt_stored_supra',
    # grids / vectors of the last day
    'dem', 'As', 'LVactual', 'crevasse_depth', 'Ki_lake', 'RR',
    'moulin_grid', 'drainage_grid',
]


def git_version():
    try:
        return subprocess.check_output(['git', 'describe', '--tags', '--always', '--dirty'],
                                       cwd=REPO, text=True).strip()
    except Exception:
        return 'unknown'


def main():
    os.makedirs(GOLD_FOLDER, exist_ok=True)

    # 1) one run with default settings
    print('Running the model with default settings ...')
    results = run_model()
    gold = {name: np.asarray(results[name]) for name in GOLD_VARIABLES}
    gold['made_from_version'] = np.array(git_version())
    gold['made_on'] = np.array(datetime.date.today().isoformat())
    path = os.path.join(GOLD_FOLDER, 'gold_standard.npz')
    np.savez_compressed(path, **gold)
    print('saved', path, '(%.1f MB)' % (os.path.getsize(path) / 1e6))

    # 2) small sensitivity analysis
    path = os.path.join(GOLD_FOLDER, 'sensitivity_gold.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TS', 'CrevWidth', 'PercentMeltToBed', 'NumDrainedLakes', 'NumMoulins'])
        for TS in TS_VALUES:
            for crev_width in CREV_WIDTH_VALUES:
                print('Running TS = %g Pa, crev_width = %g m ...' % (TS, crev_width))
                s = sensitivity_summary(run_model(TS=TS, crev_width=crev_width))
                writer.writerow([TS, crev_width, repr(s['PercentMeltToBed']),
                                 s['NumDrainedLakes'], s['NumMoulins']])
    print('saved', path)


if __name__ == '__main__':
    main()
