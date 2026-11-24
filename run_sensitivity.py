"""
run_sensitivity.py
------------------
Sensitivity analysis: Python version of the MATLAB sensitivity script.
Runs the model for every combination of tensile strength (TS) and crevasse
width, collects per run
    TS, CrevWidth, PercentMeltToBed, NumDrainedLakes, NumMoulins
and saves the table after every run to results/sensitivity_results.csv
(so nothing is lost if the analysis is interrupted).

Run from the repository folder:
    python run_sensitivity.py
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests'))
from helpers import run_model, sensitivity_summary   # noqa: E402

# Kic_storage = [100000, 200000, 400000]           # fracture toughness (not varied)
TS_storage = [1000 * x for x in [25, 50, 100, 250, 400, 600, 800]]   # tensile strength (Pa)
crev_width_storage = [0.5, 1, 2, 4]                                  # crevasse width (m)

OUTPUT = os.path.join('results', 'sensitivity_results.csv')
COLUMNS = ['TS', 'CrevWidth', 'PercentMeltToBed', 'NumDrainedLakes', 'NumMoulins']


def main():
    os.makedirs('results', exist_ok=True)
    n_total = len(TS_storage) * len(crev_width_storage)
    results = []
    run_number = 0
    start = time.time()

    for TS in TS_storage:
        for crev_width in crev_width_storage:
            run_number += 1
            print()
            print('========================================')
            print('Run %d of %d' % (run_number, n_total))
            print('TS         = %.0f Pa' % TS)
            print('Crev width = %.2f m' % crev_width)
            print('========================================')

            # run model
            row = {'TS': TS, 'CrevWidth': crev_width}
            row.update(sensitivity_summary(run_model(TS=TS, crev_width=crev_width)))
            results.append(row)

            # timing information
            elapsed = time.time() - start
            remaining = elapsed / run_number * (n_total - run_number)
            print('Elapsed: %.1f min' % (elapsed / 60))
            print('Estimated remaining: %.1f min' % (remaining / 60))

            # save intermediate results
            with open(OUTPUT, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(results)

    print('\nSensitivity analysis complete. Results in', OUTPUT)
    print('Total runtime = %.2f hours' % ((time.time() - start) / 3600))


if __name__ == '__main__':
    main()
