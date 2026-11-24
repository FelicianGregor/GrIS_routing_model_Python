# Tests

Three test scripts check that the model still gives the same, sensible results.

| File | What it does | Model runs |
|---|---|---|
| `test_1_gold_standard.py` | One run with default settings, compared with the gold standard (29 variables, each with its own allowed difference) | 1 full run |
| `test_2_sensitivity.py` | Small sensitivity analysis (TS × crevasse width, 2 × 2), compared with the gold standard; marked **slow** | 4 full runs |
| `test_3_synthetic_dem.py` | The model on three small artificial glaciers: it must run and give sensible results | 3 small runs |

## How to run

From the repository folder, in the `routing_model` conda environment:

```
pytest                         # all tests
pytest -m "not slow"           # all except the sensitivity test
pytest tests/test_1_gold_standard.py   # just one file
pytest -s                      # also show the progress messages
```

`.` = passed, `F` = failed, `s` = skipped. If a test fails, pytest prints which
variables differ and by how much.

The model always runs in a temporary folder, so your `results/` folder is not changed.

## The gold standard

`gold/gold_standard.npz` and `gold/sensitivity_gold.csv` hold the results of
the version validated against MATLAB (tag `validation-v1` on GitHub).

**When you change the model on purpose** and the results change, the tests
will fail. Check the change is what you wanted, then make a new gold standard:

```
python tests/make_gold_standard.py
```

and mention it in the git commit message.

## Other files here

- `helpers.py`: `run_model(...)` (runs the model in a temporary folder) and `load_gold_standard()`
- `make_gold_standard.py`: creates the gold-standard files

The full validation against MATLAB (about 300 tests, MATLAB reference files)
is kept in the GitHub release `validation-v1`.
