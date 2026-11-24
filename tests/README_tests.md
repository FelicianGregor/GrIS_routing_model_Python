# Running the tests

## How to run

In the Anaconda Prompt, from the repository folder:

```
conda activate routing_model
pytest -m "not slow" -v                  # tests 1 and 3 (normal check)
pytest -v -s                             # all tests, incl. the slow test 2
pytest tests/test_1_gold_standard.py -v  # only one test file
```

`-v` = one line per test (PASSED / FAILED), `-s` = show the model's progress
messages, `Ctrl + C` = stop. The last line shows the summary, e.g. `4 passed`.

## What each test does

| Test | What it checks | Model runs |
|---|---|---|
| `test_1_gold_standard.py` | Runs the model once (default settings) and compares 29 key variables with the gold standard. Allowed differences are set per variable at the top of the file. | 1 full run |
| `test_2_sensitivity.py` (**slow**) | Small sensitivity analysis (TS = 50 / 400 kPa × crevasse width = 1 / 2 m). Compares % melt to bed, drained lakes and moulins with the gold standard; checks the % is 0–100 and that higher TS never gives more moulins. | 4 full runs |
| `test_3_synthetic_dem.py` | Runs the model on 3 small artificial glaciers (slope; slope with flat area and lake; bumpy slope). Checks: no NaN, no negative water, moulins only on the ice. | 3 small runs |

## Gold standard

`gold/` holds the results of the version validated against MATLAB (tag `validation-v1`).
If you change the model **on purpose** and the results change, tests 1 and 2 fail.
Then check the change is intended and make a new gold standard:

```
python tests/make_gold_standard.py
```

(and mention it in the git commit message).
