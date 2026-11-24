# Validation test suite (Python translation vs MATLAB)

## Running

```bash
conda env update -f environment.yml      # adds pytest
pytest                                   # everything (incl. the slow seed study)
pytest -m "not slow"                     # skip the seed study (recommended day-to-day)
pytest -m "not model"                    # only the fast unit tests (< 2 s)
pytest -m slow                           # seed-spread study (3 extra model runs)
pytest tests/test_model_vs_matlab.py -v  # one file, verbose
```

The full model is run **once per pytest session** in a temporary folder
(about 1–2 minutes); your `results/` folder is never touched by the tests.

## What is tested

| Level | File | Needs | Expectation |
|---|---|---|---|
| 1 | `test_matlab_compat.py` | (MATLAB `.mat` for one test) | MATLAB `round`, linear indexing, `sortrows`, `xlsread` reproduced |
| 1 | `test_io.py` | MATLAB `.mat` | rasters, coordinates, column-major flattening identical |
| 1 | `test_routing_helpers.py` | nothing | properties of `ixneighbors`, `fillsinks`, `flowdir`, `routeflats`, `flowdistance`, `flowaccwithlakes` (mass conservation, no uphill flow, no cycles, …) |
| 1 | `test_helpers_vs_matlab_fixtures.py` | `tests/fixtures/helpers_matlab.mat` | helpers **bit-identical** to MATLAB on 6 small DEMs |
| 2 | `test_model_vs_matlab.py` | MATLAB `.mat` (standard run) | inputs, snow/melt, crevasses, lakes, moulins identical; margin routing differs only by the random choice; totals to 1e-12 |
| 2 | `test_model_invariants.py` | nothing | persistence of connections, non-negativity, lake capacity, consistent totals |
| 2 | `test_plots_smoke.py` | nothing | the three plot scripts run on Python output |
| 3 | `test_model_fixed_order_vs_matlab.py` | `MATLAB_reference_output/workspace_routing_model_matlab_fixedorder.mat` | **every** variable identical, incl. flow matrix and off-ice cells |
| 4 | `test_seed_spread.py` (slow) | MATLAB `.mat` | other routing seeds give the same on-ice results |

Tests whose MATLAB reference file is missing are **skipped**, not failed.

## Producing the extra MATLAB references (one-off, needs MATLAB)

1. In MATLAB, `cd` to the original MATLAB model folder (where you made the standard run).
2. `run('<this repo>/tests/matlab/make_matlab_references.m')`
3. Copy from the new `matlab_references_for_python/` folder:
   - `workspace_routing_model_matlab_fixedorder.mat` → `MATLAB_reference_output/`
   - `helpers_matlab.mat` → `tests/fixtures/`

The script does not modify any MATLAB file: it puts a temporary copy of
`routeflats.m` with `randperm(8)` → `1:8` first on the path and removes it again.

## Divergence report

To compare any two workspaces variable by variable (and see on which day
results first diverge):

```bash
python -c "import runpy; runpy.run_path('Devel_SNOW_DELAY_FULL_MOULINS_AND_LAKES_MODEL_updated.py', init_globals={'SAVE_FULL_WORKSPACE': True})"
python tools/compare_with_matlab.py MATLAB_reference_output/workspace_routing_model_matlab.mat results/workspace_routing_model_python.mat --csv results/comparison.csv
```

## Model switches added for testing (defaults = original behaviour)

Set via `runpy.run_path(..., init_globals={...})`, like `Kic`/`TS`/`crev_width`:

- `SAVE_FULL_WORKSPACE` (False): also save all variables to `results/workspace_routing_model_python.mat`.
- `ROUTEFLATS_FIXED_ORDER` (False): test-only deterministic neighbour order in `routeflats`.
- `ROUTEFLATS_SEED` (0): seed of the random flat routing.
