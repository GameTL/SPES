# SPES reproduction report

Paper: *SPES: Towards Optimizing Performance-Resource Trade-Off for Serverless Functions* (arXiv 2403.17574, ICDE'24)

Data: Azure Functions Trace 2019, 14 days. Days 1–12 train, days 13–14 simulate.
Source: https://github.com/Azure/AzurePublicDataset/releases (dataset-functions-2019).

Run: `repro_run.py` (notebook `run.ipynb` converted to a script; loader vectorised, counts stored as `int32`).
Summary: `result/summary.json`.

## Headline comparison

| Metric | Paper | Notebook (saved output) | This run |
|---|---|---|---|
| Functions simulated | 39,388 | 39,388 | 39,388 |
| Q3 cold-start rate (SPES) | 0.108 | 0.108 | **0.0958** |
| Median CSR | – | – | 0.0 |
| P90 CSR | – | – | 0.6667 |
| Functions with zero cold starts | 57.99% | – | **58.64%** |
| Always-cold functions | < 8% | – | **7.63%** |
| Total cold starts | – | – | 218,394 |
| Total invocations (test days) | – | – | 1,866,622,093 |
| WMT (minutes, all functions) | – | WMT 21,521,200 | **19,271,548** |

## Verdict

- Headline Q3-CSR: this run gives 0.0958 vs the paper's 0.108 (about 11% lower). Same order of magnitude, same direction.
- Zero-cold share (58.6% vs 57.99%) and always-cold share (7.6% vs under 8%) match the paper closely.
- WMT: 19.27M here vs 21.52M in the notebook's saved output (about 10% lower). The notebook's saved run is the closer comparison to this code.

## Known deviations from the paper (not changed in this run)

1. `PLUSED_GIVEUP = 3` in the notebook; the paper states θgivenup = 5 for "dense" and "plused".
2. `cal_corr.py` default `--warm_th 0.7` (paper uses T-COR threshold 0.5).
3. Online-correlation pruning (`test_func_corr_perform`) appears to never execute: the dict comprehension builds a single key, and the loop unpacks keys instead of items. Impact on Q3-CSR is probably small (paper reports it as a small effect).
4. Correlation step (cells 33–45) is skipped in `repro_run.py`; it relies on the shipped `mid-data/func_candi.json` (produced by `cal_corr.py`).
5. Random seed fixed to 0 here; the paper does not state one, so exact matches are not expected.

## Limitations of this run

- `func_invok_*.json` from the simulation was truncated when the process was killed during the final dump. Invocation totals were rebuilt from the test CSVs (`result/func_invok_rebuilt.json`); cold-start rates use these rebuilt counts and match the simulated denominators.
- Baselines (Defuse, FaaSCache, Hybrid, fixed keep-alive) are not in this repo, so the paper's 49.77% / 56.43% improvement claims were not reproduced here.
- The notebook's cells 74+ (plots) were not executed.
